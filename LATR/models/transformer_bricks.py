import numpy as np
import math
import warnings

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init, constant_init)
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction

from .scatter_utils import scatter_mean
from .utils import inverse_sigmoid

@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False

        if not (dim_per_head & (dim_per_head - 1) == 0 and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dims in MultiScaleDeformAttention to make "
                'the dimension of each attention head a power of 2 for efficiency.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 1, 2).repeat(1, 1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[..., i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        if value is None:
            value = key
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if key_pos is not None:
            value = value + key_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2) #[1, 800, 4, 1, 8, 2]
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points) ##[1, 800, 4, 1, 8, 2]

        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  # [W, H]
        sampling_locations = reference_points.view(
            bs, num_query, 1, 1, 1, 2) + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)
        output = self.dropout(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output

@TRANSFORMER_LAYER.register_module()
class LATRDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(LATRDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        query = super().forward(
            query=query, key=key, value=value,
            query_pos=query_pos, key_pos=key_pos,
            attn_masks=attn_masks, query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask, **kwargs)
        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LATRTransformerDecoder(TransformerLayerSequence):
    def __init__(self,
                 *args, embed_dims=None,
                 post_norm_cfg=dict(type='LN'),
                 num_query=None,
                 num_anchor_per_query=None,
                 anchor_y_steps=None,
                 **kwargs):
        super(LATRTransformerDecoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

        self.num_query = num_query
        self.num_anchor_per_query = num_anchor_per_query
        self.anchor_y_steps = anchor_y_steps
        self.num_points_per_anchor = len(anchor_y_steps) // num_anchor_per_query
        self.embed_dims = embed_dims

    def forward(self, query, key, value,
                reference_points=None,
                key_pos=None,
                reg_branches=None, cls_branches=None,
                sin_embed=None, query_pos=None,
                **kwargs):

        batch_size = query.shape[1]
        intermediate = []
        outputs_classes = []
        outputs_coords = []

        for layer_idx, layer in enumerate(self.layers):
            # 构造 key_pos（BEV 模式下直接使用 sin_embed）
            key_pos = sin_embed.flatten(2, 3).permute(2, 0, 1).contiguous()

            query = layer(
                query=query, key=key, value=value,
                key_pos=key_pos,
                reference_points=reference_points,
                query_pos=query_pos,
                **kwargs
            )

            # 保存中间结果
            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)

            # 回归与分类分支
            query = query.permute(1, 0, 2)  # [bs, num_query, dim]
            tmp = reg_branches[layer_idx](query)  # [bs, num_query, 3]

            bs = tmp.shape[0]
            tmp = tmp.view(bs, self.num_query, self.num_anchor_per_query, -1, 3)

            reference_points = reference_points.view(
                bs, self.num_query, self.num_anchor_per_query,
                self.num_points_per_anchor, 2
            )

            reference_points = inverse_sigmoid(reference_points)
            new_reference_points = torch.stack([
                reference_points[..., 0] + tmp[..., 0],
                reference_points[..., 1] + tmp[..., 1],
            ], dim=-1).sigmoid()

            cls_feat = query.view(bs, self.num_query, self.num_anchor_per_query, -1)
            cls_feat = torch.max(cls_feat, dim=2)[0]
            outputs_class = cls_branches[layer_idx](cls_feat)

            outputs_classes.append(outputs_class)
            outputs_coords.append(torch.cat([new_reference_points, tmp[..., -1:]], dim=-1))

            # 更新 query/ref point
            reference_points = new_reference_points.view(
                bs, self.num_query * self.num_anchor_per_query,
                self.num_points_per_anchor * 2
            ).detach()
            query = query.permute(1, 0, 2)  # [num_query, bs, dim]

        return torch.stack(intermediate), outputs_classes, outputs_coords

@TRANSFORMER.register_module()
class LATRTransformer(BaseModule):
    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(LATRTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.init_weights()

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query,
                query_embed, pos_embed,
                reference_points=None,
                reg_branches=None, cls_branches=None,
                spatial_shapes=None,
                level_start_index=None,
                mlvl_masks=None,
                mlvl_positional_encodings=None,
                pos_embed2d=None,
                **kwargs):
        memory = x
        query_embed = query_embed.permute(1, 0, 2)
        target = query.permute(1, 0, 2)

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec, outputs_classes, outputs_coords = \
            self.decoder(
                query=target,
                key=memory,
                value=memory,
                key_pos=pos_embed,
                query_pos=query_embed,
                key_padding_mask=mask.astype(torch.bool) if mask is not None else None,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs
            )
        return out_dec.permute(0, 2, 1, 3), outputs_classes, outputs_coords
