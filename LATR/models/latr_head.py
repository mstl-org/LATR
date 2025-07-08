import numpy as np
import math
import cv2
import time
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.init import normal_

from mmcv.cnn import bias_init_with_prob
from mmdet.models.builder import build_loss
from mmdet.models.utils import build_transformer
from mmdet.core import multi_apply

from mmcv.utils import Config
from .utils import inverse_sigmoid
from .transformer_bricks import *

from scipy.optimize import linear_sum_assignment

class SineBEVPositionalEncoding(nn.Module):
    def __init__(self, num_feats, xbound, ybound, downsample=2, bev_w=180, bev_h=180):
        super().__init__()
        self.num_feats = num_feats
        self.xbound = [xbound[0], xbound[0] + xbound[2] * downsample * (bev_w - 1), xbound[2] * downsample]
        self.ybound = [ybound[0], ybound[0] + ybound[2] * downsample * (bev_h - 1), ybound[2] * downsample]

    def forward(self, mask):
        B, H, W = mask.shape
        dtype = mask.dtype
        device = mask.device

        y_range = torch.linspace(self.ybound[0], self.ybound[1], H, dtype=dtype, device=device)# [-36.0, 35.6]
        x_range = torch.linspace(self.xbound[0], self.xbound[1], W, dtype=dtype, device=device)# [6.0, 77.6]
        y_embed, x_embed = torch.meshgrid(y_range, x_range, indexing='ij')

        y_embed = y_embed.unsqueeze(0).expand(B, -1, -1)
        x_embed = x_embed.unsqueeze(0).expand(B, -1, -1)

        dim_t = torch.arange(self.num_feats, dtype=dtype, device=device)
        dim_t = 10000 ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # B, C, H, W
        return pos

class LATRHead(nn.Module):
    def __init__(self, args,
                 num_group=1,
                 num_classes=21,
                 num_query=40,
                 embed_dims=256,
                 transformer=None, 
                 num_reg_fcs=2,
                 depth_num=50,
                 depth_start=3,
                 top_view_region=[[-10, 103], [10, 103], [-10, 3], [10, 3]],
                 position_range=[-30, -17, -5, 30, 123, 5],
                 xbound=[6.0, 78.0, 0.2],
                 ybound=[-36.0, 36.0, 0.2],
                 pred_dim=20,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=10.0),
                 loss_reg=dict(type='L1Loss', loss_weight=2.0),
                 loss_vis=dict(type='BCEWithLogitsLoss', reduction='mean'),
                 xs_loss_weight=2.0,
                 zs_loss_weight=10.0,
                 vis_loss_weight=1.0,
                 cls_loss_weight=10,
                 project_loss_weight=1.0,
                 trans_params=dict(
                     init_z=0, bev_h=180, bev_w=180),
                 pt_as_query=False,
                 num_pt_per_line=20,
                 num_feature_levels=1,
                 gt_project_h=20,
                 gt_project_w=30,
                 project_crit=dict(
                     type='SmoothL1Loss',
                     reduction='none'),
                 ):
        super().__init__()
        self.trans_params = dict(
            top_view_region=top_view_region,
            z_region=[position_range[2], position_range[5]])
        self.trans_params.update(trans_params)
        self.gt_project_h = gt_project_h
        self.gt_project_w = gt_project_w

        self.num_y_steps = args.num_y_steps #20
        self.cls_cost_weight = args.cls_cost_weight #5.0
        self.register_buffer('anchor_y_steps',
            torch.from_numpy(args.anchor_y_steps).float())
        self.register_buffer('anchor_y_steps_dense',
            torch.from_numpy(args.anchor_y_steps_dense).float())

        project_crit['reduction'] = 'none'
        self.project_crit = getattr(
            nn, project_crit.pop('type'))(**project_crit)

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        # points num along y-axis.
        self.code_size = pred_dim #20
        self.num_query = num_query #40
        self.num_group = num_group #1
        self.num_pred = transformer['decoder']['num_layers'] #2
        self.pc_range = position_range
        self.xs_loss_weight = xs_loss_weight
        self.zs_loss_weight = zs_loss_weight
        self.vis_loss_weight = vis_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.project_loss_weight = project_loss_weight

        loss_reg['reduction'] = 'none'
        self.reg_crit = build_loss(loss_reg)
        self.cls_crit = build_loss(loss_cls)
        self.bce_loss = build_nn_loss(loss_vis)

        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.depth_start = depth_start

        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.positional_encoding = SineBEVPositionalEncoding(
            num_feats=128,
            xbound=xbound,
            ybound=ybound
        )

        self.transformer = build_transformer(transformer)
        self.query_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 从整个 BEV 上池化为全局特征 [B, 256, 1, 1]
            nn.Conv2d(embed_dims, embed_dims, kernel_size=1),  # [B, 256, 1, 1]
            nn.ReLU(),
            nn.Flatten(start_dim=1),  # [B, 256]
            nn.Linear(embed_dims, self.num_group * self.num_query * num_pt_per_line * embed_dims)
        )

        # build pred layer: cls, reg, vis
        self.num_reg_fcs = num_reg_fcs
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(
            nn.Linear(
                self.embed_dims,
                3 * self.code_size // num_pt_per_line)) # 3 * 20 = 60
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.num_pt_per_line = num_pt_per_line #20
        self.point_embedding = nn.Embedding(
            self.num_pt_per_line, self.embed_dims)
        self.num_feature_levels = num_feature_levels
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))

        self._init_weights()

    def _init_weights(self):
        self.transformer.init_weights()
        if self.cls_crit.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        normal_(self.level_embeds)

    def forward(self, input_dict, is_training=True):
        output_dict = {}
        # 获取 BEV 特征
        bev_feats = input_dict['fusion_features']  # [2, 256, 180, 180],图像和点云bev特征融合在lidar坐标系，分辨率为0.4m×0.4m
        if not isinstance(bev_feats, (list, tuple)):
            bev_feats = [bev_feats]

        # 动态生成查询
        B, C, H, W = bev_feats[0].shape
        device = bev_feats[0].device

        query_embeds = self.query_generator(bev_feats[0])
        query_embeds = query_embeds.view(B, self.num_group * self.num_query * self.num_pt_per_line, C)
        query = query_embeds.clone()  # 非全 0，增强 early expression

        # 添加位置编码
        masks = bev_feats[0].new_zeros((B, H, W)) # [2, 180, 180]
        sin_embed = self.positional_encoding(masks)  # [2, 256, 180, 180]

        # 准备 Transformer 输入
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(bev_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(2, 0, 1)  # [32400, 2, 256]
            feat = feat + self.level_embeds[None, lvl:lvl+1, :].to(feat.device)
            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)
        feat_flatten = torch.cat(feat_flatten, 0)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_feats[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        ref_x = torch.linspace(self.pc_range[0], self.pc_range[3], steps=self.num_query, device=device)  # [-30, 30]
        ref_z = torch.linspace(self.pc_range[2], self.pc_range[5], steps=self.num_group, device=device)  # [-5, 5]
        ref_x_norm = (ref_x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        ref_z_norm = (ref_z - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        ref_grid = torch.stack(torch.meshgrid(ref_x_norm, ref_z_norm, indexing='ij'), dim=-1).reshape(-1, 2)  # [40, 2]
        ref_grid = ref_grid.unsqueeze(1).repeat(1, self.num_pt_per_line, 1).reshape(-1, 2)  # [800, 2]
        ref_points = ref_grid.unsqueeze(0).repeat(B, 1, 1)  # [2, 800, 2]
        ref_points = ref_points.detach()  # 防止误梯度传播

        pos_embed = None
        pos_embed2d = None
        
        # Transformer 解码
        outs_dec, outputs_classes, outputs_coords = self.transformer(
            feat_flatten, None, #[32400,2,256]         
            query, query_embeds, pos_embed,  #[2,800,256],[2,800,256],None
            reference_points=ref_points, #[2,800,2]
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            img_feats=bev_feats[0], #[2,256,180,180]
            lidar2img=input_dict['lidar2img'],#[2,3,4]
            pad_shape=input_dict['pad_shape'],#(360,480)
            sin_embed=sin_embed, #[2,256,180,180]          
            spatial_shapes=spatial_shapes, #[180,180]
            level_start_index=level_start_index, #0
            pos_embed2d=pos_embed2d, #None
            image=input_dict['image'], #[2,3,360,480]
            **self.trans_params
        )

        # 解码器输出处理
        all_cls_scores = torch.stack(outputs_classes) #(2,2,40,21)
        all_line_preds = torch.stack(outputs_coords) #(2,2,40,20,1,3)
        all_line_preds[..., 0] = (all_line_preds[..., 0]
            * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_line_preds[..., 1] = (all_line_preds[..., 1]
            * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
        all_line_preds[..., 2] = torch.sigmoid(all_line_preds[..., 2])  # vis

        # reshape to original format
        all_line_preds = all_line_preds.view(
            len(outputs_classes), bs, self.num_query,
            self.transformer.decoder.num_anchor_per_query,
            self.transformer.decoder.num_points_per_anchor, 2 + 1 # xz+vis
        )
        all_line_preds = all_line_preds.permute(0, 1, 2, 5, 3, 4)
        all_line_preds = all_line_preds.flatten(3, 5)

        output_dict.update({
            'all_cls_scores': all_cls_scores,
            'all_line_preds': all_line_preds,
        })

        if is_training:
            losses = self.get_loss(output_dict, input_dict)
            output_dict.update(losses)

        return output_dict

    def get_loss(self, output_dict, input_dict):
        all_cls_pred = output_dict['all_cls_scores']  # [num_layers, B, num_query, num_classes]
        all_lane_pred = output_dict['all_line_preds']  # [num_layers, B, num_query, 3*num_pt_per_line]
        gt_lanes = input_dict['ground_lanes']  # [B, max_lanes, 3*num_y_steps+num_classes]

        num_layers = all_lane_pred.shape[0]

        def single_layer_loss(layer_idx):
            cls_pred = all_cls_pred[layer_idx]  # [B, num_query, num_classes]
            lane_pred = all_lane_pred[layer_idx]  # [B, num_query, 3*num_pt_per_line]

            pred_xs = lane_pred[:, :, :self.num_pt_per_line]
            pred_zs = lane_pred[:, :, self.num_pt_per_line:2*self.num_pt_per_line]
            pred_vis = lane_pred[:, :, 2*self.num_pt_per_line:]

            per_xs_loss = 0.0
            per_zs_loss = 0.0
            per_vis_loss = 0.0
            per_cls_loss = 0.0

            for b_idx in range(cls_pred.shape[0]):
                cls_pred_b = cls_pred[b_idx]
                pred_xs_b = pred_xs[b_idx]
                pred_zs_b = pred_zs[b_idx]
                pred_vis_b = pred_vis[b_idx]

                gt_lane = gt_lanes[b_idx]
                gt_xs = gt_lane[:, :self.num_y_steps]
                gt_zs = gt_lane[:, self.num_y_steps:2*self.num_y_steps]
                gt_vis = gt_lane[:, 2*self.num_y_steps:3*self.num_y_steps]
                gt_cls = gt_lane[:, 3*self.num_y_steps:]

                valid_gt = gt_vis.sum(dim=1) > 0
                if valid_gt.sum() == 0:
                    cls_target = cls_pred_b.new_zeros(cls_pred_b.shape[0]).long()
                    cls_loss = self.cls_crit(cls_pred_b, cls_target)
                    per_cls_loss += cls_loss
                    continue

                gt_xs = gt_xs[valid_gt]
                gt_zs = gt_zs[valid_gt]
                gt_vis = gt_vis[valid_gt]
                gt_cls = torch.argmax(gt_cls[valid_gt], dim=1)

                N_gt = gt_xs.shape[0]
                N_pred = cls_pred_b.shape[0]

                # Compute cost matrix: [N_gt, N_pred]
                with torch.no_grad():
                    # L1 position cost
                    pos_cost = (torch.cdist(gt_xs, pred_xs_b, p=1) + torch.cdist(gt_zs, pred_zs_b, p=1)) / self.num_y_steps

                    # visibility cost: BCE between [N_gt, pt] & [N_pred, pt] → [N_gt, N_pred]
                    vis_cost = F.binary_cross_entropy(
                        pred_vis_b.unsqueeze(0).expand(N_gt, -1, -1),
                        gt_vis.unsqueeze(1).expand(-1, N_pred, -1),
                        reduction='none').mean(dim=2)

                    # classification cost: focal loss proxy (neg dot product)
                    cls_prob = cls_pred_b.sigmoid()  # [N_pred, num_cls]
                    cls_cost = -cls_prob[:, gt_cls].T  # [N_gt, N_pred]

                    # total cost
                    cost_matrix = (
                        self.xs_loss_weight * pos_cost +
                        self.vis_loss_weight * vis_cost +
                        self.cls_cost_weight * cls_cost
                    ).cpu()

                    matched_gt_idx, matched_pred_idx = linear_sum_assignment(cost_matrix)

                # get matched
                pred_xs_b = pred_xs_b[matched_pred_idx]
                pred_zs_b = pred_zs_b[matched_pred_idx]
                pred_vis_b = pred_vis_b[matched_pred_idx]
                cls_pred_b = cls_pred_b[matched_pred_idx]

                gt_xs = gt_xs[matched_gt_idx]
                gt_zs = gt_zs[matched_gt_idx]
                gt_vis = gt_vis[matched_gt_idx]
                gt_cls = gt_cls[matched_gt_idx]

                # loss计算
                loc_mask = gt_vis > 0
                xs_loss = self.reg_crit(pred_xs_b, gt_xs)
                zs_loss = self.reg_crit(pred_zs_b, gt_zs)
                xs_loss = (xs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), min=1)
                zs_loss = (zs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), min=1)
                vis_loss = self.bce_loss(pred_vis_b, gt_vis)
                cls_loss = self.cls_crit(cls_pred_b, gt_cls)

                per_xs_loss += xs_loss
                per_zs_loss += zs_loss
                per_vis_loss += vis_loss
                per_cls_loss += cls_loss

            return tuple(map(lambda x: x / cls_pred.shape[0],
                            [per_xs_loss, per_zs_loss, per_vis_loss, per_cls_loss]))

        all_xs_loss, all_zs_loss, all_vis_loss, all_cls_loss = multi_apply(single_layer_loss, range(num_layers))
        all_xs_loss = sum(all_xs_loss) / num_layers
        all_zs_loss = sum(all_zs_loss) / num_layers
        all_vis_loss = sum(all_vis_loss) / num_layers
        all_cls_loss = sum(all_cls_loss) / num_layers

        return dict(
            all_xs_loss=self.xs_loss_weight * all_xs_loss,
            all_zs_loss=self.zs_loss_weight * all_zs_loss,
            all_vis_loss=self.vis_loss_weight * all_vis_loss,
            all_cls_loss=self.cls_loss_weight * all_cls_loss,
        )

def build_nn_loss(loss_cfg):
    crit_t = loss_cfg.pop('type')
    return getattr(nn, crit_t)(**loss_cfg)
