import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from mmdet.core import multi_apply
from mmdet.models.builder import build_loss
from mmcv.utils import Config
from .latr_head import LATRHead
from .backbone import ImageBackbone, PointCloudBackbone
from .neck import ImageNeck, ViewTransform
from .fusion import FusionLayer
from .head import LaneDetectionHead3D
from scipy.optimize import linear_sum_assignment

from PIL import Image
import matplotlib.pyplot as plt

class LATR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # no centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        self.num_pt_per_line = args.num_pt_per_line
        _dim_ = args.latr_cfg.fpn_dim
        num_query = args.latr_cfg.num_query
        num_group = args.latr_cfg.num_group
        self.num_query = num_query

        # 图像主干
        self.encoder = ImageBackbone()
        self.neck = ImageNeck()
        self.view_transform = ViewTransform()
        self.encoder.init_weights()

        # 点云主干
        self.pts_backbone = PointCloudBackbone()

        self.fusion_layer = FusionLayer()

        # 3D车道线解码头
        self.head = LATRHead(
            args=args,
            num_group=num_group,
            position_range=args.position_range,
            top_view_region=args.top_view_region,
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=_dim_// 2, normalize=True),
            num_query=num_query,
            pred_dim=self.num_y_steps,
            num_classes=args.num_category,
            embed_dims=_dim_,
            transformer=args.transformer,
            **args.latr_cfg.get('head', {}),
            trans_params=args.latr_cfg.get('trans_params', {})
        )

    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_backbone.pts_voxel_layer(res)
            f, c = ret if len(ret) == 2 else ret[:2]
            n = ret[2] if len(ret) == 3 else None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)
        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if sizes:
            sizes = torch.cat(sizes, dim=0)
            if self.pts_backbone.voxelize_reduce:
                feats = feats.sum(dim=1) / sizes.type_as(feats).view(-1, 1)
        return feats, coords, sizes

    def forward(self, image, point_cloud, _M_inv=None, is_training=True, extra_dict=None):
        # 图像 BEV
        out_featList = self.encoder.swin(image)
        neck_out = self.neck.neck(out_featList)
        neck_out = neck_out[0] #[2,256,45,60]
        img_bev_out = self.view_transform.transform(
            neck_out,
            point_cloud,
            extra_dict['waymo_lidar2image'],
            extra_dict['waymo_intrinsics'],
            extra_dict['waymo_camera2lidar']
        ) #[2,80,180,180]

        # 点云 BEV
        points = [p.squeeze(0) for p in point_cloud]
        feats, coords, sizes = self.voxelize(points) #[12000,4] [12000,3] [12000]
        point_bev_out = self.pts_backbone.encoder(feats, coords, coords[-1, 0] + 1) #[2,256,180,180]

        fusion_features = self.fusion_layer.fuser([img_bev_out, point_bev_out])
        
        # 送入 Head
        output = self.head(
            dict(
                fusion_features = fusion_features, #[2, 256, 180, 180]融合bev特征
                lidar2img=extra_dict['waymo_lidar2image'], #[2, 4, 4]
                pad_shape=extra_dict['pad_shape'], #(360,480)原始分辨率
                ground_lanes=extra_dict['ground_lanes'] if is_training else None, #[2,20,81]真实稀疏车道线
                image=image, #[2, 3, 360, 480]原始前视图
            ),
            is_training=is_training,
        )
        return output