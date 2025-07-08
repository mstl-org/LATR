import numpy as np
from mmcv.utils import Config

_base_ = [
    '../_base_/base_res101_bs16xep100.py',  # 可能需要调整为你的基础配置
    '../_base_/optimizer.py'
]

mod = 'release_iccv/fusionlane_300_baseline_lite'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

dataset = '300'
dataset_dir = './data/openlane/data_final/'
data_dir = './data/openlane/lane3d_300/'
train_pkl = '/home/dfz/Project/LATR/data/openlane/data_final/waymo_infos_train_new.pkl'
val_pkl = '/home/dfz/Project/LATR/data/openlane/data_final/waymo_infos_validation_new.pkl'
vis_dir = '/media/home/data_share/OpenLane/waymo/data/results/data_final/vis_results/'

batch_size = 1
nworkers = 10
num_category = 21
pos_threshold = 0.3 #0.3

clip_grad_norm = 20

top_view_region = np.array([
    [-10, 103], [10, 103], [-10, 3], [10, 3]]) # 定义了鸟瞰图的前后左右边界
enlarge_length = 20 #扩展 top view 范围
position_range = [
    top_view_region[0][0] - enlarge_length,
    top_view_region[2][1] - enlarge_length,
    -5,
    top_view_region[1][0] + enlarge_length,
    top_view_region[0][1] + enlarge_length,
    5.]
anchor_y_steps = np.linspace(3, 103, 20)
num_y_steps = len(anchor_y_steps)
cls_cost_weight = 5.0

# extra aug
photo_aug = dict(
    brightness_delta=32 // 2,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=9)

_dim_ = 256
num_query = 40
num_pt_per_line = 20

latr_cfg = dict(
    fpn_dim=_dim_,
    num_query=num_query,
    num_group=1,
    sparse_num_group=4,
    encoder=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
    ),
    neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=_dim_,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=_dim_,
        out_channels=80,
        image_size=[360, 480],
        feature_size=[45, 60],
        xbound=[6.0, 78.0, 0.2],
        ybound=[-36.0, 36.0, 0.2],
        zbound=[-3.0, 6.0, 9.0],
        dbound=[1.0, 80.0, 0.5],
        downsample=2
    ),
    head=dict(
        num_pt_per_line=num_pt_per_line,
        xs_loss_weight=2.0,
        zs_loss_weight=10.0,
        vis_loss_weight=1.0,
        cls_loss_weight=10,
    ),
    trans_params=dict(init_z=0, bev_h=180, bev_w=180),  # 根据 xbound, ybound 调整
)

transformer=dict(
    type='LATRTransformer',
    decoder=dict(
        type='LATRTransformerDecoder',
        embed_dims=_dim_,
        num_layers=2,
        num_query=num_query,
        num_anchor_per_query=num_pt_per_line,
        anchor_y_steps=anchor_y_steps,
        transformerlayers=dict(
            type='LATRDecoderLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=_dim_,
                    num_heads=4,
                    dropout=0.1),
                dict(
                    type='MSDeformableAttention3D',
                    embed_dims=_dim_,
                    num_heads=4,
                    num_levels=1,
                    num_points=8,
                    batch_first=False,
                    dropout=0.1),
                ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=_dim_,
                feedforward_channels=_dim_*8,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            feedforward_channels=_dim_ * 8,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')),
))

resize_h = 360  # 与 image_size 匹配
resize_w = 480

nepochs = 24
eval_freq = 1
optimizer_cfg = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
