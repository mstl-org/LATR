import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from scipy.interpolate import UnivariateSpline

# 文件夹路径
base_dir = '/media/home/data_share/OpenLane/waymo/data/results/data_final/vis_results_raw_latr/'

# 递归读取所有 JSON 文件
json_files = sorted(glob.glob(os.path.join(base_dir, '**', '*.json'), recursive=True))

# 按文件夹分组并排序
file_dict = {}
for file_path in json_files:
    # 获取相对路径并提取文件夹部分
    rel_path = os.path.relpath(file_path, base_dir)
    folder = os.path.dirname(rel_path)
    
    if folder not in file_dict:
        file_dict[folder] = []
    file_dict[folder].append(file_path)

# 将字典转换为嵌套列表，文件夹和文件都排序
folders = sorted(file_dict.keys())
all_files = [sorted(file_dict[folder]) for folder in folders]

json_file = all_files[6][10]  # 你可以根据需要修改索引
print(f"\nLoading file: {json_file}")

with open(json_file, 'r') as file:
    data = json.load(file)

# 提取车道线数据
gt_lanes = data['gt_lanes']
pre_lanes = data['pred_lanes']

# 创建 3D 图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Lane Lines (GT vs Predicted)")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# 固定坐标轴范围
ax.set_xlim(-36, 36)  # X轴范围：-10到10
ax.set_ylim(3, 103)   # Y轴范围：3到103
ax.set_zlim(-3, 6)    # Z轴范围：-3到3

# 定义颜色数组（注释掉，改为根据 category 决定颜色）
# colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'yellow']

# 为 21 个类别 (0-20) 定义颜色
category_colors = {
    0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple',
    5: 'cyan', 6: 'yellow', 7: 'magenta', 8: 'gray', 9: 'brown',
    10: 'pink', 11: 'teal', 12: 'lime', 13: 'indigo', 14: 'violet',
    15: 'maroon', 16: 'navy', 17: 'olive', 18: 'silver', 19: 'gold',
    20: 'black'  # 确保覆盖 0-20 的 21 个类别
}

# 绘制地面真值车道线 (gt_lanes) - 实线
for lane_idx, lane in enumerate(gt_lanes):
    # 假设 lane 是 [x, z, visibility, category] 的列表
    x = np.array(lane['x'])  # 提取 x 坐标 (20 个值)
    z = np.array(lane['z'])  # 提取 z 坐标 (20 个值)
    vis = np.array(lane['visibility']) > 0.5  # 提取可见性掩码 (20 个值)
    category = lane['category']  # 提取类别
    y = np.linspace(3, 103, 100)  # y 坐标

    # 应用可见性掩码
    x = x[vis]
    y = y[vis]
    z = z[vis]

    # 根据 category 决定颜色，默认为 'gray' 如果 category 未定义
    color = category_colors.get(category, 'gray')

    # 绘制实线表示地面真值
    ax.plot(x, y, z, 
            color=color, 
            linestyle='-',  # 实线
            label=f"GT{lane_idx}", 
            linewidth=2)

# 绘制预测车道线 (pre_lanes) - 虚线
for lane_idx, lane in enumerate(pre_lanes):
    # 假设 lane 是 [x, z, visibility, category] 的列表
    x = np.array(lane['x'])  # 提取 x 坐标 (20 个值)
    z = np.array(lane['z'])  # 提取 z 坐标 (20 个值)
    vis = np.array(lane['visibility']) > 0.5  # 提取可见性掩码 (20 个值)
    category = lane['category']  # 提取类别
    y = np.linspace(3, 103, 100)  # y 坐标

    # 应用可见性掩码
    x = x[vis]
    y = y[vis]
    z = z[vis]

    # 根据 category 决定颜色，默认为 'gray' 如果 category 未定义
    color = category_colors.get(category, 'gray')

    # 绘制虚线表示预测值
    ax.plot(x, y, z, 
            color=color, 
            linestyle='--',  # 虚线
            label=f"Pred{lane_idx}", 
            linewidth=2)

# 添加图例
ax.legend(loc='upper left', fontsize=8)
plt.show()