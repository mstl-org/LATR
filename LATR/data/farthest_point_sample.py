import os
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
import time

def farthest_point_sample(points, n_points):
    N, D = points.shape
    centroids = np.zeros(n_points, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    sampled_points = np.zeros((n_points, D))

    for i in range(n_points):
        centroids[i] = farthest
        sampled_points[i] = points[farthest]
        centroid = points[farthest, :][None, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=0)

    return sampled_points

def process_subfolder(args):
    """
    处理单个子文件夹中的 .bin 文件。
    
    Args:
        args (tuple): (subfolder_path, n_points)
    """
    subfolder_path, n_points = args
    velodyne_folder = os.path.join(subfolder_path, 'velodyne_filter')
    
    if not os.path.exists(velodyne_folder):
        print(f"跳过 {subfolder_path}，未找到 velodyne_filter 文件夹")
        return

    # 创建输出文件夹
    output_folder = os.path.join(subfolder_path, 'farthest_filter')
    os.makedirs(output_folder, exist_ok=True)

    # 遍历 velodyne_filter 中的 .bin 文件
    for bin_file in os.listdir(velodyne_folder):
        if not bin_file.endswith('.bin'):
            continue

        bin_file_path = os.path.join(velodyne_folder, bin_file)
        
        # 读取 .bin 文件
        points = np.fromfile(bin_file_path, dtype=np.float32)
        points = points.reshape(-1, 6)[:, :3]  # 取 x, y, z
        
        # 采样或补齐
        if points.shape[0] > n_points:
            points = farthest_point_sample(points, n_points)
        elif points.shape[0] < n_points:
            indices = np.random.choice(points.shape[0], n_points - points.shape[0])
            padding_points = points[indices]
            points = np.concatenate([points, padding_points], axis=0)
        
        # 验证点数
        if points.shape[0] != n_points:
            print(f"错误: {bin_file_path} 处理后点数 {points.shape[0]} 不等于 {n_points}")
            continue

        # 转换为 torch 张量并保存
        points = np.asarray(points, dtype=np.float32)
        output_file_path = os.path.join(output_folder, bin_file)
        points.tofile(output_file_path)
        print(f"已处理: {bin_file_path} -> {output_file_path}")
        

def process_bin_files(base_folder, n_points=16384, num_processes=None):
    """
    使用多进程遍历 base_folder 下所有子文件夹，处理 velodyne_filter 中的 .bin 文件。
    
    Args:
        base_folder (str): 根目录路径
        n_points (int): 采样点数，默认为 16384
        num_processes (int): 进程数，默认为 CPU 核心数
    """
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 获取所有子文件夹
    subfolders = [
        os.path.join(base_folder, subfolder)
        for subfolder in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, subfolder))
    ]
    
    # 设置进程数
    if num_processes is None:
        num_processes = min(cpu_count(), len(subfolders))
    num_processes = max(1, 32)  # 至少 1 个进程
    
    print(f"使用 {num_processes} 个进程处理 {len(subfolders)} 个子文件夹")
    
    # 创建进程池
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        pool.map(process_subfolder, [(subfolder, n_points) for subfolder in subfolders])
    
    print(f"总处理时间: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    base_folder = '/media/home/data_share/OpenLane/waymo/data/results/data_final/validation/'
    n_points = 16384
    num_processes = cpu_count()  # 使用所有 CPU 核心
    process_bin_files(base_folder, n_points, num_processes)