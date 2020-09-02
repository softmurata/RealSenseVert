from __future__ import print_function
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glob
import numpy as np
from skimage import io
import open3d as o3d
from generate_lidar_point_cloud import project_disp_to_depth


def generate_point_cloud_with_colors():
    # parameters
    parser = argparse.ArgumentParser(description='Generate Lidar Data')
    parser.add_argument('--calib_file', type=str, default='./Kitti/training/calib/001000.txt')  # fix, same as kitti calibration parameters
    
    parser.add_argument('--rgb_dir', type=str, default='./MyDataset/rgb/')
    parser.add_argument('--depth_dir', type=str, default='./MyDataset/depth/')
    parser.add_argument('--save_dir', type=str, default='./own_results/')
    parser.add_argument('--max_high', type=int, default=1)
    
    args = parser.parse_args()
    
    depth_files = glob.glob(args.depth_dir + '*.npy')
    rgb_files = glob.glob(args.rgb_dir + '*.png')
    
    pic_numbers = sorted([int(x.split('/')[-1].split('.')[0]) for x in depth_files])
    rgb_files = list(sorted(rgb_files, key=lambda x: int(x.split('/')[-1].split('.')[0])))
    depth_files = list(sorted(depth_files, key=lambda x: int(x.split('/')[-1].split('.')[0])))
    
    # build camera calibration class
    calib = Calibration(args.calib_file)
    
    # list to restore points and colors
    all_points = []
    all_colors = []
    
    for rgb_image_path, depth_image_path, pic_number in zip(rgb_files, depth_files, pic_numbers):
        # rgb image
        rgb = io.imread(rgb_image_path)
        # depth image(.npy)
        depth = np.load(depth_image_path)
        
        lidar_pcd, lidar_pcd_colors = project_disp_to_depth(calib, rgb, depth, args.max_high)
        
        all_colors.extend(lidar_pcd_colors)
        all_points.extend(lidar_pcd)
        
    all_colors = np.asarray(all_colors)
    all_points = np.asarray(all_points)
    
    # create total point cloud class
    tot_pcd = o3d.geometry.PointCloud()
    tot_pcd.points = o3d.utility.Vector3dVector(all_points)
    tot_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # create raw point directory and point cloud directory
    all_pt_data = np.concatenate([all_points, all_colors], axis=1)
    
    raw_point_save = args.save_dir + 'raw_point/'
    if not os.path.exists(raw_point_save):
        os.mkdir(raw_point_save)
        
    point_cloud_save = args.save_dir + 'point_cloud/'
    if not os.path.exists(point_cloud_save):
        os.mkdir(point_cloud_save)
        
    print('---results confirmation----')
    print('pt data shape:', all_pt_data.shape)
    np.save(raw_point_save + 'multi_test.npy', all_pt_data)
    
    o3d.io.write_point_cloud(point_cloud_save + 'multi_test.ply', tot_pcd)
    


# main function
if __name__ == '__main__':
    generate_point_cloud_with_colors()
    
    
    
    
    