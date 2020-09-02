import pyrealsense2 as rs
import numpy as np
import cv2
import os
import open3d as o3d
"""
camera parameters
IntelRealsenseCalib/DynamicCalibrationAPI/2.11.0.0/data
"""


def initial_test():
    test_dataset_dir = 'TestDatasetBox/'
    if not os.path.exists(test_dataset_dir):
        os.mkdir(test_dataset_dir)
    rgb_image_dir = test_dataset_dir + 'rgb/'
    depth_image_dir = test_dataset_dir + 'depth/'
    
    if not os.path.exists(rgb_image_dir):
        os.mkdir(rgb_image_dir)
    
    if not os.path.exists(depth_image_dir):
        os.mkdir(depth_image_dir)  
    
    FPS = 30
    
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)

    print(config)
    
    # create timestamp

    # Start streaming
    profile = pipeline.start(config)
    
    # adjust frame angle
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # intrinsic matrix
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    
    print('intrinsic:', intr)
    
    # print("cx is: ",intr.ppx)
    # print("cy is: ",intr.ppy)
    # print("fx is: ",intr.fx)
    # print("fy is: ",intr.fy)
    # print('depth scale:', 1 / depth_scale)
    
    start_count = 30
    finish_count = 400
    
    try:
        count = 0
        all_pcd = o3d.geometry.PointCloud()
        all_points = []
        while True:
            
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            """
            # point cloud
            pc = rs.pointcloud()
            points = rs.points
            
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            vertices = np.asanyarray(points.get_vertices())
            point_vtx = np.zeros((len(vertices), 3))
            
            for i in range(len(vertices)):
                point_vtx[i][0] = np.float(vertices[i][0])
                point_vtx[i][1] = np.float(vertices[i][1])
                point_vtx[i][2] = np.float(vertices[i][2])
            
            print('')
            # print(np.max(depth_image))
            # print(np.min(depth_image))
            # all_points.extend(point_vtx)
            # print(all_points)
            all_pcd.points = o3d.utility.Vector3dVector(point_vtx)
            o3d.io.write_point_cloud('output.ply', all_pcd)
            """
            
            
            # save rgb image and depth
            save_rgb_image = color_image.astype(np.uint8)
            if test_dataset_dir == 'TestDatasetBit8/':
                save_depth_image = depth_image.astype(np.uint8)
            else:
                save_depth_image = depth_image.astype(np.uint16)
            
            """   
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            """
            print(np.max(save_depth_image))
            print('')
            
            
            cv2.imwrite(rgb_image_dir + '{}.png'.format(count), save_rgb_image)
            cv2.imwrite(depth_image_dir + '{}.png'.format(count), save_depth_image)
            # np.save(depth_image_dir + '{}.npy'.format(count), save_depth_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)
            
            count += 1
            if count == finish_count:
                break
            # print(count)

    finally:

        # Stop streaming
        pipeline.stop()
        
        
if __name__ == '__main__':
    
    initial_test()        
