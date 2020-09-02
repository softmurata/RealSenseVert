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
    test_dataset_dir = 'MyDataset/'
    if not os.path.exists(test_dataset_dir):
        os.mkdir(test_dataset_dir)
    rgb_image_dir = test_dataset_dir + 'rgb/'
    depth_image_dir = test_dataset_dir + 'depth/'
    image_left_dir = test_dataset_dir + 'image_left/'
    image_right_dir = test_dataset_dir + 'image_right/'
    
    if not os.path.exists(rgb_image_dir):
        os.mkdir(rgb_image_dir)
    
    if not os.path.exists(depth_image_dir):
        os.mkdir(depth_image_dir)  
    
    if not os.path.exists(image_left_dir):
        os.mkdir(image_left_dir)
    
    if not os.path.exists(image_right_dir):
        os.mkdir(image_right_dir)
    
    FPS = 30
    
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, FPS)
    
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
            infr_frame1 = aligned_frames.get_infrared_frame(1)
            infr_frame2 = aligned_frames.get_infrared_frame(2)
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            
            # save rgb image and depth
            save_rgb_image = color_image.astype(np.uint8)
            
            """   
            grey_color = 153
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            """
            print(np.max(depth_image))
            print('')
            
            # save image
            cv2.imwrite(rgb_image_dir + '{}.png'.format(count), save_rgb_image)
            np.save(depth_image_dir + '{}.npy'.format(count), depth_image)
            # np.save(depth_image_dir + '{}.npy'.format(count), save_depth_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            infr1 = np.asanyarray(infr_frame1.get_data())
            infr2 = np.asanyarray(infr_frame2.get_data())
            cv2.imwrite(image_right_dir + '{}.png'.format(count), infr1)
            cv2.imwrite(image_left_dir + '{}.png'.format(count), infr2)
            
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            infr_images = np.hstack((infr1, infr2))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.imshow('Infrared', infr_images)
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
