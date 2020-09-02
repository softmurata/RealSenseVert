import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

HEIGHT = 640
WIDTH = 480
FPS = 30

# set stream
config = rs.config()
config.enable_stream(rs.stream.color, HEIGHT, WIDTH, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, HEIGHT, WIDTH, rs.format.z16, FPS)

# make pipeline
pipeline = rs.pipeline()
# start pipeline
profile = pipeline.start(config)

# create align object()
align_to = rs.stream.color
align = rs.align(align_to)

try:
    count = 0
    all_points = []
    all_colors = []
    all_pcd = o3d.geometry.PointCloud()
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        # get color and depth images
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # convert open3d format rgb and d image
        color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
        depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        # create rgbd image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
        print(rgbd_image)
        
        # create point cloud from rgbd imgae
        # ToDo: camera matrix from calibration(RealSense Calibration API and get data)
        # if you get calibration value
        # o3d.camera.PinholeCameraIntrinsic(width: int, height: int, fx: float, fy: float, cx: float, cy: float
        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1, project_valid_depth_only=True)
        
        # pre process for patching mesh
        # revolve
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))
        
        print(all_points)
        
        all_pcd.points = o3d.utility.Vector3dVector(all_points)
        all_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        
        
        
        
        """
        # calculate vertical normalized line
        pcd.estimate_normals()
        
        # downsampling
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
        
        # create mesh from point cloud
        distances = voxel_down_pcd.compute_nearest_neighbor_distance()
        base_dist = np.mean(distances)
        
        radii = [base_dist * 1.5, base_dist * 3]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud(voxel_down_pcd, o3d.utility.DoubleVector(radii))
        
        # recompute normal vectors
        mesh.compute_vertex_normals()
        
        
        # save mesh
        o3d.io.write_triangle_mesh("output.obj", mesh)
        """
        
        count += 1
        
        
finally:
    
    pipeline.stop()
        
        
        
        
        
        
        
        
        