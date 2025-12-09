#using open3d to visualize the ground truth of bin file
import open3d as o3d
import numpy as np
import os
from mmcv import Config, DictAction
from mmdet3d.core.visualizer.open3d_vis import Visualizer
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
import os.path as osp

from mmdet3d.datasets import build_dataloader, build_dataset

def read_bin_file(bin_file_path, feature_dim=5):
    points = np.fromfile(bin_file_path, dtype=np.float32)
    points = points.reshape(-1, feature_dim)
    return points

def visualize_points(points, file_name, axis_size=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # the name of window is the name of the file

	# Axes
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    geometries = [pcd, axis]
    o3d.visualization.draw_geometries(geometries, window_name=file_name)
    

def vis_gt_data(gt_folder):

    for file in os.listdir(gt_folder):
        
        if file.endswith(".bin"):
            if 'trailer' not in file:
                continue
            print('vis gt data', gt_folder, file)
            bin_file_path = os.path.join(gt_folder, file)
            points = read_bin_file(bin_file_path, feature_dim=5)
            if points is not None:
                if points.shape[0] > 100:
                    vis = visualize_points(points, bin_file_path)


def vis_point_with_bboxes(points, bboxes, window_name, gt_names=None, colors=None, axis_size=1, src_origin=(0.5, 0.5, 0.5)):
    dst_origin=(0.5, 0.5, 0.5)
    geometries = []
    # Initialize all points as gray (0.5, 0.5, 0.5)
    points_colors = np.full((points.shape[0], 3), 0.5)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    if bboxes is not None:
        if src_origin != dst_origin:
            bboxes[:, :3] = bboxes[:, :3] + (np.array(dst_origin) - np.array(src_origin)) * bboxes[:, 3:6]
    
        for i, (gt_box, gt_name) in enumerate(zip(bboxes, gt_names)):
            
            center = gt_box[:3]  # x, y, z
            size = gt_box[3:6]   # width, length, height
            yaw = gt_box[6]      # rotation around z-axis

            # Create an oriented bounding box
            obb = o3d.geometry.OrientedBoundingBox()
            obb.center = center
            obb.extent = size
            obb.R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
            
            # Set bounding box color
            obb.color = colors[i] if colors is not None else (1, 0, 0)
            
            geometries.append(obb)
            
            
            # Change the color of points which are in this box
            indices = obb.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = colors[i] if colors is not None else (1, 0, 0)
    
            # find the center of front face (heading direction)
            # then connect the bbox center with the front center -> heading direction
            front_center = center + size[1] * np.array([-np.sin(yaw), np.cos(yaw), 0])
            # append geometry line set from center to front center
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([center, front_center])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([colors[i] if colors is not None else (1, 0, 0)])
            geometries.append(line_set)

    # Axes
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    geometries.append(axis)
    
    # Set point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(points_colors)
    geometries.append(pcd)

    # Visualize the point cloud and bounding boxes
    print('window_name', window_name)
    o3d.visualization.draw_geometries(geometries, window_name=window_name)


# python -m tools.gt_vis
if __name__ == "__main__":
    # gt_folder = "/hdd/hiep/CODE/FocalFormer3D/data/nuscenes/nuscenes_gt_database"
    gt_folder = "/hdd/hiep/CODE/FocalFormer3D/data/nuscenes_katech/BATCH1/nuscenes_katech_gt_database"
    cfg_path = "/hdd/hiep/CODE/FocalFormer3D/projects/configs/focalformer3d/FocalFormer3D_L.py"
    split = 'train'
    
    # vis_gt_data(gt_folder)
    
    cfg = Config.fromfile(cfg_path)
    
    train_config = cfg.data.train
    train_dataset = build_dataset(train_config)
    val_config = cfg.data.val
    val_dataset = build_dataset(val_config)
    
    pipeline = val_dataset.datasets[0].pipeline
    if split == 'train':
        datasets = train_dataset.dataset
        
        for dataset in datasets.datasets:
            
            len_data = len(dataset)
            print(f"length of dataset: {len_data}")
            
            for i in range(len_data):
                data_info = dataset.data_infos[i]
                pts_path = data_info['lidar_path']
                
                points = dataset._extract_data(i, pipeline, 'points').numpy()
                # points = read_bin_file(pts_path)
                
                
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                            Coord3DMode.LIDAR)
                
                gt_bboxes = dataset.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
                
                gt_labels = dataset.get_ann_info(i)['gt_labels_3d']
                
                gt_names = dataset.get_ann_info(i)['gt_names'].tolist()
                
                colors = [dataset.ID_COLOR_MAP[label] for label in gt_labels]
                
                # check if trailer class is in the gt_bboxes
                dispplay_class = 'trailer'
                
                if dispplay_class not in gt_names:
                    continue
                # filter, only keep the trailer class
                gt_bboxes = gt_bboxes[gt_names == dispplay_class]
                gt_labels = gt_labels[gt_names == dispplay_class]
                gt_names = gt_names[gt_names == dispplay_class]
                colors = colors[gt_names == dispplay_class]
                

                if len(gt_bboxes) > 0:
                    vis_point_with_bboxes(points, gt_bboxes, pts_path, gt_names=gt_names, colors=colors, axis_size=1, src_origin=(0.5, 0.5, 0.0))
                    
                else:
                    print(f"No bounding boxes for {pts_path}")

                
        
        
        
    
    else:
        if hasattr(val_dataset, 'datasets'):
            
            for dataset in val_dataset.datasets:
        
                pipeline = dataset.pipeline
                
                len_data = len(dataset)
                print(f"length of dataset: {len_data}")
                
                for i in range(len_data):
                    data_info = dataset.data_infos[i]
                    pts_path = data_info['lidar_path']
                    
                    # file_name = osp.split(pts_path)[-1].split('.')[0]
                    
                    
                    points = dataset._extract_data(i, pipeline, 'points').numpy()
                    # points = read_bin_file(pts_path)
                    
                    
                    points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                Coord3DMode.LIDAR)
                    
                    gt_bboxes = dataset.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
                    gt_labels = dataset.get_ann_info(i)['gt_labels_3d']
                    colors = [dataset.ID_COLOR_MAP[label] for label in gt_labels]
                    gt_names = dataset.get_ann_info(i)['gt_names'].tolist()
                    
                    # check if trailer class is in the gt_bboxes
                    if 'trailer' not in gt_names:
                        continue
                    

                    if len(gt_bboxes) > 0:
                        vis_point_with_bboxes(points, gt_bboxes, pts_path, gt_names=gt_names, colors=colors, axis_size=1, src_origin=(0.5, 0.5, 0.0))
                        
                        # vis = Visualizer(points, window_name=pts_path)
                        # vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 1, 0))
                        # vis.show(save_path=pts_path)
                    
                    else:
                        print(f"No bounding boxes for {pts_path}")
                
                    print("visualized", pts_path)
        
                