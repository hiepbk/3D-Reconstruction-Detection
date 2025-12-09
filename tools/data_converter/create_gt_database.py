# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pickle
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from os import path as osp
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from typing import List

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps


#debug
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        if ann['area'] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get('iscrowd', False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann['segmentation'])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(
        bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair
    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = (
        torch.arange(num_pos,
                     device=device).to(dtype=pos_proposals.dtype)[:, None])
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks).to(device).index_select(
            0, pos_assigned_gt_inds).to(dtype=rois.dtype))
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = (
        roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1))
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1:y1 + h, x1:x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1:y1 + h, x1:x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


def create_groundtruth_database(dataset_class_name,
                                data_path,
                                info_prefix,
                                info_path=None,
                                mask_anno_path=None,
                                used_classes=None,
                                database_save_path=None,
                                db_info_save_path=None,
                                relative_path=True,
                                add_rgb=False,
                                lidar_only=False,
                                bev_only=False,
                                coors_range=None,
                                with_mask=False,
                                debug=False):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        mask_anno_path (str): Path of the mask_anno.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.
    """
    print(f'Create GT Database of {dataset_class_name}')
    dataset_cfg = dict(
        type=dataset_class_name, data_root=data_path, ann_file=info_path)
    if dataset_class_name == 'KittiDataset':
        file_client_args = dict(backend='disk')
        dataset_cfg.update(
            test_mode=False,
            split='training',
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=with_mask,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    file_client_args=file_client_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=file_client_args)
            ])

    elif dataset_class_name == 'NuScenesDataset':
        
        dataset_cfg.update(
            use_valid_flag=True,
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5, #modify for kakao, katech or custom here, since depends on the dataset, will be different
                    use_dim=[0, 1, 2, 3, 4], #modify for kakao, katech or custom here, since depends on the dataset, will be different
                    ),
                # dict(
                #     type='LoadPointsFromMultiSweeps',
                #     sweeps_num=10,
                #     load_dim=5, #modify for kakao, katech or custom here, since depends on the dataset, will be different
                #     use_dim=[0, 1, 2, 3, 4], #modify for kakao, katech or custom here, since depends on the dataset, will be different
                #     pad_empty_sweeps=True,
                #     remove_close=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True)
            ])

    elif dataset_class_name == 'WaymoDataset':
        file_client_args = dict(backend='disk')
        dataset_cfg.update(
            test_mode=False,
            split='training',
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=6,
                    use_dim=5,
                    file_client_args=file_client_args),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=file_client_args)
            ])

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f'{info_prefix}_gt_database')
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path,
                                     f'{info_prefix}_dbinfos_train.pkl')
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    if with_mask:
        coco = COCO(osp.join(data_path, mask_anno_path))
        imgIds = coco.getImgIds()
        file2id = dict()
        for i in imgIds:
            info = coco.loadImgs([i])[0]
            file2id.update({info['file_name']: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example['ann_info']
        image_idx = example['sample_idx']
        points = example['points'].tensor.numpy()
        pts_file_name = example['pts_filename']
        
        
        # print(f"points shape: {points.shape}")
        
        # following the dataset class of NuScenes, this bbox will be converted to the bottom center point (0.5, 0.5, 0) (kitti format)
        # Then use the box_np_ops.points_in_rbbox(points, gt_boxes_3d) also use 0.5, 0.5, 0 (kitti format) for consistent pooling
        gt_boxes_3d = annos['gt_bboxes_3d'].tensor.numpy()
        
        
        # debug
        # However, to visualize, we need use gravity center (0,0,0)
        # and then convert SECOND format back to nuscene format
        if debug:
            gt_boxes_3d_ori = gt_boxes_3d.copy()
            src = np.array((0.5, 0.5, 0))
            dst = np.array((0.5, 0.5, 0.5))
            gt_boxes_3d_ori[:, :3] += gt_boxes_3d_ori[:, 3:6] * (dst - src)
            
            # because Forcalformer3D create_data.py convert the gt to the Second format,
            # so we need convert back to nuscene format
            
            gt_boxes_3d_ori[:, 6] = -gt_boxes_3d_ori[:, 6] - np.pi / 2
            
            ## for visualization, we need swap the width and length
            gt_boxes_3d_ori[..., [3, 4]] = gt_boxes_3d_ori[..., [4, 3]]

            vis_point = []
            vis_point.append(points)
        
        
        
        names = annos['gt_names']
        group_dict = dict()
        if 'group_ids' in annos:
            group_ids = annos['group_ids']
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if 'difficulty' in annos:
            difficulty = annos['difficulty']

        num_obj = gt_boxes_3d.shape[0]
        
        # this function suitable with the box of Second format
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        if with_mask:
            # prepare masks
            gt_boxes = annos['gt_bboxes']
            img_path = osp.split(example['img_info']['filename'])[-1]
            if img_path not in file2id.keys():
                print(f'skip image {img_path} for empty mask')
                continue
            img_id = file2id[img_path]
            kins_annIds = coco.getAnnIds(imgIds=img_id)
            kins_raw_info = coco.loadAnns(kins_annIds)
            kins_ann_info = _parse_coco_ann_info(kins_raw_info)
            h, w = annos['img_shape'][:2]
            gt_masks = [
                _poly2mask(mask, h, w) for mask in kins_ann_info['masks']
            ]
            # get mask inds based on iou mapping
            bbox_iou = bbox_overlaps(kins_ann_info['bboxes'], gt_boxes)
            mask_inds = bbox_iou.argmax(axis=0)
            valid_inds = (bbox_iou.max(axis=0) > 0.5)

            # mask the image
            # use more precise crop when it is ready
            # object_img_patches = np.ascontiguousarray(
            #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
            # crop image patches using roi_align
            # object_img_patches = crop_image_patch_v2(
            #     torch.Tensor(gt_boxes),
            #     torch.Tensor(mask_inds).long(), object_img_patches)
            object_img_patches, object_masks = crop_image_patch(
                gt_boxes, gt_masks, mask_inds, annos['img'])
            
            
        #debug
        # if debug:
        #     closest_obj_index = 0
        #     closest_obj_distance = 1000000
        for i in range(num_obj):
            filename = f'{image_idx}_{names[i]}_{i}.bin'
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f'{info_prefix}_gt_database', filename)
            
            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            
            # original gt_points is in lidar coordinate, minus the center of box to transform to box coordinate
            gt_points[:, :3] -= gt_boxes_3d[i, :3]
            
            ##debug
            if debug:
                relative_gt_boxs = gt_boxes_3d[i, None, :].copy()
                relative_gt_boxs[:, :3] -= gt_boxes_3d[i, :3]
                
                center = gt_boxes_3d.copy()[i,:3]
                
                #calculate distance between center and (0,0,0)
                distance = np.linalg.norm(center[:3])
                
                # if distance < closest_obj_distance:
                #     closest_obj_distance = distance
                #     closest_obj_index = i
                #     closest_file_name = filename
                vis_point.append(points[point_indices[:, i]].copy())

            if with_mask:
                if object_masks[i].sum() == 0 or not valid_inds[i]:
                    # Skip object for empty or invalid mask
                    continue
                img_patch_path = abs_filepath + '.png'
                mask_patch_path = abs_filepath + '.mask.png'
                mmcv.imwrite(object_img_patches[i], img_patch_path)
                mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    'name': names[i],
                    'path': rel_filepath,
                    'image_idx': image_idx,
                    'gt_idx': i,
                    'box3d_lidar': gt_boxes_3d[i],
                    'num_points_in_gt': gt_points.shape[0],
                    'difficulty': difficulty[i],
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if 'score' in annos:
                    db_info['score'] = annos['score'][i]
                if with_mask:
                    db_info.update({'box2d_camera': gt_boxes[i]})
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
        if debug:
            # print(f'closest_obj_index: {closest_obj_index}, with file: {closest_file_name}')
            #debug
            vis_point_with_gt_box(vis_point, gt_boxes_3d_ori, window_name=f'{pts_file_name} - Vis raw_points + gt_points, no_points: {points.shape[0]}', voxel_size=None, axis_size=0.5)

    for k, v in all_db_infos.items():
        print(f'load {len(v)} {k} database infos')

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)


def create_point_geometry(points, voxel_size=0.1, colors=None):
    # Filter points too far from origin
    mask = (points[:, 0] < 50) & (points[:, 1] < 50) & (points[:, 2] < 50) & \
           (points[:, 0] > -50) & (points[:, 1] > -50) & (points[:, 2] > -50)
    points = points[mask]

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Assuming the first 3 columns are XYZ

    if voxel_size is not None:
        # Voxelization
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    else:
        downsampled_pcd = pcd

    # Set colors
    if colors is None:
        # Create a gray color array with the same number of points
        colors = np.ones((len(downsampled_pcd.points), 3)) * 0.3
    downsampled_pcd.colors = o3d.utility.Vector3dVector(colors)

    return downsampled_pcd

def vis_point_with_gt_box(points: List[np.ndarray], gt_boxes=None, window_name='', voxel_size=0.1, axis_size=5.0):
    
    geometries = []
    N = len(points)
    
    # Create a list of color arrays, one for each point cloud
    colors = [None] + [np.tile(np.random.rand(1, 3), (point.shape[0], 1)) for point in points[1:]]
    
    if N > 1:
        voxel_sizes = [0.1] + [None] * (N-1)
    else:
        voxel_sizes = [0.1]
    
    
    for i, point in enumerate(points):
        pcd = create_point_geometry(point, voxel_sizes[i], colors[i])
        geometries.append(pcd)


    if gt_boxes is not None:
        for i, gt_box in enumerate(gt_boxes):
            center = gt_box[:3]  # x, y, z
            size = gt_box[3:6]   # width, length, height
            yaw = gt_box[6]      # rotation around z-axis

            # Create an oriented bounding box
            obb = o3d.geometry.OrientedBoundingBox()
            obb.center = center
            obb.extent = size
            obb.R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
            obb.color = (1, 0, 0)  # red color for box lines


            geometries.append(obb)
            
            
             # find the center of front face (heading direction)
            # then connect the bbox center with the front center -> heading direction
            front_center = center + size[0] * np.array([np.cos(yaw), np.sin(yaw), 0])
            # append geometry line set from center to front center
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([center, front_center])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([(1, 0, 0)])
            geometries.append(line_set)
            
 
    # Axes
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    geometries.append(axis)

    
    # Visualize the point cloud and bounding boxes
    o3d.visualization.draw_geometries(geometries, window_name=window_name)
    
    