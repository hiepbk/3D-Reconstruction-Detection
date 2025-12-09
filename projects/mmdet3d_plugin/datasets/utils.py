import copy

import cv2
import numpy as np
import torch
import open3d as o3d


from projects.mmdet3d_plugin.core.box3d import *


def box3d_to_corners(box3d):
    if isinstance(box3d, torch.Tensor):
        box3d = box3d.detach().cpu().numpy()
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box3d[:, None, [W, L, H]] * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    rot_cos = np.cos(box3d[:, YAW])
    rot_sin = np.sin(box3d[:, YAW])
    rot_mat = np.tile(np.eye(3)[None], (box3d.shape[0], 1, 1))
    rot_mat[:, 0, 0] = rot_cos
    rot_mat[:, 0, 1] = -rot_sin
    rot_mat[:, 1, 0] = rot_sin
    rot_mat[:, 1, 1] = rot_cos
    corners = (rot_mat[:, None] @ corners[..., None]).squeeze(axis=-1)
    corners += box3d[:, None, :3]
    return corners



def draw_lidar_bbox3d_on_pc(
    points,
    pred_dict=None,
    gt_dict=None,

):
    """Draw the 3D bbox on the point cloud.
    """
    
    if gt_dict is not None:
        gt_boxes = gt_dict.get("bboxes_3d", None)
        gt_colors = gt_dict.get("colors", None)
        #normalize gt_colors to 0-1
        gt_colors = [(b / 255.0, g / 255.0, r / 255.0) for (r, g, b) in gt_colors]
        gt_labels = gt_dict.get("labels_3d", None)
    else:
        gt_boxes = None
        gt_colors = None
        gt_labels = None
    if pred_dict is not None:
        pred_boxes = pred_dict.get("bboxes_3d", None)
        pred_colors = pred_dict.get("colors", None)
        #normalize pred_colors to 0-1
        pred_colors = [(b / 255.0, g / 255.0, r / 255.0) for (r, g, b) in pred_colors]
        pred_labels = pred_dict.get("labels_3d", None)
    else:
        pred_boxes = None
        pred_colors = None
        pred_labels = None
    
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Assuming the first 3 columns are XYZ

    # Filter points too far from origin
    mask = (points[:, 0] < 50) & (points[:, 1] < 50) & (points[:, 2] < 50) & \
           (points[:, 0] > -50) & (points[:, 1] > -50) & (points[:, 2] > -50)
    points = points[mask]

    # Voxelization
    voxel_size = 0.1
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Color gray
    colors = np.ones_like(points[:, :3]) * 0.3
    downsampled_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    
    geometries = [downsampled_pcd]
    
    # Draw GT boxes
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
            obb.color = gt_colors[i] if gt_colors is not None else (0, 0, 0)
            geometries.append(obb)
    
    # Draw pred boxes
    if pred_boxes is not None:
        for i, pred_box in enumerate(pred_boxes):
            center = pred_box[:3]  # x, y, z
            size = pred_box[3:6]   # width, length, height
            yaw = pred_box[6]      # rotation around z-axis 

            # Create an oriented bounding box
            obb = o3d.geometry.OrientedBoundingBox()
            obb.center = center
            obb.extent = size
            obb.R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
            obb.color = pred_colors[i] if pred_colors is not None else (0, 0, 0)
            geometries.append(obb)
    # Axes
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
    geometries.append(axis)

    return geometries


def draw_lidar_bbox3d_on_img(
    raw_img, 
    pred_dict=None, 
    gt_dict=None,
    lidar2img_rt=None, 
    img_metas=None, 
    thickness=1
):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        raw_img (numpy.array): The numpy array of image.
        pred_dict (dict): The dictionary of prediction.
        gt_dict (dict): The dictionary of ground truth.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
    """
    plot_img = raw_img.copy()
    # corners_3d = bboxes3d.corners
    
    # convert pred_bboxes3d to corners_3d
    if pred_dict is not None:
        pred_corners_3d = box3d_to_corners(pred_dict["bboxes_3d"])
        num_pred_bbox = pred_corners_3d.shape[0]
        
        assert len(pred_dict["labels_3d"]) == len(pred_dict["cat_names"]) == len(pred_dict["scores_3d"]) == num_pred_bbox, \
        "The length of labels_3d, cat_names, and scores_3d must be equal to the number of bboxes."
        
        pred_pts_4d = np.concatenate([pred_corners_3d.reshape(-1, 3), np.ones((num_pred_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pred_pts_2d = pred_pts_4d @ lidar2img_rt.T

        pred_pts_2d[:, 2] = np.clip(pred_pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pred_pts_2d[:, 0] /= pred_pts_2d[:, 2]
        pred_pts_2d[:, 1] /= pred_pts_2d[:, 2]
        pred_imgfov_pts_2d = pred_pts_2d[..., :2].reshape(num_pred_bbox, 8, 2)
        
        pred_dict['img_fov_pts_2d'] = pred_imgfov_pts_2d
        pred_dict['num_rects'] = num_pred_bbox
        
        plot_img = plot_rect3d_on_img(plot_img, pred_dict, thickness)
        
    # convert gt_bboxes_3d to corners_3d
    if gt_dict is not None:
        gt_corners_3d = box3d_to_corners(gt_dict["bboxes_3d"])
        num_gt_bbox = gt_corners_3d.shape[0]
        # convert gt_bboxes_3d to corners_3d
        gt_pts_4d = np.concatenate(
            [gt_corners_3d.reshape(-1, 3), np.ones((num_gt_bbox * 8, 1))], axis=-1
        )
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        gt_pts_2d = gt_pts_4d @ lidar2img_rt.T
        
        gt_pts_2d[:, 2] = np.clip(gt_pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        gt_pts_2d[:, 0] /= gt_pts_2d[:, 2]
        gt_pts_2d[:, 1] /= gt_pts_2d[:, 2]
        gt_imgfov_pts_2d = gt_pts_2d[..., :2].reshape(num_gt_bbox, 8, 2)
        
        gt_dict['img_fov_pts_2d'] = gt_imgfov_pts_2d
        gt_dict['num_rects'] = num_gt_bbox
        plot_img = plot_rect3d_on_img(plot_img, gt_dict, thickness)
    
    return plot_img


def plot_rect3d_on_img(
    img, visualize_dict, thickness=1
):
    """Plot the boundary lines of 3D rectangular on 2D images.

    Args:
        img (numpy.array): The numpy array of image.
        visualize_dict (dict): The dictionary of visualization.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    
    if visualize_dict is None:
        return img
    num_rects = visualize_dict['num_rects']
    rect_corners = visualize_dict['img_fov_pts_2d']
    labels_3d = visualize_dict['labels_3d']
    cat_names = visualize_dict['cat_names']
    scores_3d = visualize_dict['scores_3d']
    color = visualize_dict['colors']
    
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    h, w = img.shape[:2]
    for i in range(num_rects):
        corners = np.clip(rect_corners[i], -1e4, 1e5).astype(np.int32)
        center = np.mean(corners, axis=0).tolist()
        
        # draw text for the labels_3d, labels_3d_to_cat, scores_3d
        # text = f"{cat_names[i]}({labels_3d[i]}):{scores_3d[i]:.2f}"
        # if isinstance(color[0], int):
        #     cv2.putText(img, text, (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)
        # else:
        #     cv2.putText(img, text, (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 3, color[i], 2)
        
        for start, end in line_indices:
            if (
                (corners[start, 1] >= h or corners[start, 1] < 0)
                or (corners[start, 0] >= w or corners[start, 0] < 0)
            ) and (
                (corners[end, 1] >= h or corners[end, 1] < 0)
                or (corners[end, 0] >= w or corners[end, 0] < 0)
            ):
                continue
            if isinstance(color[0], int):
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.line(
                    img,
                    (corners[start, 0], corners[start, 1]),
                    (corners[end, 0], corners[end, 1]),
                    color[i],
                    thickness,
                    cv2.LINE_AA,
                )

    return img.astype(np.uint8)


def draw_points_on_img(points, img, lidar2img_rt, color=(0, 255, 0), circle=4):
    img = img.copy()
    N = points.shape[0]
    points = points.cpu().numpy()
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = (
        np.sum(points[:, :, None] * lidar2img_rt[:3, :3], axis=-1)
        + lidar2img_rt[:3, 3]
    )
    pts_2d[..., 2] = np.clip(pts_2d[..., 2], a_min=1e-5, a_max=1e5)
    pts_2d = pts_2d[..., :2] / pts_2d[..., 2:3]
    pts_2d = np.clip(pts_2d, -1e4, 1e4).astype(np.int32)

    for i in range(N):
        for point in pts_2d[i]:
            if isinstance(color[0], int):
                color_tmp = color
            else:
                color_tmp = color[i]
            cv2.circle(img, point.tolist(), circle, color_tmp, thickness=-1)
    return img.astype(np.uint8)


def draw_lidar_bbox3d_on_bev(
    pred_dict, 
    gt_dict,
    bev_size, 
    bev_range=115, 
    thickness=3):
    
    '''
    pred_dict:
        bboxes_3d:
        labels_3d:
        cat_names:
        scores_3d:
        colors:
    gt_dict:
    
    '''
    
    
    if isinstance(bev_size, (list, tuple)):
        bev_h, bev_w = bev_size
    else:
        bev_h, bev_w = bev_size, bev_size
    bev = np.zeros([bev_h, bev_w, 3])

    marking_color = (127, 127, 127)
    bev_resolution = bev_range / bev_h
    for cir in range(int(bev_range / 2 / 10)):
        cv2.circle(
            bev,
            (int(bev_h / 2), int(bev_w / 2)),
            int((cir + 1) * 10 / bev_resolution),
            marking_color,
            thickness=thickness,
        )
    cv2.line(
        bev,
        (0, int(bev_h / 2)),
        (bev_w, int(bev_h / 2)),
        marking_color,
    )
    cv2.line(
        bev,
        (int(bev_w / 2), 0),
        (int(bev_w / 2), bev_h),
        marking_color,
    )
    
    
    
    
    if pred_dict is not None:
        pred_bboxes_3d = pred_dict["bboxes_3d"]
        if len(pred_bboxes_3d) != 0:
            bev_corners = box3d_to_corners(pred_bboxes_3d)[:, [0, 3, 4, 7]][
                ..., [0, 1]
            ]
            xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
            ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
            for obj_idx, (x, y) in enumerate(zip(xs, ys)):
                # center = np.mean(bev_corners[obj_idx], axis=0).tolist()
                # text = f"{cat_names[obj_idx]}({labels_3d[obj_idx]}):{scores_3d[obj_idx]:.2f}"
                # if isinstance(color[0], int):
                #     cv2.putText(bev, text, (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # else:
                #     cv2.putText(bev, text, (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[obj_idx], 2)
                
                for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                    if isinstance(pred_dict["colors"][0], (list, tuple)):
                        tmp = pred_dict["colors"][obj_idx]
                    else:
                        tmp = pred_dict["colors"]
                    cv2.line(
                        bev,
                        (int(x[p1]), int(y[p1])),
                        (int(x[p2]), int(y[p2])),
                        tmp,
                        thickness=thickness,
                )
                    
    if gt_dict is not None:
        gt_bboxes_3d = gt_dict["bboxes_3d"]
        if len(gt_bboxes_3d) != 0:
            bev_corners = box3d_to_corners(gt_bboxes_3d)[:, [0, 3, 4, 7]][
                ..., [0, 1]
            ]
            xs = bev_corners[..., 0] / bev_resolution + bev_w / 2
            ys = -bev_corners[..., 1] / bev_resolution + bev_h / 2
            for obj_idx, (x, y) in enumerate(zip(xs, ys)):
                for p1, p2 in ((0, 1), (0, 2), (1, 3), (2, 3)):
                    if isinstance(gt_dict["colors"][0], (list, tuple)):
                        tmp = gt_dict["colors"][obj_idx]
                    else:
                        tmp = gt_dict["colors"]
                    cv2.line(
                        bev,
                        (int(x[p1]), int(y[p1])),
                        (int(x[p2]), int(y[p2])),
                        tmp,
                        thickness=thickness,
                    )
    return bev.astype(np.uint8)


def draw_lidar_bbox3d(bboxes_3d, imgs, lidar2imgs, color=(255, 0, 0)):
    vis_imgs = []
    for i, (img, lidar2img) in enumerate(zip(imgs, lidar2imgs)):
        vis_imgs.append(
            draw_lidar_bbox3d_on_img(bboxes_3d, img, lidar2img, color=color)
        )

    num_imgs = len(vis_imgs)
    if num_imgs < 4 or num_imgs % 2 != 0:
        vis_imgs = np.concatenate(vis_imgs, axis=1)
    else:
        vis_imgs = np.concatenate([
            np.concatenate(vis_imgs[:num_imgs//2], axis=1),
            np.concatenate(vis_imgs[num_imgs//2:], axis=1)
        ], axis=0)

    bev = draw_lidar_bbox3d_on_bev(bboxes_3d, vis_imgs.shape[0], color=color)
    vis_imgs = np.concatenate([bev, vis_imgs], axis=1)
    return vis_imgs
