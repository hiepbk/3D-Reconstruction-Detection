# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import os

import datetime
from mmcv.parallel import DataContainer as DC
from mmdet3d.core import Box3DMode, Coord3DMode, show_result, LiDARInstance3DBoxes

from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)


from projects.mmdet3d_plugin.datasets.utils import (
    draw_lidar_bbox3d_on_img,
    draw_lidar_bbox3d_on_bev,
    draw_lidar_bbox3d_on_pc,
)


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3, UI_result=False, score_threshold=0.1):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    import json
    import cv2
    import numpy as np

    # Initialize output JSON for the entire dataset

    
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    videoWriter = None
    
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # sample_idx = 1447
            # data = data_loader.dataset[sample_idx]
            result = model(return_loss=False, rescale=True, **data)
        
        if UI_result:
            single_frame_output_json = {
                'metadata': {
                    'classes': dataset.CLASSES,
                    'class_colors': {cls: dataset.ID_COLOR_MAP[cls_idx] for cls_idx, cls in enumerate(dataset.CLASSES)},
                    'cat2id': dataset.cat2id,
                    'coordinate_system': {"x": "right", "y": "forward", "z": "up"},
                    'frame_index': f'{i:06d}'
                },
                'results': []
            }
            
            # make dir if not exist
            os.makedirs(out_dir, exist_ok=True)
            
            for batch_id in range(len(result)):
                result_batch = result[batch_id]['pts_bbox']
                
                
                imgs = []
                
                if isinstance(data['points'][0], DC):
                    points = data['points'][0]._data[0][batch_id].numpy()
                elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                    points = data['points'][0][batch_id]
                else:
                    raise ValueError(f"Unsupported data type {type(data['points'][0])} "
                            f'for visualization!')
                    
                    
                if isinstance(data['img'][0], DC):
                    raw_imgs = data['img'][0]._data[0][batch_id]
                elif mmcv.is_list_of(data['img'][0], torch.Tensor):
                    raw_imgs = data['img'][0][batch_id]
                else:
                    raise ValueError(f"Unsupported data type {type(data['img'][0])} "
                            f'for visualization!')

                
                if isinstance(data['img_metas'][0], DC):
                    pts_filename = data['img_metas'][0]._data[0][batch_id][
                        'pts_filename']
                    box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                        'box_mode_3d']
                elif mmcv.is_list_of(data['img_metas'][0], dict):
                    pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                    box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
                else:
                    ValueError(
                        f"Unsupported data type {type(data['img_metas'][0])} "
                        f'for visualization!')
                file_name = osp.split(pts_filename)[-1].split('.')[0]
                
                # filter out low score boxes
                mask_inds = result_batch['scores_3d'] > score_threshold

                pred_bboxes = result_batch['boxes_3d'][mask_inds]
                pred_scores = result_batch['scores_3d'][mask_inds]
                pred_labels = result_batch['labels_3d'][mask_inds]
                
                # Convert points and bbox into LIDAR mode
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR, Coord3DMode.LIDAR)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d, Box3DMode.LIDAR)
                
                # Convert bboxes to numpy for processing
                pred_bboxes = pred_bboxes.tensor.cpu().numpy()
                
                # Prepare gravity center bboxes
                gravity_center_bboxes = pred_bboxes.copy()
                gravity_center_bboxes[:, 2] += gravity_center_bboxes[:, 5] / 2
                
                
                
                ## for visualization, we need to convert the bbox to the nuscene format
                
                ## because Forcalformer3D predict the yaw angle as Second format,
                #  then we need to convert yaw back to nuscene format
                # for visualization, we need to swap the width and length
                gravity_center_bboxes[..., [3, 4]] = gravity_center_bboxes[..., [4, 3]]
                gravity_center_bboxes[..., 6] = -gravity_center_bboxes[..., 6] - np.pi / 2
                
                # Prepare frame prediction
                frame_prediction = {
                    'frame_id': f'{i:06d}',
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")[:-3],
                    'pts_filename': pts_filename,
                    'predictions': []
                }
                
                # Prepare detections
                for box_id in range(len(pred_bboxes)):
                    label = pred_labels[box_id].item()
                    category = dataset.CLASSES[label]
                    
                    prediction = {
                        'instance_id': box_id,
                        'label_id': label,
                        'category': category,
                        'score': float(pred_scores[box_id].item()),
                        'bbox3d': {
                            'x': float(gravity_center_bboxes[box_id][0]),
                            'y': float(gravity_center_bboxes[box_id][1]),
                            'z': float(gravity_center_bboxes[box_id][2]),
                            'dx': float(gravity_center_bboxes[box_id][3]),
                            'dy': float(gravity_center_bboxes[box_id][4]),
                            'dz': float(gravity_center_bboxes[box_id][5]),
                            'yaw': float(gravity_center_bboxes[box_id][6])
                        },
                        'velocity': {
                            'vx': float(gravity_center_bboxes[box_id][7]),
                            'vy': float(gravity_center_bboxes[box_id][8]),
                            'vz': 0.0  # Adding vz as 0.0 since original data doesn't have it
                        },
                        'attribute': [],
                        "track_id": []
                    }
                    frame_prediction['predictions'].append(prediction)
                
                # Add frame prediction to output JSON
                single_frame_output_json['results'].append(frame_prediction)
                
                lidar2img = data["img_metas"][0]._data[0][batch_id]["lidar2img"]
                pred_cat_names = [dataset.CLASSES[label] for label in pred_labels]
                pred_color = []
                for id in pred_labels:
                    pred_color.append(dataset.ID_COLOR_MAP[id])
                pred_dict = {
                    "bboxes_3d": gravity_center_bboxes,
                    "labels_3d": pred_labels,
                    "cat_names": pred_cat_names,
                    "scores_3d": pred_scores,
                    "colors": pred_color,
                }
                
                gt_dict = None
                # ===== draw boxes_3d to images =====
                for j, img_origin in enumerate(raw_imgs):
                    img = img_origin.permute(1, 2, 0).numpy().astype(np.uint8).copy()
                    img = draw_lidar_bbox3d_on_img(
                        img,
                        pred_dict,
                        gt_dict,
                        lidar2img[j],
                        img_metas=None,
                        thickness=3,
                    )
                    imgs.append(img)

                # ===== draw boxes_3d to BEV =====
                bev = draw_lidar_bbox3d_on_bev(
                    pred_dict,
                    gt_dict,
                    bev_size=img.shape[0] * 2,
                )
                
                # put the text of frame on the center up of bev image
                cv2.rectangle(bev, (bev.shape[1]//4, 0), (bev.shape[1] *3//4, 100), (255, 255, 255), -1)
                cv2.putText(bev, f"Frame {i:06d}", (bev.shape[1] //2 - cv2.getTextSize(f"Frame {i:06d}", cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0][0]//2 , 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
                

                # ===== put text and concat =====
                for j, name in enumerate(
                    [
                        "front",
                        "front right",
                        "front left",
                        "rear",
                        "rear left",
                        "rear right",
                    ]
                ):
                    imgs[j] = cv2.rectangle(
                        imgs[j],
                        (0, 0),
                        (440, 80),
                        color=(255, 255, 255),
                        thickness=-1,
                    )
                    w, h = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                    text_x = int(220 - w / 2)
                    text_y = int(40 + h / 2)

                    imgs[j] = cv2.putText(
                        imgs[j],
                        name,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                image = np.concatenate(
                    [
                        np.concatenate([imgs[2], imgs[0], imgs[1]], axis=1),
                        np.concatenate([imgs[5], imgs[3], imgs[4]], axis=1),
                    ],
                    axis=0,
                )
                image = np.concatenate([image, bev], axis=1)
                

                # ===== save video =====
                if videoWriter is None:
                    videoWriter = cv2.VideoWriter(
                        osp.join(out_dir, "video.avi"),
                        fourcc,
                        7,
                        image.shape[:2][::-1],
                    )
                videoWriter.write(image)
                
                                # write output_json to json file
                with open(osp.join(out_dir, f'result_frame{i:06d}.json'), 'w') as f:
                    json.dump(single_frame_output_json, f, indent=2)
                    

                
                # print(f"Saved {i-4} to {i} frames predictions to {osp.join(out_dir, f'result_frame{i-4}_to_{i}.json')}")

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
                    
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
            
    if videoWriter is not None:
        videoWriter.release()
    
    return results
