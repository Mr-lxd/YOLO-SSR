import os
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import torch
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results


def get_lr_boundary_points_from_contour(contour_points, num_y_samples, height, y_buffer_percentage=0.01):
    if contour_points is None or len(contour_points) < 3:
        return np.array([]), np.array([])
    if contour_points.ndim == 3 and contour_points.shape[1] == 1:
        contour_points = contour_points.reshape(-1, 2)
    elif contour_points.ndim != 2 or contour_points.shape[1] != 2:
        return np.array([]), np.array([])

    min_y_contour_orig = np.min(contour_points[:, 1])
    max_y_contour_orig = np.max(contour_points[:, 1])
    contour_height = max_y_contour_orig - min_y_contour_orig

    buffer_pixels = int(contour_height * y_buffer_percentage)
    min_y_sample = min_y_contour_orig + buffer_pixels
    max_y_sample = max_y_contour_orig - buffer_pixels

    if min_y_sample >= max_y_sample:
        min_y_sample = min_y_contour_orig
        max_y_sample = max_y_contour_orig

    left_boundary_pts = []
    right_boundary_pts = []

    if max_y_sample > min_y_sample:
        sampled_y_coords = np.linspace(min_y_sample, max_y_sample, num_y_samples).astype(int)
        sampled_y_coords = np.unique(sampled_y_coords)
    else:
        if contour_height >= 1:
            sampled_y_coords = np.array([int((min_y_contour_orig + max_y_contour_orig) / 2)])
        else:
            return np.array([]), np.array([])

    for y_s in sampled_y_coords:
        x_at_y_s = []
        for i in range(len(contour_points)):
            p1 = contour_points[i]
            p2 = contour_points[(i + 1) % len(contour_points)]
            y1, y2 = p1[1], p2[1]
            x1, x2 = p1[0], p2[0]
            if (y1 <= y_s < y2) or (y2 <= y_s < y1):
                if abs(y2 - y1) > 1e-6:
                    intersect_x = x1 + (x2 - x1) * (y_s - y1) / (y2 - y1)
                    x_at_y_s.append(intersect_x)
            elif abs(y1 - y_s) < 1e-6 and abs(y2 - y_s) < 1e-6:
                x_at_y_s.extend(sorted([x1, x2]))
            elif abs(y1 - y_s) < 1e-6:
                x_at_y_s.append(x1)

        if len(x_at_y_s) >= 2:
            left_x = min(x_at_y_s)
            right_x = max(x_at_y_s)
            left_boundary_pts.append([left_x, y_s])
            right_boundary_pts.append([right_x, y_s])

    return np.array(left_boundary_pts), np.array(right_boundary_pts)

def is_detection_in_mask(detection, mask_region, overlap_threshold=0.5, min_conf=0.25):
    if mask_region is None or not np.any(mask_region):
        return True

    conf = detection.get('score', 1.0)
    if conf < min_conf:
        return False

    x, y, w, h = detection['bbox']
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    x1 = max(0, min(x1, mask_region.shape[1] - 1))
    y1 = max(0, min(y1, mask_region.shape[0] - 1))
    x2 = max(0, min(x2, mask_region.shape[1] - 1))
    y2 = max(0, min(y2, mask_region.shape[0] - 1))

    if x2 <= x1 or y2 <= y1:
        return False

    bbox_mask_region = mask_region[y1:y2, x1:x2]
    bbox_area = (x2 - x1) * (y2 - y1)

    if bbox_area == 0:
        return False

    overlap_pixels = np.sum(bbox_mask_region > 0)
    overlap_ratio = overlap_pixels / bbox_area

    return overlap_ratio > overlap_threshold


def extract_detection_centerline(detections, mask_region=None, min_detections=3, overlap_threshold=0.5, min_conf=0.25):
    if not detections:
        return [], 0.0

    center_points = []
    valid_confidences = []

    for det in detections:
        conf = det.get('score', 0.0)

        if conf < min_conf:
            continue
        if mask_region is not None and not is_detection_in_mask(det, mask_region, overlap_threshold, min_conf):
            continue

        x, y, w, h = det['bbox']
        center_points.append([x + w / 2, y + h / 2])
        valid_confidences.append(conf)

    avg_confidence = np.mean(valid_confidences) if valid_confidences else 0.0

    if len(center_points) < min_detections:
        return [], avg_confidence

    center_points = np.array(center_points)
    center_points = center_points[np.argsort(center_points[:, 1])]

    try:
        degree = 1
        poly_coeffs = np.polyfit(center_points[:, 1], center_points[:, 0], degree)
        y_min, y_max = np.min(center_points[:, 1]), np.max(center_points[:, 1])
        y_range = np.linspace(y_min, y_max, 100)
        x_range = np.polyval(poly_coeffs, y_range)
        return list(zip(x_range.astype(int), y_range.astype(int))), avg_confidence
    except Exception as e:
        return [], avg_confidence


def combine_segmentation_detection_centerline(seg_centerline_points, det_centerline_points,
                                              conf_seg, conf_det,
                                              num_combined_points=100):
    """
    Adaptive fused method
    """
    if not seg_centerline_points and not det_centerline_points:
        return []
    if not seg_centerline_points:
        return det_centerline_points
    if not det_centerline_points:
        return seg_centerline_points

    epsilon = 1e-9
    total_conf = conf_seg + conf_det + epsilon
    w_seg = conf_seg / total_conf
    w_det = 1 - w_seg

    try:
        seg_points_np = np.array(seg_centerline_points)
        det_points_np = np.array(det_centerline_points)
        seg_points_np = seg_points_np[np.argsort(seg_points_np[:, 1])]
        det_points_np = det_points_np[np.argsort(det_points_np[:, 1])]

        seg_y_min, seg_y_max = np.min(seg_points_np[:, 1]), np.max(seg_points_np[:, 1])
        det_y_min, det_y_max = np.min(det_points_np[:, 1]), np.max(det_points_np[:, 1])
        combined_y_min = min(seg_y_min, det_y_min)
        combined_y_max = max(seg_y_max, det_y_max)

        if combined_y_min >= combined_y_max:
            return seg_centerline_points

        y_combined_smooth = np.linspace(combined_y_min, combined_y_max, num_combined_points)
        x_combined_smooth = np.zeros_like(y_combined_smooth)

        interp_seg_x = lambda y_val: np.interp(y_val, seg_points_np[:, 1], seg_points_np[:, 0])
        interp_det_x = lambda y_val: np.interp(y_val, det_points_np[:, 1], det_points_np[:, 0])

        for i, y_curr in enumerate(y_combined_smooth):
            in_seg_range = seg_y_min <= y_curr <= seg_y_max
            in_det_range = det_y_min <= y_curr <= det_y_max
            if in_seg_range and in_det_range:
                x_s = interp_seg_x(y_curr)
                x_d = interp_det_x(y_curr)
                x_combined_smooth[i] = w_seg * x_s + w_det * x_d
            elif in_seg_range:
                x_combined_smooth[i] = interp_seg_x(y_curr)
            elif in_det_range:
                x_combined_smooth[i] = interp_det_x(y_curr)
            else:
                x_combined_smooth[i] = interp_seg_x(y_curr)

        return list(zip(x_combined_smooth.astype(int), y_combined_smooth.astype(int)))

    except Exception:
        return seg_centerline_points if seg_centerline_points else det_centerline_points


def process_and_visualize_on_image(input_image, yolo_masks_xy, yolo_masks_conf, yolo_boxes, yolo_boxes_conf, height,
                                   width,
                                   num_y_samples=50, smooth_boundaries=True, left_color=(255, 100, 100),
                                   right_color=(100, 100, 255), centerline_color=(0, 255, 0),
                                   detection_min_conf=0.25, detection_overlap_threshold=0.5,
                                   segmentation_min_conf=0.25):
    output_image = input_image.copy()

    best_mask = np.zeros((height, width), dtype=np.uint8)
    main_road_contour = None
    conf_seg = 0.0
    if yolo_masks_xy and yolo_masks_conf:
        valid_masks_indices = [i for i, conf in enumerate(yolo_masks_conf) if conf >= segmentation_min_conf]
        if valid_masks_indices:
            max_conf_idx = max(valid_masks_indices, key=lambda i: yolo_masks_conf[i])
            main_road_contour = yolo_masks_xy[max_conf_idx].astype(np.int32)
            conf_seg = yolo_masks_conf[max_conf_idx]
            cv2.drawContours(best_mask, [main_road_contour], -1, 255, thickness=cv2.FILLED)

    seg_centerline_points = []
    if main_road_contour is not None and len(main_road_contour) >= 3:
        left_boundary_pts, right_boundary_pts = get_lr_boundary_points_from_contour(main_road_contour, num_y_samples,
                                                                                    height)

        for pts, color in [(left_boundary_pts, left_color), (right_boundary_pts, right_color)]:
            if pts.size > 0:
                pts = pts[np.argsort(pts[:, 1])]
                if smooth_boundaries and len(pts) >= 4:
                    unique_y, idx = np.unique(pts[:, 1], return_index=True)
                    if len(unique_y) >= 4:
                        try:
                            spline = UnivariateSpline(unique_y, pts[idx, 0], s=len(unique_y) * 2,
                                                      k=min(3, len(unique_y) - 1))
                            smooth_y = np.linspace(min(unique_y), max(unique_y), num_y_samples).astype(int)
                            smooth_x = spline(smooth_y)
                            pts = np.vstack((smooth_x, smooth_y)).T.astype(np.int32)
                        except Exception:
                            pass
                cv2.polylines(output_image, [pts], isClosed=False, color=color, thickness=4)

        if left_boundary_pts.size > 0 and right_boundary_pts.size > 0:
            left_dict = {int(pt[1]): pt[0] for pt in left_boundary_pts}
            right_dict = {int(pt[1]): pt[0] for pt in right_boundary_pts}
            common_y = sorted(set(left_dict.keys()) & set(right_dict.keys()))
            seg_centerline_points = [(int((left_dict[y] + right_dict[y]) / 2), y) for y in common_y]

    detections_list = [{'bbox': box, 'score': conf} for box, conf in zip(yolo_boxes, yolo_boxes_conf)]
    det_centerline_points, conf_det = extract_detection_centerline(
        detections=detections_list,
        mask_region=best_mask if np.any(best_mask) else None,
        min_detections=2,
        overlap_threshold=detection_overlap_threshold,
        min_conf=detection_min_conf
    )

    combined_centerline_points = combine_segmentation_detection_centerline(
        seg_centerline_points,
        det_centerline_points,
        conf_seg,
        conf_det,
        num_combined_points=num_y_samples
    )

    if combined_centerline_points:
        points_np = np.array(combined_centerline_points, dtype=np.int32)
        if len(points_np) > 1:
            cv2.polylines(output_image, [points_np], isClosed=False, color=centerline_color, thickness=4,
                          lineType=cv2.LINE_AA)

    return output_image, combined_centerline_points


def run_yolo_and_postprocess():
    model_path = r"/home/yy/YOLOv8-multi-task-main/runs/20250406-n-DICS_Res_add_v11_1_Seg_1_DGCST_3/weights/best.engine"
    source_path = r'/home/yy/YOLOv8-multi-task-main/test'
    output_dir = '/home/yy/YOLOv8-multi-task-main/runs/navigation_output_efficient'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        model = YOLO(model_path, task="multi")
    except Exception as e:
        print(f"model loading fail: {e}")
        return

    results_generator = model.predict(source=source_path, device=[0], imgsz=640, conf=0.25, stream=True, save=False)

    for i, results_list in enumerate(results_generator):
        if not isinstance(results_list, list):
            continue

        for idx, result in enumerate(results_list):
            if not isinstance(result, list) or len(result) < 2:
                continue

            actual_result = result[0]
            if not isinstance(actual_result, Results):
                continue

            mask_tensor = result[1]
            if not isinstance(mask_tensor, torch.Tensor):
                continue

            orig_img_bgr = actual_result.orig_img
            img_filename = actual_result.path
            if orig_img_bgr is None:
                continue

            height, width = orig_img_bgr.shape[:2]

            yolo_masks_xy_all = []
            yolo_masks_conf_all = []

            mask_confs = []
            if actual_result.masks and actual_result.masks.conf is not None:
                mask_confs = actual_result.masks.conf.cpu().numpy()
            else:
                # if model not provide mask conf, default set to 1
                num_masks = mask_tensor.shape[0] if len(mask_tensor.shape) == 3 else 1
                mask_confs = [1.0] * num_masks

            if len(mask_tensor.shape) == 3:  # [num_masks, height, width]
                for k in range(mask_tensor.shape[0]):
                    mask_k_binary = (mask_tensor[k].cpu().numpy() > 0.5).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_k_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                        yolo_masks_xy_all.append(main_contour)
                        yolo_masks_conf_all.append(mask_confs[k])
            elif len(mask_tensor.shape) == 2:  # [height, width]
                mask_binary = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
                    yolo_masks_xy_all.append(main_contour)
                    yolo_masks_conf_all.append(mask_confs[0])

            yolo_boxes_coords_all = []
            yolo_boxes_conf_all = []
            if actual_result.boxes is not None:
                boxes_xywh_abs = actual_result.boxes.xywh.cpu().numpy()
                yolo_boxes_conf_all = actual_result.boxes.conf.cpu().numpy().tolist()
                for j in range(len(boxes_xywh_abs)):
                    xc, yc, w, h_box = boxes_xywh_abs[j]
                    x_tl = xc - w / 2
                    y_tl = yc - h_box / 2
                    yolo_boxes_coords_all.append([x_tl, y_tl, w, h_box])

            processed_image, final_nav_line = process_and_visualize_on_image(
                input_image=orig_img_bgr,
                yolo_masks_xy=yolo_masks_xy_all,
                yolo_masks_conf=yolo_masks_conf_all,
                yolo_boxes=yolo_boxes_coords_all,
                yolo_boxes_conf=yolo_boxes_conf_all,
                height=height,
                width=width,
                num_y_samples=70,
                smooth_boundaries=True,
                left_color=(255, 100, 100),
                right_color=(100, 100, 255),
                centerline_color=(255, 0, 255),
                detection_min_conf=0.25,
                segmentation_min_conf=0.01,
                detection_overlap_threshold=0.5
            )

            base_filename = os.path.splitext(os.path.basename(img_filename))[0]
            output_path = os.path.join(output_dir, f"{base_filename}_nav_processed.png")
            cv2.imwrite(output_path, processed_image)


if __name__ == '__main__':
    run_yolo_and_postprocess()