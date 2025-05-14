import math
import numpy as np
from boundbox_algo import resize_bbox, crop_upper_bbox, make_square_bbox
import logging

# Define confidence threshold (consistent with label_json.py)
CONF_THRESH = 0.75

def class_split(result_boxes, result_scores, result_classid):
    result_boxes_adult, result_scores_adult, result_boxes_child, result_scores_child = [], [], [], []
    # Split the list to 2 which are adult and child, child as 1, adult as 0
    for object in range(len(result_classid)):
        if result_classid[object] == 1:
            result_boxes_child.append(result_boxes[object])
            result_scores_child.append(result_scores[object])
        elif result_classid[object] == 0:
            result_boxes_adult.append(result_boxes[object])
            result_scores_adult.append(result_scores[object])

    result_boxes_adult = np.array(result_boxes_adult)
    result_scores_adult = np.array(result_scores_adult)
    result_boxes_child = np.array(result_boxes_child)
    result_scores_child = np.array(result_scores_child)

    return result_boxes_adult, result_scores_adult, result_boxes_child, result_scores_child

def process_detections(result_boxes_adult, result_scores_adult, result_boxes_child, result_scores_child,
                      bbox_resize_factor=1.0, height=None, width=None, square=False, upper_crop=False, crop_ratio=1.6):
    # Handling the boundbox attribute, change at here for boundbox adjustments.
    id_counter = 0
    ents = []
    # Process adult detections
    for box, score in zip(result_boxes_adult, result_scores_adult):
        if score < CONF_THRESH:  # Filter low-confidence detections
            continue
        id_counter += 1
        original_box = box.tolist()  # 儲存原始 YOLO 邊界框
        resized_box = original_box  # 預設使用原始邊界框
        processing_applied = False

        if upper_crop and height is not None and width is not None:
            resized_box = crop_upper_bbox(resized_box, width, height, crop_ratio)
            processing_applied = True
            logging.debug(f"Adult ID {id_counter}: Applied upper_crop, bbox={resized_box}")
        if square and height is not None and width is not None:
            resized_box = make_square_bbox(resized_box, width, height)
            processing_applied = True
            logging.debug(f"Adult ID {id_counter}: Applied square, bbox={resized_box}")
        elif bbox_resize_factor != 1.0 and height is not None and width is not None:
            resized_box = resize_bbox(resized_box, bbox_resize_factor, width, height)
            processing_applied = True
            logging.debug(f"Adult ID {id_counter}: Applied resize, bbox={resized_box}")

        # 確保 resized_box 是 Python list
        if isinstance(resized_box, np.ndarray):
            resized_box = resized_box.tolist()

        # 驗證正方形框（僅當 square=True 時）
        if square:
            x1, y1, x2, y2 = resized_box
            if abs((x2 - x1) - (y2 - y1)) > 1:
                logging.warning(f"Adult ID {id_counter}: Non-square bbox after make_square_bbox: {resized_box}")
        elif not processing_applied:
            logging.debug(f"Adult ID {id_counter}: No processing applied, using original bbox={resized_box}")

        ents.append({
            'bbox': resized_box,  # 處理後的邊界框，用於顯示
            'original_bbox': original_box,  # 原始邊界框，用於儲存
            'class': 0,  # Adult (CLASS_ADULT)
            'score': float(score),
            'id': id_counter
        })

    # Process child detections
    for box, score in zip(result_boxes_child, result_scores_child):
        if score < CONF_THRESH:  # Filter low-confidence detections
            continue
        id_counter += 1
        original_box = box.tolist()  # 儲存原始 YOLO 邊界框
        resized_box = original_box  # 預設使用原始邊界框
        processing_applied = False

        if upper_crop and height is not None and width is not None:
            resized_box = crop_upper_bbox(resized_box, width, height, crop_ratio)
            processing_applied = True
            logging.debug(f"Child ID {id_counter}: Applied upper_crop, bbox={resized_box}")
        if square and height is not None and width is not None:
            resized_box = make_square_bbox(resized_box, width, height)
            processing_applied = True
            logging.debug(f"Child ID {id_counter}: Applied square, bbox={resized_box}")
        elif bbox_resize_factor != 1.0 and height is not None and width is not None:
            resized_box = resize_bbox(resized_box, bbox_resize_factor, width, height)
            processing_applied = True
            logging.debug(f"Child ID {id_counter}: Applied resize, bbox={resized_box}")

        # 確保 resized_box 是 Python list
        if isinstance(resized_box, np.ndarray):
            resized_box = resized_box.tolist()

        # 驗證正方形框（僅當 square=True 時）
        if square:
            x1, y1, x2, y2 = resized_box
            if abs((x2 - x1) - (y2 - y1)) > 1:
                logging.warning(f"Child ID {id_counter}: Non-square bbox after make_square_bbox: {resized_box}")
        elif not processing_applied:
            logging.debug(f"Child ID {id_counter}: No processing applied, using original bbox={resized_box}")

        ents.append({
            'bbox': resized_box,  # 處理後的邊界框，用於顯示
            'original_bbox': original_box,  # 原始邊界框，用於儲存
            'class': 1,  # Child (CLASS_CHILD)
            'score': float(score),
            'id': id_counter
        })
    return ents