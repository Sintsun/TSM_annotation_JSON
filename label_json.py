import argparse
import ctypes
import cv2
import numpy as np
import json
import os
import logging
from collections import deque
from utils import *
from yolov7_trt import YoLov7_TRT
from test import class_split, process_detections
from labeller import gen_txt, label_frames, save_categories, save_json_annotation
from boundbox_algo import combine_overlapping_boxes, crop_upper_bbox, make_square_bbox, draw_bounding_boxes
from boundbox_algo import resize_bbox, crop_upper_bbox, make_square_bbox, expand_crop_bbox  # 添加 expand_crop_bbox
# Detection and Tracking Thresholds
CONF_THRESH = 0.75
IOU_THRESHOLD = 0.2
DEL_THRESHOLD = 20

# Define Class IDs
CLASS_ADULT = 0
CLASS_CHILD = 1

# Categories for labeling
categories = ["Normal", "Hitting", "Shaking"]

# Desired size of image output
image_width, image_height = 400, 400

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv7 Detection and TSM Annotation Script")
    parser.add_argument("--yolov7_engine_file_path",
                        default='/media/nvidia/128 GB/tsm_pipline/2_tsm_annotate/engine_plugin/childabuse_28032025_re.engine',
                        help="Path to YOLOv7 TensorRT engine file")
    parser.add_argument("--plugin_library", default="./engine_plugin/libmyplugins.so",
                        help="Path to custom plugin library")
    parser.add_argument("--temporal_size", type=int, default=24, help="Temporal size for TSM inference")
    parser.add_argument("--input_video_path", type=lambda x: int(x) if x.isdigit() else x,
                        default="/media/nvidia/4A6E-1997/Data_waiting_toprocess/lf_sahk/TSM/lok-fu/L4_2024_11_18__17_14_51.mkv",
                        help="Path to video, or folder of images, or webcam")
    parser.add_argument("--dataset_path", default="datasets_tsm/test_json",
                        help="Path to dataset storing the annotations")
    parser.add_argument('--remove_child', action='store_true', help="Remove entities with class 'child'")
    parser.add_argument("--square", action="store_true", help="Make the bounding box become a square using make_square_bbox")
    parser.add_argument("--upper_crop", action="store_true", help="Enable cropping for upper bounding boxes")
    parser.add_argument("--bbox_resize_factor", type=float, default=1.0, help="Factor to resize bounding boxes (ignored if --square is enabled)")
    parser.add_argument("--crop_ratio", type=float, default=1.6, help="Height-to-width ratio threshold for cropping")
    return parser.parse_args()

def draw_bounding_boxes(frame, ents):
    """
    Draw bounding boxes and labels on the frame using processed bbox.
    """
    class_map = {
        0: 'adult',
        1: 'child'
    }

    for ent in ents:
        bbox = ent['bbox']  # 使用處理後的 bbox 進行顯示（可能是原始或正方形）
        class_label = ent['class']
        score = ent['score']
        detected_id = ent['id']
        x1, y1, x2, y2 = map(int, bbox)

        if isinstance(class_label, int):
            class_label = class_map.get(class_label, 'unknown')
        else:
            class_label = class_label.lower()

        color = {'adult': (0, 255, 0), 'child': (0, 0, 255)}.get(class_label, (255, 0, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        display_label = class_label.capitalize()
        label = f"{display_label} ID:{detected_id} {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def show_past_frames_bounding_boxes_and_label(current_ents, past_frames, width, height, frameno, video_path,
                                              dataset_path, temporal_size):
    if len(past_frames) < temporal_size:
        print(f"Not enough past frames to display (requires {temporal_size} frames).")
        return

    class_map = {0: 'Adult', 1: 'Child'}

    processed_entities = []
    for ent in current_ents:
        bbox = ent['bbox']  # 使用處理後的 bbox 進行顯示（可能是原始或正方形）
        class_label = ent['class']
        detected_id = ent['id']

        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        box_regions = []
        for past_frame, past_ents in past_frames:
            if y2 > past_frame.shape[0] or x2 > past_frame.shape[1]:
                box_region = None
            else:
                box_region = past_frame[y1:y2, x1:x2].copy()
            box_regions.append(box_region)

        processed_entities.append({
            'class_label': class_label,
            'detected_id': detected_id,
            'box_regions': box_regions,
            'width': x2 - x1,
            'height': y2 - y1,
            'ent': ent
        })

    if not processed_entities:
        print("No entities to display.")
        return

    current_entity_index = 0
    total_entities = len(processed_entities)
    labeled_entities = set()

    while current_entity_index < total_entities:
        entity = processed_entities[current_entity_index]
        class_label = entity['class_label']
        detected_id = entity['detected_id']
        box_regions = entity['box_regions']
        box_width = entity['width']
        box_height = entity['height']
        ent = entity['ent']

        display_label = class_map.get(class_label, 'Unknown')
        label_status = "Labeled" if detected_id in labeled_entities else "Unlabeled"
        window_name = f"{display_label} ID:{detected_id}, {label_status}, Press r=next, q=quit, 0=Normal, 1=Hitting, 2=Shaking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 448, 448)

        region_index = 0
        num_regions = len(box_regions)
        has_label = False

        while True:
            region = box_regions[region_index]
            if region is None or region.size == 0:
                blank_image = np.zeros((box_height, box_width, 3), dtype=np.uint8)
                cv2.putText(blank_image, "No Data", (10, box_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                display_image = blank_image
            else:
                display_image = region.copy()
                # 在 region 上繪製邊界框（紅色框線）
                cv2.rectangle(display_image, (0, 0), (box_width-1, box_height-1), (0, 0, 255), 1)

            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(200) & 0xFF

            if key == ord('q'):
                print("Exit requested.")
                cv2.destroyAllWindows()
                if len(labeled_entities) < total_entities:
                    logging.warning(f"Frame {frameno}: Not all entities labeled. Labeled IDs: {labeled_entities}")
                return
            elif key == ord('r'):
                print("Skipping to next bounding box.")
                break

            if not has_label:
                has_label, cls = label_frames(
                    box_regions, frameno, key, video_path, dataset_path, categories, detected_id, ent
                )
                if has_label:
                    labeled_entities.add(detected_id)
                    break

            region_index = (region_index + 1) % num_regions

        cv2.destroyWindow(window_name)
        current_entity_index = (current_entity_index + 1) % total_entities

    if len(labeled_entities) < total_entities:
        logging.warning(f"Frame {frameno}: Not all entities labeled. Labeled IDs: {labeled_entities}")

def filter_ents(ents, remove_child=False):
    """Filter entities, removing 'child' class if remove_child is True"""
    if remove_child:
        return [item for item in ents if item.get('class') != CLASS_CHILD]
    return ents

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_arguments()
    if args is None:
        logging.error("Failed to parse arguments. Exiting.")
        return

    # 記錄命令列參數
    logging.info(f"Parsed arguments: dataset_path={args.dataset_path}, square={args.square}, "
                 f"upper_crop={args.upper_crop}, bbox_resize_factor={args.bbox_resize_factor}, "
                 f"crop_ratio={args.crop_ratio}, remove_child={args.remove_child}")

    try:
        save_categories(categories, args.dataset_path)
    except AttributeError as e:
        logging.error(f"Error accessing dataset_path: {e}")
        return

    yolov7_engine_file_path = args.yolov7_engine_file_path
    PLUGIN_LIBRARY = args.plugin_library
    temporal_size = args.temporal_size
    input_video_path = args.input_video_path
    dataset_path = args.dataset_path
    remove_child = args.remove_child
    bbox_resize_factor = args.bbox_resize_factor
    square = args.square
    upper_crop = args.upper_crop
    crop_ratio = args.crop_ratio

    logging.info(f"YOLOv7 Engine File Path: {yolov7_engine_file_path}")
    logging.info(f"Plugin Library Path: {PLUGIN_LIBRARY}")
    logging.info(f"Temporal Size: {temporal_size}")
    logging.info(f"Input Video Path: {input_video_path}")
    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Remove Child: {remove_child}")
    logging.info(f"Bounding Box Resize Factor: {bbox_resize_factor}")
    logging.info(f"Square: {square}, Upper Crop: {upper_crop}, Crop Ratio: {crop_ratio}")

    images_dataset_path = os.path.join(dataset_path, 'images')
    os.makedirs(images_dataset_path, exist_ok=True)
    logging.info(f"Images dataset directory: {images_dataset_path}")

    try:
        ctypes.CDLL(PLUGIN_LIBRARY)
        logging.info(f"Successfully loaded plugin library: {PLUGIN_LIBRARY}")
    except OSError as e:
        logging.error(f"Error loading plugin library: {e}")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video source: {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        logging.error(f"Invalid video dimensions: width={width}, height={height}")
        cap.release()
        return
    logging.info(f"Video Details - FPS: {fps}, Resolution: {width}x{height}, Total Frames: {total_frame}")

    frameno = 0
    yolov7_wrapper = None
    past_frames = deque(maxlen=temporal_size)

    try:
        yolov7_wrapper = YoLov7_TRT(yolov7_engine_file_path, CONF_THRESH, IOU_THRESHOLD)
        logging.info("YOLOv7 TensorRT Wrapper Initialized.")

        while True:
            frameno += 1
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video or cannot fetch the frame.")
                break

            img = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result_boxes, result_scores, result_classid, yolo_infer_time = yolov7_wrapper.infer(img)
            logging.info(f'Frame {frameno}: Detected class IDs: {result_classid}, Count: {len(result_classid)}')
            logging.info(
                f'Frame {frameno}: Scores range: min={np.min(result_scores) if len(result_scores) > 0 else 0}, max={np.max(result_scores) if len(result_scores) > 0 else 0}')

            result_boxes_adult, result_scores_adult, result_boxes_child, result_scores_child = class_split(
                result_boxes, result_scores, result_classid)

            ents = process_detections(result_boxes_adult, result_scores_adult, result_boxes_child, result_scores_child,
                                     bbox_resize_factor, height, width, square, upper_crop, crop_ratio)
            ents = filter_ents(ents, remove_child)
            logging.info(f'Frame {frameno}: Filtered ents count: {len(ents)}')

            past_frames.append((frame.copy(), list(ents)))

            draw_bounding_boxes(frame, ents)
            frame = cv2.resize(frame, (1440, 900))

            cv2.imshow(
                f"SpaceBar:Next frame, P:Quit, R:Show Past {temporal_size} Frames",
                frame)
            key = cv2.waitKey(0)

            if key == ord('p'):
                logging.info("Exit requested by user.")
                break
            elif key == ord('r'):
                logging.info(f'Displaying past {temporal_size} frames bounding box regions.')
                show_past_frames_bounding_boxes_and_label(ents, past_frames, width, height, frameno, input_video_path,
                                                         dataset_path, temporal_size)
                continue
            elif key == ord('n'):
                continue
            else:
                logging.info("Unhandled key pressed. Continuing to next frame.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
    finally:
        logging.info("End of processing. Cleaning up...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if yolov7_wrapper is not None:
            yolov7_wrapper.destroy()
        gen_txt(val_portion=0.1, dataset_path=dataset_path, images_path=images_dataset_path)
        logging.info("Processing completed and resources released.")

if __name__ == "__main__":
    main()