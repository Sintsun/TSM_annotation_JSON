import argparse
import cv2
import numpy as np
import json
import os
import glob
import logging
from collections import deque
from labeller import gen_txt
from boundbox_algo import resize_bbox, crop_upper_bbox, make_square_bbox, expand_crop_bbox  # 添加 expand_crop_bbox
def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset Generation Script for TSM from JSON Annotations")
    parser.add_argument("--video_folder_path", type=str, default="/media/nvidia/128 GB/PLK_02042025/plk-ncw",
                        help="Path to folder containing video files")
    parser.add_argument("--dataset_path", default="datasets_tsm/test_json",
                        help="Path to dataset containing annotations and images")
    parser.add_argument("--temporal_size", type=int, default=24, help="Temporal size for TSM inference")
    parser.add_argument("--square", action="store_true", help="Make the bounding box become a square instead of rectangle.")
    parser.add_argument("--upper_crop", action="store_true", help="Enable the cropping function for upper bounding boxes.")
    parser.add_argument("--bbox_resize_factor", "-re", type=float, default=1.0,
                        help="Factor to resize bounding boxes. >1.0 to enlarge, <1.0 to shrink.")
    parser.add_argument("--crop_ratio", type=float, default=1.6, help="Height-to-width ratio threshold for cropping.")
    parser.add_argument("--expand_crop_factor", type=float, default=1.0,
                        help="Factor to expand or shrink the cropping area. >1.0 to enlarge, <1.0 to shrink.")
    return parser.parse_args()

def load_json_annotations(annotations_dir, video_folder_path):
    """Load JSON annotations, filter by existing video files."""
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    annotations = {}
    valid_videos = {os.path.basename(f) for f in glob.glob(os.path.join(video_folder_path, "*.mkv"))}

    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            required_fields = ['video_name', 'frame', 'tsm_class', 'detection']
            for field in required_fields:
                if field not in data:
                    logging.error(f"Skipping JSON {json_path}: Missing required field '{field}'")
                    continue

            video_name = data['video_name']
            tsm_class = data['tsm_class']
            frame_no = data['frame']

            if video_name not in valid_videos:
                logging.warning(f"Skipping JSON {json_path}: Video {video_name} not found in {video_folder_path}")
                continue

            try:
                ppl_id = int(os.path.basename(json_path).split('_ppl')[1].split('.')[0])
            except (IndexError, ValueError):
                logging.error(f"Skipping JSON {json_path}: Invalid ppl_id in filename")
                continue

            if video_name not in annotations:
                annotations[video_name] = {}
            if frame_no not in annotations[video_name]:
                annotations[video_name][frame_no] = []
            annotations[video_name][frame_no].append({
                'detections': data['detection'],
                'tsm_class': tsm_class,
                'ppl_id': ppl_id,
                'json_path': json_path
            })
        except json.JSONDecodeError as e:
            logging.error(f"Skipping JSON {json_path}: Invalid JSON format - {str(e)}")
            continue
        except Exception as e:
            logging.error(f"Error loading JSON {json_path}: {str(e)}")
            continue

    if not annotations:
        logging.warning(f"No valid JSON annotations found in {annotations_dir}")
    else:
        logging.info(f"Loaded {len(json_files)} JSON annotations, {len(annotations)} videos with valid files")
    return annotations

def process_video(video_path, annotations, dataset_path, temporal_size, square, upper_crop, bbox_resize_factor, crop_ratio, expand_crop_factor):
    """處理視頻，僅提取有標註的幀的24幀序列"""
    video_name = os.path.basename(video_path)
    images_dataset_path = os.path.join(dataset_path, 'images')
    os.makedirs(images_dataset_path, exist_ok=True)

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Error: Unable to open video {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Processing video: {video_name}, Resolution: {width}x{height}, Total Frames: {total_frames}")

        past_frames = deque(maxlen=temporal_size)
        frameno = 0

        video_annotations = annotations.get(video_name, {})
        if not video_annotations:
            logging.info(f"Skipping video {video_name}: No annotations found")
            cap.release()
            return

        annotated_frames = sorted([int(k) for k in video_annotations.keys()])
        max_annotated_frame = max(annotated_frames) if annotated_frames else 0
        logging.info(f"Video {video_name}: {len(annotated_frames)} annotated frames, max frame {max_annotated_frame}")

        while True:
            frameno += 1
            ret, frame = cap.read()
            if not ret:
                logging.info(f"End of video {video_name} at frame {frameno-1}")
                break

            past_frames.append(frame if frame is not None else None)

            if frameno in video_annotations:
                logging.debug(f"Processing annotated frame {frameno}")
                for anno in video_annotations[frameno]:
                    tsm_class = anno['tsm_class']
                    ppl_id = anno['ppl_id']
                    detections = anno['detections']

                    if len(past_frames) < temporal_size or any(f is None for f in past_frames):
                        logging.debug(f"Frame {frameno}: Not enough valid past frames ({len(past_frames)}/{temporal_size})")
                        continue

                    for det in detections:
                        try:
                            bbox = det.get('value', {})
                            required_bbox_fields = ['x', 'y', 'width', 'height']
                            if not all(field in bbox for field in required_bbox_fields):
                                logging.warning(f"Frame {frameno}: Invalid bounding box - missing fields")
                                continue

                            x1 = bbox['x']
                            y1 = bbox['y']
                            x2 = x1 + bbox['width']
                            y2 = y1 + bbox['height']
                            logging.debug(f"Frame {frameno}, ppl_id {ppl_id}: Original bbox=[{x1}, {y1}, {x2}, {y2}]")

                            # 應用邊界框調整
                            bbox_adj = [x1, y1, x2, y2]
                            if upper_crop:
                                bbox_adj = crop_upper_bbox(bbox_adj, width, height, crop_ratio)
                                logging.debug(f"Frame {frameno}, ppl_id {ppl_id}: After upper_crop, bbox={bbox_adj}")
                            if expand_crop_factor != 1.0:
                                bbox_adj = expand_crop_bbox(bbox_adj, width, height, expand_crop_factor)
                                logging.debug(f"Frame {frameno}, ppl_id {ppl_id}: After expand_crop_bbox, bbox={bbox_adj}")
                            if square:
                                bbox_adj = make_square_bbox(bbox_adj, width, height)
                                logging.debug(f"Frame {frameno}, ppl_id {ppl_id}: After make_square_bbox, bbox={bbox_adj}")
                            elif bbox_resize_factor != 1.0:
                                bbox_adj = resize_bbox(bbox_adj, bbox_resize_factor, width, height)
                                logging.debug(f"Frame {frameno}, ppl_id {ppl_id}: After resize_bbox, bbox={bbox_adj}")

                            x1, y1, x2, y2 = map(int, bbox_adj)

                            # 驗證邊界框
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(width, x2)
                            y2 = min(height, y2)
                            if x2 <= x1 or y2 <= y1:
                                logging.warning(f"Invalid bounding box at frame {frameno}: ({x1}, {y1}, {x2}, {y2})")
                                continue

                            # 驗證正方形框
                            if square:
                                width_adj = x2 - x1
                                height_adj = y2 - y1
                                if abs(width_adj - height_adj) > 1:
                                    logging.warning(f"Frame {frameno}, ppl_id {ppl_id}: Non-square bbox after make_square_bbox: [{x1}, {y1}, {x2}, {y2}]")

                            # 創建子文件夾
                            video_name_base = os.path.splitext(video_name)[0]
                            subfolder_name = f"cls{tsm_class}_{video_name_base}_frame{frameno:06d}_ppl{ppl_id}"
                            subfolder_path = os.path.join(images_dataset_path, subfolder_name)
                            os.makedirs(subfolder_path, exist_ok=True)

                            # 提取並保存24幀序列
                            image_count = 0
                            for i, past_frame in enumerate(past_frames):
                                try:
                                    if past_frame is None:
                                        logging.warning(f"Frame {frameno}, index {i}: Past frame is None")
                                        continue
                                    if y2 > past_frame.shape[0] or x2 > past_frame.shape[1]:
                                        logging.warning(f"Frame {frameno}, index {i}: Bounding box out of bounds")
                                        continue
                                    region = past_frame[y1:y2, x1:x2]
                                    if region.size == 0:
                                        logging.warning(f"Frame {frameno}, index {i}: Empty region")
                                        continue
                                    img_filename = f"{i:02d}.jpg"
                                    img_path = os.path.join(subfolder_path, img_filename)
                                    if cv2.imwrite(img_path, region):
                                        image_count += 1
                                        logging.debug(f"Saved image: {img_path} (Size: {region.shape[1]}x{region.shape[0]})")
                                    else:
                                        logging.error(f"Failed to save image: {img_path}")
                                except Exception as e:
                                    logging.error(f"Error saving image {img_path}: {str(e)}")
                                    continue

                            anno['subfolder_name'] = subfolder_name
                            anno['image_count'] = image_count
                        except Exception as e:
                            logging.error(f"Error processing detection at frame {frameno}: {str(e)}")
                            continue

            if frameno >= max_annotated_frame and annotated_frames:
                logging.info(f"All annotated frames processed for {video_name} at frame {frameno}")
                break

        cap.release()
        logging.info(f"Completed processing video {video_name}")
    except Exception as e:
        logging.error(f"Error processing video {video_path}: {str(e)}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    args = parse_arguments()

    video_folder_path = args.video_folder_path
    dataset_path = args.dataset_path
    temporal_size = args.temporal_size
    square = args.square
    upper_crop = args.upper_crop
    bbox_resize_factor = args.bbox_resize_factor
    crop_ratio = args.crop_ratio
    expand_crop_factor = args.expand_crop_factor  # 新增

    logging.info(f"Video Folder Path: {video_folder_path}")
    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Temporal Size: {temporal_size}")
    logging.info(f"Square: {square}, Upper Crop: {upper_crop}, BBox Resize Factor: {bbox_resize_factor}, "
                 f"Crop Ratio: {crop_ratio}, Expand Crop Factor: {expand_crop_factor}")

    annotations_dir = os.path.join(dataset_path, 'annotations')
    if not os.path.exists(annotations_dir):
        logging.error(f"Annotations directory {annotations_dir} does not exist")
        return

    try:
        annotations = load_json_annotations(annotations_dir, video_folder_path)
        if not annotations:
            logging.error(f"No valid JSON annotations with matching videos found in {annotations_dir}")
            return

        video_files = glob.glob(os.path.join(video_folder_path, "*.mkv"))
        if not video_files:
            logging.error(f"No .mkv videos found in {video_folder_path}")
            return
        logging.info(f"Found {len(video_files)} video files")

        for video_path in video_files:
            video_name = os.path.basename(video_path)
            try:
                if video_name not in annotations:
                    logging.info(f"Skipping video {video_name}: No annotations found")
                    continue
                logging.info(f"Starting processing for video {video_name}")
                process_video(video_path, annotations, dataset_path, temporal_size, square, upper_crop,
                              bbox_resize_factor, crop_ratio, expand_crop_factor)  # 傳遞新參數
            except Exception as e:
                logging.error(f"Error processing video {video_path}: {str(e)}")
                continue

        images_dataset_path = os.path.join(dataset_path, 'images')
        try:
            logging.info("Generating train.txt and val.txt")
            gen_txt(val_portion=0.2, dataset_path=dataset_path, images_path=images_dataset_path)
        except Exception as e:
            logging.error(f"Error generating train.txt and val.txt: {str(e)}")
        logging.info("Dataset generation completed.")
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()