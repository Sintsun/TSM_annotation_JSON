# TSM Dataset Generation and Annotation Processing

This repository contains two Python scripts for generating datasets for Temporal Shift Module (TSM) processing and managing video annotations:

- **`generate_datasets.py`**: Processes video files and JSON annotations to extract 24-frame sequences, apply bounding box adjustments (e.g., square, upper crop, expand crop), and save cropped images for TSM inference.
- **`label_json.py`**: Creates or manages JSON annotation files containing original YOLO bounding boxes for video frames, serving as input for `generate_datasets.py`.

## Prerequisites

- **Python Version**: Python 3.8 or higher
- **Dependencies**: Install required packages using:
  ```bash
  pip install opencv-python numpy argparse
  ```
- **Hardware**: Sufficient storage for video files and generated images (e.g., 128 GB as per video folder path).
- **Input Files**:
  - Video files in `.mkv` format (e.g., `/media/nvidia/128 GB/PLK_02042025/plk-ncw`).
  - JSON annotation files in the dataset’s `annotations` directory (e.g., `datasets_tsm/test_json/annotations`).

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` with:
   ```
   opencv-python
   numpy
   argparse
   ```

3. **Prepare Input Data**:
   - Place video files in the specified folder (default: `/media/nvidia/128 GB/PLK_02042025/plk-ncw`).
   - Ensure JSON annotation files are in the dataset’s `annotations` directory, with the format:
     ```json
     {
       "video_name": "plk2f-r21_2025_02_06__15_48_32.mkv",
       "frame": 75,
       "tsm_class": 1,
       "detection": [
         {
           "value": {
             "x": 1368,
             "y": 728,
             "width": 311,
             "height": 350
           }
         }
       ]
     }
     ```

4. **Verify Directory Structure**:
   ```
   <project-root>/
   ├── generate_datasets.py
   ├── label_json.py
   ├── boundbox_algo.py
   ├── labeller.py
   ├── datasets_tsm/
   │   └── test_json/
   │       ├── annotations/
   │       └── images/
   └── /media/nvidia/128 GB/PLK_02042025/plk-ncw/
   ```

## Scripts Overview

### `generate_datasets.py`

**Purpose**: Processes video files using JSON annotations to extract 24-frame sequences around annotated frames, applies bounding box adjustments, and saves cropped images in subfolders for TSM training. Generates `train.txt` and `val.txt` for dataset splitting.

**Features**:
- Loads JSON annotations and matches them with `.mkv` video files.
- Supports bounding box adjustments:
  - `--square`: Converts bounding boxes to squares.
  - `--upper_crop`: Crops the upper body if the height-to-width ratio exceeds `crop_ratio`.
  - `--bbox_resize_factor`: Scales bounding boxes (default 1.0, no scaling).
  - `--expand_crop_factor`: Expands or shrinks the cropping area (default 1.0, no change).
- Saves 24-frame sequences as JPEG images in `datasets_tsm/test_json/images/cls<tsm_class>_<video_name>_frame<frame_no>_ppl<ppl_id>/`.
- Generates `train.txt` and `val.txt` with a 20% validation split.

**Usage**:
```bash
python generate_datasets.py --video_folder_path /media/nvidia/128 GB/PLK_02042025/plk-ncw --dataset_path datasets_tsm/test_json [options]
```

**Options**:
- `--video_folder_path`: Path to video files (default: `/media/nvidia/128 GB/PLK_02042025/plk-ncw`).
- `--dataset_path`: Path to dataset directory (default: `datasets_tsm/test_json`).
- `--temporal_size`: Number of frames per sequence (default: 24).
- `--square`: Enable square bounding boxes.
- `--upper_crop`: Enable upper body cropping.
- `--bbox_resize_factor`: Scale factor for bounding boxes (e.g., 1.2 to enlarge, 0.8 to shrink).
- `--crop_ratio`: Height-to-width ratio threshold for upper crop (default: 1.6).
- `--expand_crop_factor`: Scale factor for cropping area (e.g., 1.5 to enlarge, 0.8 to shrink, default: 1.0).

**Example Commands**:
- Process with original bounding boxes:
  ```bash
  python generate_datasets.py --dataset_path datasets_tsm/test_json --bbox_resize_factor 1.0 --expand_crop_factor 1.0
  ```
- Apply upper crop and expand crop:
  ```bash
  python generate_datasets.py --dataset_path datasets_tsm/test_json --upper_crop --expand_crop_factor 1.5
  ```
- Generate square bounding boxes:
  ```bash
  python generate_datasets.py --dataset_path datasets_tsm/test_json --square
  ```

**Output**:
- Cropped images in `datasets_tsm/test_json/images/`.
- `train.txt` and `val.txt` in `datasets_tsm/test_json/`.

### `label_json.py`

**Purpose**: Creates or manages JSON annotation files containing original YOLO bounding boxes for video frames, used as input for `generate_datasets.py`.

**Assumed Features**:
- Processes video frames to detect objects using YOLO (or similar model).
- Saves bounding box coordinates (`x`, `y`, `width`, `height`), frame number, video name, and TSM class in JSON format.
- Ensures original bounding boxes are recorded without adjustments.

**Usage**:
```bash
python label_json.py [options]
```

**Options** (assumed, please update based on actual implementation):
- `--video_path`: Path to input video or folder.
- `--output_dir`: Directory to save JSON annotations.
- `--model`: Path to YOLO model weights or configuration.

**Example Command**:
```bash
python label_json.py --video_path /media/nvidia/128 GB/PLK_02042025/plk-ncw --output_dir datasets_tsm/test_json/annotations
```

**Output**:
- JSON files in `datasets_tsm/test_json/annotations/` (e.g., `plk2f-r21_2025_02_06__15_48_32_ppl1.json`).

## Bounding Box Adjustments

The `generate_datasets.py` script uses functions from `boundbox_algo.py` to adjust bounding boxes:

- **`crop_upper_bbox`**: Crops the upper body if the height-to-width ratio exceeds `crop_ratio` (default 1.6), retaining 75% of the calculated height to focus on the upper body.
- **`expand_crop_bbox`**: Expands or shrinks the cropping area based on `expand_crop_factor`, keeping the center fixed.
- **`make_square_bbox`**: Converts bounding boxes to squares by using the larger of width or height as the side length, keeping the center fixed.
- **`resize_bbox`**: Scales bounding boxes by `bbox_resize_factor`.

**Behavior Without Parameters**:
- When `--square=False`, `--upper_crop=False`, `--bbox_resize_factor=1.0`, and `--expand_crop_factor=1.0`, the original bounding boxes from JSON are used for cropping and saving images.

## Troubleshooting

1. **Error: `name 'expand_crop_bbox' is not defined`**:
   - **Cause**: The `expand_crop_bbox` function is missing in `boundbox_algo.py` or not imported.
   - **Fix**:
     - Add the following to `boundbox_algo.py`:
       ```python
       import logging
       def expand_crop_bbox(bbox, image_width, image_height, expand_factor=1.0):
           x1, y1, x2, y2 = bbox
           width = x2 - x1
           height = y2 - y1
           center_x = x1 + width / 2
           center_y = y1 + height / 2
           new_width = width * expand_factor
           new_height = height * expand_factor
           new_x1 = center_x - new_width / 2
           new_y1 = center_y - new_height / 2
           new_x2 = center_x + new_width / 2
           new_y2 = center_y + new_height / 2
           new_x1 = max(0, new_x1)
           new_y1 = max(0, new_y1)
           new_x2 = min(image_width, new_x2)
           new_y2 = min(image_height, new_y2)
           logging.debug(f"expand_crop_bbox: Input bbox={bbox}, Expand_factor={expand_factor}, "
                         f"Center=({center_x:.2f}, {center_y:.2f}), Output bbox=[{new_x1:.2f}, {new_y1:.2f}, {new_x2:.2f}, {new_y2:.2f}]")
           if new_x2 <= new_x1 or new_y2 <= new_y1:
               logging.warning(f"Invalid bbox after expand_crop_bbox: [{new_x1}, {new_y1}, {new_x2}, {new_y2}]")
               return bbox
           return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
       ```
     - Update the import statement in `generate_datasets.py`:
       ```python
       from boundbox_algo import resize_bbox, crop_upper_bbox, make_square_bbox, expand_crop_bbox
       ```

2. **Upper Crop Not Applied**:
   - **Cause**: Bounding box height-to-width ratio is below `crop_ratio` (default 1.6).
   - **Fix**: Lower `crop_ratio` (e.g., `--crop_ratio 1.2`) or adjust `crop_upper_bbox` in `boundbox_algo.py`:
     ```python
     def crop_upper_bbox(bbox, image_width, image_height, ratio_threshold=1.6):
         x1, y1, x2, y2 = bbox
         width = x2 - x1
         height = y2 - y1
         if height >= ratio_threshold * width:
             new_height = width * ratio_threshold * 0.75
             y1 = y2 - new_height
             logging.debug(f"crop_upper_bbox: Cropped height from {height} to {new_height}")
         else:
             logging.debug(f"crop_upper_bbox: No cropping applied, ratio {height/width:.2f} < {ratio_threshold}")
         x1 = max(0, x1)
         y1 = max(0, y1)
         x2 = min(image_width, x2)
         y2 = min(image_height, y2)
         return [int(x1), int(y1), int(x2), int(y2)]
     ```

3. **Square Bounding Boxes Shift to Lower Body**:
   - **Cause**: `crop_upper_bbox` may shift the center point downward.
   - **Fix**: Ensure `make_square_bbox` and `expand_crop_bbox` keep the center fixed. Add visualization for debugging:
     ```python
     if frame is not None:
         frame_copy = frame.copy()
         cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
         cv2.imshow(f"Frame {frameno}, ppl_id {ppl_id}", frame_copy)
         cv2.waitKey(0)
         cv2.destroyAllWindows()
     ```

4. **Videos Skipped Due to No Annotations**:
   - **Cause**: JSON files missing for some videos (e.g., `plk2f-r21_2025_02_06__16_00_01.mkv`).
   - **Fix**: Run `label_json.py` to generate missing annotations or verify `annotations` directory.

## Testing

1. **Generate Dataset with Original Boxes**:
   ```bash
   python generate_datasets.py --dataset_path datasets_tsm/test_json --bbox_resize_factor 1.0 --expand_crop_factor 1.0
   ```
   - Verify: Images in `datasets_tsm/test_json/images/` match JSON bounding box dimensions.

2. **Test Upper Crop and Expand Crop**:
   ```bash
   python generate_datasets.py --dataset_path datasets_tsm/test_json --upper_crop --expand_crop_factor 1.5
   ```
   - Verify: Images are cropped to upper body and enlarged by 50%.

3. **Test Square Boxes**:
   ```bash
   python generate_datasets.py --dataset_path datasets_tsm/test_json --square
   ```
   - Verify: Images are square (width = height).

4. **Check Logs**:
   - Look for debug logs in the terminal or log file, e.g.:
     ```
     DEBUG: Frame 75, ppl_id 1: Original bbox=[1368, 728, 1679, 1078]
     DEBUG: Frame 75, ppl_id 1: After upper_crop, bbox=[1368, 728, 1679, 1078]
     DEBUG: Frame 75, ppl_id 1: After expand_crop_bbox, bbox=[...]
     ```

## Notes

- **JSON Annotations**: `label_json.py` must generate JSON files with original YOLO bounding boxes, as `generate_datasets.py` relies on these for processing.
- **Bounding Box Validation**: The script validates bounding boxes to ensure they are within image bounds and non-empty.
- **Performance**: Processing large videos may require significant disk space and memory. Test with a small subset first.

## Contributing

Please submit issues or pull requests for improvements, especially for `label_json.py` specifics or additional bounding box adjustments.

## License

This project is licensed under the MIT License.