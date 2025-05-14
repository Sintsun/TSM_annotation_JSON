# TSM Dataset Generation and Annotation Processing

This repository contains two Python scripts for generating datasets for Temporal Shift Module (TSM) processing and managing video annotations:

- **`label_json.py`**: Creates or manages JSON annotation files containing original YOLO bounding boxes for video frames, serving as input for `generate_datasets.py`.
- **`generate_datasets.py`**: Processes video files and JSON annotations to extract 24-frame sequences, apply bounding box adjustments (e.g., square, upper crop, expand crop), and save cropped images for TSM inference.


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



