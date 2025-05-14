import os
import logging
import json
import glob
import random

def save_categories(categories, dataset_path):
    """Save categories to a JSON file."""
    categories_path = os.path.join(dataset_path, 'categories.json')
    with open(categories_path, 'w') as f:
        json.dump(categories, f)
    logging.info(f"Saved categories to {categories_path}")

def label_frames(box_regions, frameno, key, video_path, dataset_path, categories, detected_id, ent):
    """Process labeling input for a specific person and save to a dedicated JSON file."""
    has_label = False
    cls = None

    if key in [ord('0'), ord('1'), ord('2')]:
        cls = int(chr(key))
        has_label = True
        logging.info(f"Frame {frameno}, Person ID {detected_id} labeled as {categories[cls]}, Original Bbox: {ent.get('original_bbox', ent['bbox'])}, Processed Bbox: {ent['bbox']}")
        save_json_annotation(ent, video_path, frameno, dataset_path, cls, detected_id)

    return has_label, cls

def save_json_annotation(ent, video_name, frame_no, dataset_path, tsm_class, ppl_id):
    """Save a single detection result as a JSON file with TSM dataset naming."""
    json_data = {
        "detection": [],
        "video_name": os.path.basename(video_name),
        "frame": frame_no,
        "tsm_class": tsm_class
    }
    # 使用 original_bbox（如果存在），否則回退到 bbox
    bbox = ent.get('original_bbox', ent['bbox'])
    x1, y1, x2, y2 = bbox
    detection = {
        "value": {
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1)
        }
    }
    json_data["detection"].append(detection)

    json_dir = os.path.join(dataset_path, "annotations")
    os.makedirs(json_dir, exist_ok=True)
    video_name_base = os.path.splitext(os.path.basename(video_name))[0]
    json_filename = f"cls{tsm_class}_{video_name_base}_frame{frame_no:06d}_ppl{ppl_id}.json"
    json_path = os.path.join(json_dir, json_filename)

    temp_ppl_id = ppl_id
    while os.path.exists(json_path):
        temp_ppl_id += 1
        json_filename = f"cls{tsm_class}_{video_name_base}_frame{frame_no:06d}_ppl{temp_ppl_id}.json"
        json_path = os.path.join(json_dir, json_filename)

    try:
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        logging.info(f"Saved JSON annotation: {json_path} with tsm_class={tsm_class}, bbox={bbox}")
    except Exception as e:
        logging.error(f"Failed to save JSON to {json_path}: {e}")

    return temp_ppl_id

def gen_txt(val_portion, dataset_path, images_path):
    """Generate train.txt and val.txt based on annotations."""
    logging.info(f"Generating train.txt and val.txt in {dataset_path}")
    train_file = os.path.join(dataset_path, 'train.txt')
    val_file = os.path.join(dataset_path, 'val.txt')
    annotations_dir = os.path.join(dataset_path, 'annotations')
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))

    if not json_files:
        logging.warning(f"No JSON files found in {annotations_dir}")
        return

    random.shuffle(json_files)
    val_count = int(len(json_files) * val_portion)
    val_files = json_files[:val_count]
    train_files = json_files[val_count:]

    with open(train_file, 'w') as f:
        for json_path in train_files:
            f.write(f"{json_path}\n")
    with open(val_file, 'w') as f:
        for json_path in val_files:
            f.write(f"{json_path}\n")
    logging.info(f"Generated {len(train_files)} train and {len(val_files)} val entries")