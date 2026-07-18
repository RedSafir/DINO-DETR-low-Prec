import os
import json
import random
import yaml
from PIL import Image, ImageDraw, ImageFont

# Path configuration
DATASET_DIR = "./merged_yolo_person_ball"
OUTPUT_DIR = "."
SANITY_DIR = "./sanity_checks"

def convert_yolo_to_coco():
    print("==========================================================")
    # 1. Read class configuration
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        names = data_config.get("names", ["person", "ball"])
        # Handle dict or list format of names
        if isinstance(names, dict):
            categories = [{"id": int(k) + 1, "name": v, "supercategory": "none"} for k, v in names.items()]
            class_map = {int(k): int(k) + 1 for k in names.keys()}
        else:
            categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(names)]
            class_map = {i: i + 1 for i in range(len(names))}
    else:
        print("[WARNING] data.yaml not found. Using default mapping: {0: 'person', 1: 'ball'}")
        categories = [
            {"id": 1, "name": "person", "supercategory": "none"},
            {"id": 2, "name": "ball", "supercategory": "none"}
        ]
        class_map = {0: 1, 1: 2}

    print(f"Categories mapped: {categories}")

    # 2. Check dataset statistics (merge_stats.json)
    stats_path = os.path.join(DATASET_DIR, "merge_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Dataset stats loaded from merge_stats.json: {stats}")
    else:
        print("[INFO] merge_stats.json not found yet.")

    # 3. Locate images and labels
    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"[ERROR] Dataset paths do not exist. Please place your dataset in '{DATASET_DIR}' containing 'images/' and 'labels/'.")
        return

    # Gather matching image and label files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    valid_pairs = []
    
    for img_name in os.listdir(images_dir):
        if img_name.lower().endswith(image_extensions):
            base_name = os.path.splitext(img_name)[0]
            label_name = base_name + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            if os.path.exists(label_path):
                valid_pairs.append((img_name, label_name))

    print(f"Found {len(valid_pairs)} valid image-label pairs.")
    if len(valid_pairs) == 0:
        print("[ERROR] No image-label pairs found. Cannot convert.")
        return

    # 4. Shuffle dataset with fixed seed 42 BEFORE splitting to prevent train/val bias
    random.seed(42)
    # Sort first to ensure deterministic baseline across different OS file systems
    valid_pairs.sort(key=lambda x: x[0])
    random.shuffle(valid_pairs)

    split_idx = int(len(valid_pairs) * 0.9)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    print(f"Split: {len(train_pairs)} training images, {len(val_pairs)} validation images.")

    # Helper function to generate COCO JSON dict
    def build_coco_json(pairs, split_name):
        coco_output = {
            "info": {"description": f"Football Person and Ball dataset - {split_name} split"},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }

        ann_id_counter = 1
        for img_id, (img_name, label_name) in enumerate(pairs, 1):
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, label_name)

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"[WARNING] Skipping unreadable image {img_name}: {e}")
                continue

            coco_output["images"].append({
                "id": img_id,
                "file_name": img_name,
                "width": width,
                "height": height
            })

            # Read YOLO annotations
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w_norm, h_norm = map(float, parts)
                class_id = int(class_id)

                if class_id not in class_map:
                    print(f"[WARNING] Class ID {class_id} in {label_name} not mapped. Skipping.")
                    continue

                category_id = class_map[class_id]

                # Convert normalized YOLO coordinates to absolute COCO coordinates
                w_px = w_norm * width
                h_px = h_norm * height
                x_min = (x_center - w_norm / 2) * width
                y_min = (y_center - h_norm / 2) * height

                coco_output["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(w_px, 2), round(h_px, 2)],
                    "area": round(w_px * h_px, 2),
                    "iscrowd": 0
                })
                ann_id_counter += 1

        return coco_output

    print("Generating COCO JSON files...")
    train_coco = build_coco_json(train_pairs, "train")
    val_coco = build_coco_json(val_pairs, "val")

    with open(os.path.join(OUTPUT_DIR, "annotations_train.json"), 'w') as f:
        json.dump(train_coco, f)
    with open(os.path.join(OUTPUT_DIR, "annotations_val.json"), 'w') as f:
        json.dump(val_coco, f)

    print("[SUCCESS] annotations_train.json and annotations_val.json generated successfully!")

    # 5. Sanity-Check Plotting
    os.makedirs(SANITY_DIR, exist_ok=True)
    print(f"Creating sanity checks in '{SANITY_DIR}'...")
    
    # Select 5 random training pairs
    sanity_sample = train_pairs[:5] if len(train_pairs) >= 5 else train_pairs
    cat_names = {cat["id"]: cat["name"] for cat in categories}
    
    for idx, (img_name, label_name) in enumerate(sanity_sample, 1):
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, label_name)
        
        try:
            with Image.open(img_path) as img:
                draw = ImageDraw.Draw(img)
                width, height = img.size
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_center, y_center, w_norm, h_norm = map(float, parts)
                    class_id = int(class_id)
                    category_name = cat_names.get(class_map.get(class_id, -1), "unknown")
                    
                    # Compute absolute box
                    w_px = w_norm * width
                    h_px = h_norm * height
                    x0 = (x_center - w_norm / 2) * width
                    y0 = (y_center - h_norm / 2) * height
                    x1 = x0 + w_px
                    y1 = y0 + h_px
                    
                    # Colors: Person is Blue, Ball is Red
                    color = "blue" if category_name == "person" else "red"
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                    draw.text((x0 + 4, y0 + 4), category_name, fill=color)
                
                sanity_img_path = os.path.join(SANITY_DIR, f"sanity_check_{idx}.png")
                img.save(sanity_img_path)
                print(f"Sanity check image saved: {sanity_img_path}")
        except Exception as e:
            print(f"[WARNING] Could not plot sanity check for {img_name}: {e}")

if __name__ == '__main__':
    convert_yolo_to_coco()
