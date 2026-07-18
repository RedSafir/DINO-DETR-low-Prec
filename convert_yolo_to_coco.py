import os
import json
import yaml
from PIL import Image, ImageDraw

# Force pure Python JSON encoder to prevent low-level C-extension crashes (SystemError) on Windows/debuggers
import json.encoder
json.encoder.c_make_encoder = None

# Path configuration
DATASET_DIR = "./merged_yolo_person_ball"
OUTPUT_DIR = "."
SANITY_DIR = "./sanity_checks"

def convert_yolo_to_coco():
    print("==========================================================")
    print("         YOLO -> COCO Conversion (Split-Aware)            ")
    print("==========================================================")
    
    # 1. Read class configuration
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        names = data_config.get("names", ["person", "ball"])
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
    expected_stats = None
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            expected_stats = json.load(f)
        print(f"Expected stats: Train={expected_stats.get('train', {}).get('images', 0)} images, Val={expected_stats.get('val', {}).get('images', 0)} images.")
    else:
        print("[INFO] merge_stats.json not found.")

    # 3. Helper function to process a specific split
    def process_split(split_name):
        images_dir = os.path.join(DATASET_DIR, "images", split_name)
        labels_dir = os.path.join(DATASET_DIR, "labels", split_name)

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            print(f"[ERROR] Subdirectories for split '{split_name}' do not exist.")
            return None

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        valid_pairs = []
        
        for img_name in os.listdir(images_dir):
            if img_name.lower().endswith(image_extensions):
                base_name = os.path.splitext(img_name)[0]
                label_name = base_name + ".txt"
                label_path = os.path.join(labels_dir, label_name)
                if os.path.exists(label_path):
                    valid_pairs.append((img_name, label_name))

        # Sort for determinism
        valid_pairs.sort(key=lambda x: x[0])
        print(f"Found {len(valid_pairs)} image-label pairs for split '{split_name}'.")

        coco_output = {
            "info": {"description": f"Football Person and Ball dataset - {split_name} split"},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }

        ann_id_counter = 1
        for img_id, (img_name, label_name) in enumerate(valid_pairs, 1):
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

            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w_norm, h_norm = map(float, parts)
                class_id = int(class_id)

                if class_id not in class_map:
                    continue

                category_id = class_map[class_id]

                # Convert normalized coordinates to absolute pixel coordinates
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

        # Validation against merge_stats.json
        if expected_stats:
            expected_count = expected_stats.get(split_name, {}).get("images", 0)
            if len(coco_output["images"]) != expected_count:
                print(f"[WARNING] Mismatch in '{split_name}' count! Found {len(coco_output['images'])} in images folder, but expected {expected_count} from merge_stats.json.")
            else:
                print(f"[OK] '{split_name}' image count matches merge_stats.json exactly: {expected_count}")

        return coco_output, valid_pairs

    # 4. Run conversion
    train_coco, train_pairs = process_split("train")
    val_coco, val_pairs = process_split("val")

    if train_coco is None or val_coco is None:
        print("[ERROR] Conversion failed due to missing split folders.")
        return

    # Write output JSON files
    print("Writing annotations_train.json...")
    with open(os.path.join(OUTPUT_DIR, "annotations_train.json"), 'w') as f:
        f.write(json.dumps(train_coco))
    print("Writing annotations_val.json...")
    with open(os.path.join(OUTPUT_DIR, "annotations_val.json"), 'w') as f:
        f.write(json.dumps(val_coco))

    print("[SUCCESS] annotations_train.json and annotations_val.json generated successfully!")

    # 5. Sanity-Check Plotting (Train Set)
    os.makedirs(SANITY_DIR, exist_ok=True)
    print(f"Generating 5 sanity check images in '{SANITY_DIR}'...")
    
    # Pick first 5 items from sorted list
    sanity_sample = train_pairs[:5]
    cat_names = {cat["id"]: cat["name"] for cat in categories}
    train_images_dir = os.path.join(DATASET_DIR, "images", "train")
    train_labels_dir = os.path.join(DATASET_DIR, "labels", "train")

    for idx, (img_name, label_name) in enumerate(sanity_sample, 1):
        img_path = os.path.join(train_images_dir, img_name)
        label_path = os.path.join(train_labels_dir, label_name)
        
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
