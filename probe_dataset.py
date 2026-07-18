import os

DATASET_DIR = "./merged_yolo_person_ball"

def probe():
    print(f"Dataset root: {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        print("Dataset root does not exist.")
        return
        
    print("Files in root:")
    try:
        print(os.listdir(DATASET_DIR))
    except Exception as e:
        print(f"Error listing root: {e}")
        
    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")
    
    print("\nImages directory:")
    if os.path.exists(images_dir):
        try:
            contents = os.listdir(images_dir)
            print(f"Total items in images: {len(contents)}")
            print("First 10 items in images:")
            print(contents[:10])
            # Check if any are subdirs
            subdirs = [c for c in contents if os.path.isdir(os.path.join(images_dir, c))]
            print(f"Subdirectories inside images: {subdirs}")
        except Exception as e:
            print(f"Error listing images: {e}")
    else:
        print("Images directory does not exist.")
        
    print("\nLabels directory:")
    if os.path.exists(labels_dir):
        try:
            contents = os.listdir(labels_dir)
            print(f"Total items in labels: {len(contents)}")
            print("First 10 items in labels:")
            print(contents[:10])
            subdirs = [c for c in contents if os.path.isdir(os.path.join(labels_dir, c))]
            print(f"Subdirectories inside labels: {subdirs}")
        except Exception as e:
            print(f"Error listing labels: {e}")
    else:
        print("Labels directory does not exist.")

if __name__ == '__main__':
    probe()
