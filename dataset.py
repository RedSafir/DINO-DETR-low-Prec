import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class FootballDataset(Dataset):
    def __init__(self, annotations_path, images_dir, transforms=None):
        """
        Custom Dataset for Player and Ball detection in COCO format.
        Args:
            annotations_path (str): Path to annotations COCO JSON file.
            images_dir (str): Path to directory containing images.
            transforms (callable, optional): Transform function for images and targets.
        """
        self.images_dir = images_dir
        self.transforms = transforms
        
        # Load COCO annotations
        print(f"Loading annotations from {annotations_path}...")
        with open(annotations_path, 'r') as f:
            self.coco_data = json.load(f)
            
        # Map categories to 0-indexed labels: COCO Category ID -> 0-indexed class index
        # e.g., category 1 (person) -> 0, category 2 (ball) -> 1
        self.cat_to_label = {cat["id"]: idx for idx, cat in enumerate(self.coco_data["categories"])}
        
        # Build image dict and group annotations by image_id
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.img_ids = sorted(list(self.images.keys()))
        
        self.img_to_anns = {img_id: [] for img_id in self.img_ids}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id in self.img_to_anns:
                self.img_to_anns[img_id].append(ann)
                
        print(f"Loaded {len(self.img_ids)} images and {len(self.coco_data['annotations'])} annotations.")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        
        # Load annotations for this image
        anns = self.img_to_anns[img_id]
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO bbox format: [x_min, y_min, width, height]
            x_min, y_min, w, h = ann["bbox"]
            
            # Clamp to image boundaries to prevent boundary errors
            x_min = max(0.0, min(x_min, W))
            y_min = max(0.0, min(y_min, H))
            w = max(0.0, min(w, W - x_min))
            h = max(0.0, min(h, H - y_min))
            
            if w <= 0.0 or h <= 0.0:
                continue
                
            # Convert to [x_min, y_min, x_max, y_max] absolute coordinates
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            labels.append(self.cat_to_label[ann["category_id"]])
            areas.append(ann["area"])
            iscrowd.append(ann["iscrowd"])
            
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(H), int(W)]),
            "size": torch.as_tensor([int(H), int(W)])
        }
        
        # Apply transforms (updates both image tensor and box coordinates)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target

# ======================================================================
# Custom Augmentations for DINO-DETR (Small-Object Preserving)
# ======================================================================
class DETRTransforms:
    def __init__(self, is_train=True, min_size=800, max_size=1333, scales=None):
        self.is_train = is_train
        self.min_size = min_size
        
        # Scale max_size proportionally if using custom min_size
        if min_size != 800 and max_size == 1333:
            self.max_size = int(round(min_size * 1.666))
        else:
            self.max_size = max_size
            
        if scales is None:
            # Generate a step-based scale selection up to min_size (similar to default COCO)
            step = 32
            start = int((min_size * 0.6) // step * step)
            self.scales = list(range(start, min_size + step, step))
            if not self.scales:
                self.scales = [min_size]
        else:
            self.scales = scales
        
        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _resize(self, img, target, size):
        """
        Resizes the image and updates absolute box coordinates.
        size can be a single int (target size of the shorter edge) or (H, W).
        """
        W, H = img.size
        
        if isinstance(size, int):
            # Maintain aspect ratio: shorter edge becomes size
            min_size = float(size)
            max_size = float(self.max_size)
            factor = min_size / min(H, W)
            if max(H, W) * factor > max_size:
                factor = max_size / max(H, W)
            new_W = int(round(W * factor))
            new_H = int(round(H * factor))
        else:
            new_H, new_W = size
            
        img = img.resize((new_W, new_H), Image.BILINEAR)
        
        # Scale bounding boxes [x0, y0, x1, y1]
        if "boxes" in target and target["boxes"].shape[0] > 0:
            scale_factor = torch.tensor([new_W / W, new_H / H, new_W / W, new_H / H], dtype=torch.float32)
            target["boxes"] = target["boxes"] * scale_factor
            
        target["size"] = torch.as_tensor([int(new_H), int(new_W)])
        return img, target

    def _flip(self, img, target):
        """
        Flips the image horizontally and updates box coordinates.
        """
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        W, _ = img.size
        
        if "boxes" in target and target["boxes"].shape[0] > 0:
            boxes = target["boxes"]
            # box format: [x0, y0, x1, y1]
            # flipped: [W - x1, y0, W - x0, y1]
            flipped_boxes = boxes.clone()
            flipped_boxes[:, 0] = W - boxes[:, 2]
            flipped_boxes[:, 2] = W - boxes[:, 0]
            target["boxes"] = flipped_boxes
            
        return img, target

    def _to_tensor_and_normalize(self, img, target):
        # Convert PIL to PyTorch Tensor [C, H, W] in [0, 1] range
        import numpy as np
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        img_tensor = img_tensor.float().div(255.0)
        
        # Normalize
        mean = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Convert boxes to [cx, cy, w, h] normalized format as expected by DINO
        if "boxes" in target and target["boxes"].shape[0] > 0:
            H, W = target["size"]
            boxes = target["boxes"]
            
            # Convert [x0, y0, x1, y1] to [cx, cy, w, h]
            cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
            cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            
            # Normalize to [0, 1]
            cx_norm = cx / W
            cy_norm = cy / H
            w_norm = w / W
            h_norm = h / H
            
            target["boxes"] = torch.stack([cx_norm, cy_norm, w_norm, h_norm], dim=-1)
            
        return img_tensor, target

    def __call__(self, img, target):
        # 1. Random resize (Short-side resizing)
        if self.is_train:
            # Randomly select a target short edge size from dynamic scales
            size = random.choice(self.scales)
        else:
            size = self.min_size
            
        img, target = self._resize(img, target, size)
        
        # 2. Random horizontal flip (training only)
        if self.is_train and random.random() < 0.5:
            img, target = self._flip(img, target)
            
        # 3. Normalize and convert target boxes to DINO CX-CY-W-H format
        img_tensor, target = self._to_tensor_and_normalize(img, target)
        
        return img_tensor, target

# ======================================================================
# Collate Function for DataLoader
# ======================================================================
def collate_fn(batch):
    """
    Since images can have different dimensions after short-side resizing,
    we stack them by padding, or we group them into lists.
    DINO-DETR expects a batch of images and targets. We return tuple of lists:
    (images, targets) where images is list of tensors, targets is list of dicts.
    """
    return tuple(zip(*batch))
