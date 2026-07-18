import os
import sys
import json
import json.encoder
json.encoder.c_make_encoder = None
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import FootballDataset, DETRTransforms, collate_fn
from model import build_dino_model

def get_vram_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) # MB
        allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 2)
        cached_mem = torch.cuda.memory_reserved(0) / (1024 ** 2)
        return f"[VRAM Status] GPU: {gpu_name} | Total: {total_mem:.1f}MB | Allocated: {allocated_mem:.1f}MB | Cached: {cached_mem:.1f}MB"
    return "[VRAM Status] Running on CPU (No CUDA VRAM info)"

def evaluate(model, val_loader, device, coco_gt_path):
    model.eval()
    coco_gt = COCO(coco_gt_path)
    results = []
    
    print("Running evaluation on validation set...")
    start_time = time.time()
    
    with torch.no_grad():
        for images, targets in val_loader:
            # Prepare inputs
            # DINO expects images as nested tensors or list of tensors
            images = [img.to(device) for img in images]
            
            # Forward pass
            # Wrap in autocast disable just to guarantee FP32 murninya
            with torch.amp.autocast(device_type=device.type, enabled=False):
                outputs = model(images)
                
            # outputs: dict containing 'pred_logits' and 'pred_boxes'
            # 'pred_logits': [batch_size, num_queries, num_classes]
            # 'pred_boxes': [batch_size, num_queries, 4] in [cx, cy, w, h] normalized format
            
            # Convert outputs to COCO predictions
            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]
            
            # Calculate scores and labels using sigmoid (as standard in Focal Loss)
            prob = pred_logits.sigmoid()
            topk_values, topk_indexes = prob.view(pred_logits.shape[0], -1).topk(100, dim=1)
            
            scores = topk_values
            topk_boxes = topk_indexes // pred_logits.shape[2]
            labels = topk_indexes % pred_logits.shape[2]
            
            for i, target in enumerate(targets):
                img_id = int(target["image_id"].item())
                orig_size = target["orig_size"] # [H, W]
                H, W = orig_size[0].item(), orig_size[1].item()
                
                # Fetch prediction details for this image
                img_scores = scores[i].cpu()
                img_boxes_idx = topk_boxes[i]
                img_labels = labels[i].cpu()
                
                # Retrieve normalized boxes and scale back to absolute pixel coordinates
                img_pred_boxes = pred_boxes[i][img_boxes_idx].cpu()
                
                # Convert normalized [cx, cy, w, h] to COCO absolute [x_min, y_min, w, h]
                cx, cy, w, h = img_pred_boxes.unbind(-1)
                x_min = (cx - 0.5 * w) * W
                y_min = (cy - 0.5 * h) * H
                w_px = w * W
                h_px = h * H
                
                for box_idx in range(len(img_scores)):
                    score = float(img_scores[box_idx].item())
                    if score < 0.05: # Threshold to filter low confidences
                        continue
                    
                    label_idx = int(img_labels[box_idx].item())
                    # Map 0-indexed class back to COCO category id (class 0 -> id 1, class 1 -> id 2)
                    category_id = label_idx + 1
                    
                    box = [
                        round(float(x_min[box_idx].item()), 2),
                        round(float(y_min[box_idx].item()), 2),
                        round(float(w_px[box_idx].item()), 2),
                        round(float(h_px[box_idx].item()), 2)
                    ]
                    
                    results.append({
                        "image_id": img_id,
                        "category_id": category_id,
                        "bbox": box,
                        "score": score
                    })

    print(f"Inference completed in {time.time() - start_time:.2f} seconds. Total predictions: {len(results)}")
    
    if len(results) == 0:
        print("[WARNING] No predictions found with confidence >= 0.05. Skipping COCO evaluation.")
        return 0.0, 0.0
        
    # Write temp predictions file
    results_path = "temp_results.json"
    with open(results_path, 'w') as f:
        f.write(json.dumps(results))
        
    # Run COCO Eval
    try:
        coco_dt = coco_gt.loadRes(results_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract AP size breakdowns and class APs
        mAP_50_95 = coco_eval.stats[0] # AP @ IoU=0.50:0.95
        mAP_50 = coco_eval.stats[1]    # AP @ IoU=0.50
        
        print("\n==========================================================")
        print("          Evaluation Metrics Breakdown                    ")
        print("==========================================================")
        print(f"Overall mAP @ 0.5:0.95  : {mAP_50_95:.4f}")
        print(f"Overall mAP @ 0.5       : {mAP_50:.4f}")
        print(f"AP (Small objects)      : {coco_eval.stats[3]:.4f}")
        print(f"AP (Medium objects)     : {coco_eval.stats[4]:.4f}")
        print(f"AP (Large objects)      : {coco_eval.stats[5]:.4f}")
        
        # Compute category-wise AP
        for cat_id in sorted(coco_gt.getCatIds()):
            cat_name = coco_gt.loadCats(cat_id)[0]["name"]
            coco_eval_cat = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval_cat.params.catIds = [cat_id]
            coco_eval_cat.evaluate()
            coco_eval_cat.accumulate()
            coco_eval_cat.summarize()
            print(f"AP ({cat_name})            : {coco_eval_cat.stats[0]:.4f} (mAP@0.5:0.95) | {coco_eval_cat.stats[1]:.4f} (mAP@0.5)")
        print("==========================================================\n")
        
        os.remove(results_path)
        return mAP_50, mAP_50_95
    except Exception as e:
        print(f"[ERROR] Failed to run COCO evaluation: {e}")
        if os.path.exists(results_path):
            os.remove(results_path)
        return 0.0, 0.0

def run_overfit_check(model, device):
    """
    Overfitting sanity check: Trains the model on a tiny subset of 10 images
    for 50 epochs and verifies if the loss drops significantly.
    """
    print("\n==========================================================")
    print("       Running Overfitting Sanity Check (10 Images)       ")
    print("==========================================================")
    
    # Load dataset with only 10 images
    train_dataset = FootballDataset("annotations_train.json", "./merged_yolo_person_ball/images/train", transforms=DETRTransforms(is_train=True))
    # Slice to 10 images
    train_dataset.img_ids = train_dataset.img_ids[:10]
    
    loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    
    # Try importing official criterion or use custom loss mock
    try:
        from models.dino.dino import build_dino
        # Get criterion from builder
        class Args:
            def __init__(self):
                self.backbone = 'resnet50'
                self.dilation = False
                self.position_embedding = 'sine'
                self.pe_temperatureH = 20
                self.pe_temperatureW = 20
                self.masks = False
                self.num_feature_levels = 4
                self.enc_layers = 6
                self.dec_layers = 6
                self.dim_feedforward = 2048
                self.hidden_dim = 256
                self.dropout = 0.0
                self.nheads = 8
                self.num_queries = 900
                self.dec_n_points = 4
                self.enc_n_points = 4
                self.two_stage = True
                self.num_patterns = 0
                self.dn_number = 100
                self.dn_box_noise_scale = 0.4
                self.dn_label_noise_scale = 0.5
                self.dn_labelbook_size = 3
                self.dec_pred_class_embed_share = True
                self.num_classes = 2
        _, criterion, _ = build_dino(Args())
        criterion.to(device)
    except Exception:
        # Simple mock criterion for structural pipeline execution
        print("[WARNING] Official DINO Criterion not found. Using custom mock loss for sanity check.")
        class MockCriterion(torch.nn.Module):
            def forward(self, outputs, targets):
                cls_loss = outputs["pred_logits"].mean() * 0.0
                box_loss = torch.tensor(0.0, device=outputs["pred_boxes"].device)
                
                pred_boxes = outputs["pred_boxes"]
                for i, target in enumerate(targets):
                    tgt_boxes = target["boxes"]
                    num_boxes = tgt_boxes.shape[0]
                    if num_boxes > 0:
                        p_boxes = pred_boxes[i, :num_boxes]
                        box_loss = box_loss + F.l1_loss(p_boxes, tgt_boxes)
                return {"loss_ce": cls_loss, "loss_bbox": box_loss, "loss_giou": box_loss * 0.5}
        criterion = MockCriterion()
        
    start_loss = None
    final_loss = None
    
    for epoch in range(1, 51):
        epoch_loss = 0.0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            # Convert targets dict coordinates to device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss_dict = criterion(outputs, targets)
            weight_dict = getattr(criterion, "weight_dict", {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0})
            
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
        avg_loss = epoch_loss / len(loader)
        if start_loss is None:
            start_loss = avg_loss
        final_loss = avg_loss
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/50 - Training Loss: {avg_loss:.4f} | {get_vram_info()}")
            
    print(f"\nInitial Loss: {start_loss:.4f} -> Final Loss: {final_loss:.4f}")
    if final_loss < start_loss * 0.5:
        print("[SUCCESS] Overfitting sanity check passed successfully! Loss decreased significantly.")
    else:
        print("[WARNING] Loss did not drop as expected. Check learning rate or gradient updates.")
    print("==========================================================\n")

def train_baseline(args):
    device = torch.device(args.device)
    print(f"Target Execution Device: {device}")
    print(get_vram_info())
    
    # 1. Initialize Model
    model = build_dino_model(num_classes=2, checkpoint_path=args.checkpoint, device=device)
    
    # 2. Run Overfit Check if requested
    if args.overfit_check:
        run_overfit_check(model, device)
        return
        
    # 3. Load full Datasets
    print("Setting up full dataset dataloaders...")
    train_dataset = FootballDataset("annotations_train.json", "./merged_yolo_person_ball/images/train", transforms=DETRTransforms(is_train=True))
    val_dataset = FootballDataset("annotations_val.json", "./merged_yolo_person_ball/images/val", transforms=DETRTransforms(is_train=False))
    
    batch_size = args.batch_size
    
    # Safe Dataloader builder
    def build_loaders(bs):
        t_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2, collate_fn=collate_fn, drop_last=True)
        v_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=2, collate_fn=collate_fn)
        return t_loader, v_loader
        
    train_loader, val_loader = build_loaders(batch_size)
    
    # Setup optimizer and lr scheduler (standard AdamW + Warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define criterion
    try:
        from models.dino.dino import build_dino
        # Get criterion from builder
        class DINOArgs:
            def __init__(self):
                self.backbone = 'resnet50'
                self.dilation = False
                self.position_embedding = 'sine'
                self.pe_temperatureH = 20
                self.pe_temperatureW = 20
                self.masks = False
                self.num_feature_levels = 4
                self.enc_layers = 6
                self.dec_layers = 6
                self.dim_feedforward = 2048
                self.hidden_dim = 256
                self.dropout = 0.0
                self.nheads = 8
                self.num_queries = 900
                self.dec_n_points = 4
                self.enc_n_points = 4
                self.two_stage = True
                self.num_patterns = 0
                self.dn_number = 100
                self.dn_box_noise_scale = 0.4
                self.dn_label_noise_scale = 0.5
                self.dn_labelbook_size = 3
                self.dec_pred_class_embed_share = True
                self.num_classes = 2
        _, criterion, _ = build_dino(DINOArgs())
        criterion.to(device)
    except Exception:
        print("[WARNING] Official DINO Criterion not found. Using fallback mock loss.")
        class MockCriterion(torch.nn.Module):
            def forward(self, outputs, targets):
                cls_loss = outputs["pred_logits"].mean() * 0.0
                box_loss = torch.tensor(0.0, device=outputs["pred_boxes"].device)
                pred_boxes = outputs["pred_boxes"]
                for i, target in enumerate(targets):
                    tgt_boxes = target["boxes"]
                    num_boxes = tgt_boxes.shape[0]
                    if num_boxes > 0:
                        p_boxes = pred_boxes[i, :num_boxes]
                        box_loss = box_loss + F.l1_loss(p_boxes, tgt_boxes)
                return {"loss_ce": cls_loss, "loss_bbox": box_loss, "loss_giou": box_loss * 0.5}
        criterion = MockCriterion()

    best_map = 0.0
    print("Starting Training Loop (FP32 baseline)...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # Batch loop with OOM Fallback recovery
        loader_iter = iter(train_loader)
        batch_idx = 0
        
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
                
            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            try:
                # Force FP32 execution (no mixed precision context)
                with torch.amp.autocast(device_type=device.type, enabled=False):
                    outputs = model(images)
                    loss_dict = criterion(outputs, targets)
                    weight_dict = getattr(criterion, "weight_dict", {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0})
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                batch_idx += 1
                
                if batch_idx % args.log_interval == 0:
                    print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] - Loss: {losses.item():.4f} | {get_vram_info()}")
                    
            except RuntimeError as e:
                # Catch Out-Of-Memory (OOM) error
                if "out of memory" in str(e).lower():
                    print("\n[OOM ALERT] Out of memory detected during forward/backward pass!")
                    print(get_vram_info())
                    torch.cuda.empty_cache()
                    
                    # Fallback strategy: reduce batch size
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        print(f"[RECOVERY] Reducing batch size to {batch_size} and rebuilding loaders...")
                        train_loader, val_loader = build_loaders(batch_size)
                        # Empty cache and restart current epoch
                        torch.cuda.empty_cache()
                        print("[RECOVERY] Restarting epoch training with smaller batch size...")
                        break # Exits batch loop to restart epoch
                    else:
                        print("[FATAL OOM] Batch size is already 1. Cannot reduce further. Exiting.")
                        raise e
                else:
                    raise e
                    
        # Calculate validation metrics
        avg_loss = epoch_loss / max(1, batch_idx)
        print(f"\n--- Epoch {epoch} Complete | Average Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.2f}s ---")
        
        # Validation Eval
        val_map_50, val_map_50_95 = evaluate(model, val_loader, device, "annotations_val.json")
        
        # Checkpointing
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_map_50': val_map_50,
            'val_map_50_95': val_map_50_95,
        }
        
        # Save latest epoch checkpoint
        torch.save(checkpoint, "checkpoint_latest.pth")
        
        # Save best model
        if val_map_50 > best_map:
            best_map = val_map_50
            torch.save(checkpoint, "checkpoint_best.pth")
            print(f"[CHECKPOINT] Saved new best model with mAP@0.5: {best_map:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DINO-DETR FP32 Baseline Object Detection Training")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Execution device")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained model weights checkpoint path")
    parser.add_argument("--overfit-check", action="store_true", help="Run overfitting check on 10 images")
    parser.add_argument("--log-interval", type=int, default=50, help="Interval for printing training loss logs")
    
    args = parser.parse_args()
    train_baseline(args)
