import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import ModuleType

# Mock MultiScaleDeformableAttention C++ extension to prevent import crashes when it is not compiled
if "MultiScaleDeformableAttention" not in sys.modules:
    mock_msda = ModuleType("MultiScaleDeformableAttention")
    # Add dummy attributes to satisfy any import checks
    mock_msda.ms_deform_attn_forward = lambda *args, **kwargs: None
    mock_msda.ms_deform_attn_backward = lambda *args, **kwargs: None
    sys.modules["MultiScaleDeformableAttention"] = mock_msda

# Add the official DINO repo to system path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "official_dino")))

# ======================================================================
# 1. Pure PyTorch Fallback for Multi-Scale Deformable Attention
# ======================================================================
def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Pure PyTorch fallback for Multi-Scale Deformable Attention using F.grid_sample.
    Func:
      value: [N, S, M, D] (flattened multi-scale feature maps)
      value_spatial_shapes: [num_levels, 2] (H and W for each scale)
      sampling_locations: [N, Lq, M, num_levels, num_points, 2] (normalized sampling grids)
      attention_weights: [N, Lq, M, num_levels, num_points] (softmax normalized weights)
    """
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    
    # Split value list along spatial size of each level
    spatial_sizes = [H_ * W_ for H_, W_ in value_spatial_shapes]
    value_list = value.split(spatial_sizes, dim=1)
    
    # Normalize sampling locations to [-1, 1] range as expected by grid_sample
    sampling_grids = 2 * sampling_locations - 1
    
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # value_l_: [N_, H_*W_, M_, D_] -> [N_, H_*W_, M_*D_] -> [N_*M_, D_, H_, W_]
        value_l_ = value_list[lid_].transpose(1, 2).flatten(0, 1).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        
        # sampling_grid_l_: [N_, Lq_, M_, P_, 2] -> [N_, M_, Lq_, P_, 2] -> [N_*M_, Lq_, P_, 2]
        sampling_grid_l_ = sampling_grids[:, :, :, lid_, :, :].transpose(1, 2).flatten(0, 1)
        
        # Sample using bilinear interpolation
        # output shape: [N_*M_, D_, Lq_, P_]
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
        
    # attention_weights shape: [N_, Lq_, M_, L_, P_] -> [N_, M_, Lq_, L_, P_] -> [N_*M_, Lq_, L_*P_] -> [N_*M_, 1, Lq_, L_*P_]
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, Lq_, L_ * P_).unsqueeze(1)
    
    # Stack sampled values over levels: [N_*M_, D_, Lq_, L_*P_]
    stacked_sampled_values = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    
    # Weighted sum
    output = (stacked_sampled_values * attention_weights).sum(-1)
    
    # Reshape back to [N, Lq, M, D]
    output = output.view(N_, M_, D_, Lq_).transpose(1, 3)
    return output

def ms_deform_attn_forward_patched(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
    N, Len_q, _ = query.shape
    N, Len_in, _ = input_flatten.shape
    assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask.unsqueeze(-1), float(0))
    value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
    
    # Calculate sampling offsets and attention weights
    sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
    attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
    attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
    
    # Map reference points
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points.unsqueeze(2).unsqueeze(-2) + \
                             sampling_offsets / offset_normalizer.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif reference_points.shape[-1] == 4:
        sampling_locations = reference_points[:, :, None, :, None, :2] + \
                             sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
    else:
        raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
        
    # Execute the pure PyTorch fallback attention
    output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
    return self.output_proj(output)

# ======================================================================
# 2. Setup Compilation & Monkey-Patching for Deformable Attention
# ======================================================================
def compile_and_setup_deform_attn() -> str:
    """
    Attempts to compile official DINO CUDA operators.
    If compilation fails, monkey-patches the DINO modules to use the pure PyTorch fallback.
    Returns: A string indicating which backend ('CUDA C++' or 'Pure PyTorch Fallback') is active.
    """
    ops_dir = os.path.join(os.path.dirname(__file__), "official_dino", "models", "dino", "ops")
    compiled_successfully = False
    
    if os.path.exists(ops_dir):
        print("[INFO] Attempting to compile DINO official CUDA operators...")
        import subprocess
        try:
            # Build extension locally
            res = subprocess.run(
                [sys.executable, "setup.py", "build", "develop"],
                cwd=ops_dir, capture_output=True, text=True, timeout=120
            )
            if res.returncode == 0:
                print("[SUCCESS] DINO CUDA operators compiled successfully.")
                compiled_successfully = True
            else:
                print(f"[WARNING] DINO CUDA compilation failed (missing C++ compiler/env). Fallback will be used.")
        except Exception as e:
            print(f"[WARNING] DINO CUDA compilation encountered an exception: {e}. Falling back to PyTorch.")
            
    # Mocking the MultiScaleDeformableAttention C++ module if not compiled
    if not compiled_successfully:
        print("[INFO] Applying pure PyTorch fallback for MultiScaleDeformableAttention (F.grid_sample).")
        try:
            # Dynamically import and monkey patch MSDeformAttn to use pure PyTorch fallback
            from models.dino.ops.modules.ms_deform_attn import MSDeformAttn
            MSDeformAttn.forward = ms_deform_attn_forward_patched
            print("[SUCCESS] Monkey-patched MSDeformAttn.forward with Pure PyTorch fallback.")
        except Exception as patch_err:
            print(f"[WARNING] Failed to monkey-patch MSDeformAttn: {patch_err}")
        return "Pure PyTorch Fallback"
        
    return "CUDA C++"

# ======================================================================
# 3. Model Builder & Head Replacer
# ======================================================================
def build_dino_model(num_classes: int, checkpoint_path: str = None, device: str = "cuda") -> nn.Module:
    """
    Builds the DINO-DETR model.
    Load Order:
      1. Instantiates DINO model with the default COCO classes (91).
      2. Loads the official pretrained COCO checkpoint weights (if checkpoint_path is provided).
      3. Replaces the classification head (class_embed) with the custom num_classes.
    """
    # 1. Setup backend operators
    backend_used = compile_and_setup_deform_attn()
    print(f"[MODEL CONFIG] Active Deformable Attention Backend: {backend_used}")

    # Import the official DINO builder dynamically
    # Under official DINO repo, we can construct the model via:
    # from models.dino.dino import build_dino
    # We will assume official DINO is cloned. Here we implement the wrapper logic.
    try:
        from models.dino.dino import build_dino
        from util.slconfig import SLConfig
        
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "official_dino", "config", "DINO", "DINO_4scale.py"))
        print(f"[MODEL CONFIG] Loading official configuration from '{config_path}'...")
        args = SLConfig.fromfile(config_path)
        
        # Override runtime values
        args.device = str(device)
        args.num_classes = 91  # COCO class count first to load pretrained weights strictly
        args.dn_labelbook_size = 91 + 1
        model, criterion, postprocessors = build_dino(args)
    except ImportError:
        print("[WARNING] Official DINO repository files not imported yet. Creating placeholder PyTorch model for testing...")
        # Placeholder model for structural testing
        class PlaceholderDINO(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Parameter(torch.randn(1))
                self.class_embed = nn.ModuleList([nn.Linear(256, 91) for _ in range(6)])
                self.dec_pred_class_embed_share = True
            def forward(self, x):
                batch_size = len(x) if isinstance(x, list) else x.size(0)
                device = x[0].device if isinstance(x, list) else x.device
                logits = torch.randn(batch_size, 900, 91, device=device) * self.dummy_param
                boxes = torch.rand(batch_size, 900, 4, device=device) * self.dummy_param
                return {
                    "pred_logits": logits,
                    "pred_boxes": boxes
                }
        model = PlaceholderDINO()
        criterion = None

    # 2. Load Pretrained Checkpoint BEFORE replacing classification head
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading pretrained COCO checkpoint weights from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load state dict (model is currently at COCO class size 91, so strict=True will succeed)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[SUCCESS] Pretrained COCO checkpoint loaded successfully with strict=True.")
        except Exception as e:
            print(f"[WARNING] Strict load failed ({e}). Loading with strict=False.")
            model.load_state_dict(state_dict, strict=False)
    else:
        if checkpoint_path:
            print(f"[WARNING] Pretrained checkpoint path '{checkpoint_path}' does not exist. Skipping weight loading.")

    # 3. Replace classification head with custom class count
    # DINO uses Sigmoid Focal Loss. Head dimension is EXACTLY num_classes (negative/background mapped implicitly).
    print(f"[INFO] Replacing classification head: changing class_embed output shape from 91 to {num_classes}")
    
    hidden_dim = model.class_embed[0].in_features
    num_layers = len(model.class_embed)
    
    new_class_embed = nn.ModuleList([
        nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
    ])
    
    # Apply weight sharing if configured
    if getattr(model, "dec_pred_class_embed_share", False):
        print("[INFO] Weight sharing is active. Sharing weights across decoder layers.")
        for i in range(num_layers):
            new_class_embed[i].weight = new_class_embed[0].weight
            new_class_embed[i].bias = new_class_embed[0].bias
            
    model.class_embed = new_class_embed
    
    # Also update the decoder reference if present
    if hasattr(model, "transformer") and hasattr(model.transformer, "decoder") and hasattr(model.transformer.decoder, "class_embed"):
        model.transformer.decoder.class_embed = new_class_embed

    # Update denoising labelbook parameters
    if hasattr(model, "dn_labelbook_size"):
        # Must be at least num_classes + 1
        model.dn_labelbook_size = num_classes + 1
        print(f"[MODEL CONFIG] Updated model.dn_labelbook_size to {model.dn_labelbook_size}")

    return model.to(device)

if __name__ == '__main__':
    # Test builder
    model = build_dino_model(num_classes=2, checkpoint_path=None, device="cpu")
    print("Model test build completed successfully.")
