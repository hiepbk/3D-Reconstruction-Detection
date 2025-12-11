"""
Utility script to check VRAM usage of model components after initialization.
Run this after model is built to see memory breakdown.
"""
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import get_dist_info


def check_model_memory(cfg_path, device='cuda:0'):
    """Check VRAM usage of model components.
    
    Args:
        cfg_path: Path to config file
        device: Device to use
    """
    # Clear cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated(device) / 1024**2
        print(f"Initial VRAM: {initial_memory:.2f} MB")
    
    # Load config
    cfg = Config.fromfile(cfg_path)
    
    # Build model
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)
    
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model = model.to(device)
    model.eval()
    
    # Synchronize and measure
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        
        print("\n" + "="*60)
        print("Model Memory Usage:")
        print("="*60)
        print(f"Total Allocated: {allocated:.2f} MB")
        print(f"Total Reserved:  {reserved:.2f} MB")
        print(f"Peak Allocated:  {peak_allocated:.2f} MB")
        print(f"Peak Reserved:   {peak_reserved:.2f} MB")
        print(f"Net Usage:       {allocated - initial_memory:.2f} MB (after initial)")
        
        # Get component-wise breakdown
        print("\n" + "="*60)
        print("Component Breakdown:")
        print("="*60)
        
        if hasattr(model, 'reconstruction_backbone'):
            rb = model.reconstruction_backbone
            if hasattr(rb, 'da3_model'):
                # DA3 model
                da3_params = sum(p.numel() for p in rb.da3_model.parameters())
                da3_buffers = sum(b.numel() for b in rb.da3_model.buffers())
                da3_total = da3_params + da3_buffers
                da3_memory_estimate = (da3_total * 4) / 1024**2  # 4 bytes per float32
                print(f"DA3 Model:")
                print(f"  Parameters: {da3_params/1e6:.2f}M")
                print(f"  Buffers:    {da3_buffers/1e6:.2f}M")
                print(f"  Total:      {da3_total/1e6:.2f}M")
                print(f"  VRAM (est): {da3_memory_estimate:.2f} MB")
            
            if hasattr(rb, 'refinement'):
                if rb.refinement is not None:
                    # Refinement network
                    refinement_params = sum(p.numel() for p in rb.refinement.parameters())
                    refinement_buffers = sum(b.numel() for b in rb.refinement.buffers())
                    refinement_total = refinement_params + refinement_buffers
                    refinement_memory_estimate = (refinement_total * 4) / 1024**2
                    print(f"\nRefinement Network:")
                    print(f"  Parameters: {refinement_params/1e6:.2f}M")
                    print(f"  Buffers:    {refinement_buffers/1e6:.2f}M")
                    print(f"  Total:      {refinement_total/1e6:.2f}M")
                    print(f"  VRAM (est): {refinement_memory_estimate:.2f} MB")
                    
                    # Breakdown refinement components
                    if hasattr(rb.refinement, 'refinement_net'):
                        rn_params = sum(p.numel() for p in rb.refinement.refinement_net.parameters())
                        print(f"    - Refinement Net: {rn_params/1e6:.2f}M params")
                    
                    # Loss modules are small, just count them
                    loss_modules = [rb.refinement.loss_chamfer, rb.refinement.loss_emd, 
                                  rb.refinement.loss_smoothness, rb.refinement.loss_color]
                    loss_count = sum(1 for m in loss_modules if m is not None)
                    print(f"    - Loss modules: {loss_count} active")
        
        # Print detailed memory summary
        print("\n" + "="*60)
        print("Detailed CUDA Memory Summary:")
        print("="*60)
        print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check model VRAM usage')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    args = parser.parse_args()
    
    check_model_memory(args.config, args.device)

