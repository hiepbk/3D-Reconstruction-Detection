"""
Custom hook to log component-wise VRAM usage during training.
"""
import torch
from mmcv.runner import HOOKS, TextLoggerHook


@HOOKS.register_module()
class ComponentMemoryLoggerHook(TextLoggerHook):
    """Hook to log component-wise VRAM usage during training.
    
    Extends TextLoggerHook to add component memory breakdown to logs.
    """
    
    def __init__(self, **kwargs):
        """Initialize with same args as TextLoggerHook."""
        super().__init__(**kwargs)
    
    def _get_component_memory(self, model):
        """Get memory breakdown for model components.
        
        Returns:
            dict: Memory breakdown with keys 'da3_mb', 'refinement_mb', 'total_mb'
        """
        if not torch.cuda.is_available():
            return {'da3_mb': 0, 'refinement_mb': 0, 'total_mb': 0}
        
        torch.cuda.synchronize()
        total_mb = torch.cuda.memory_allocated() / 1024**2
        
        da3_mb = 0
        refinement_mb = 0
        
        # Handle MMDataParallel wrapper - get the actual model
        actual_model = model.module if hasattr(model, 'module') else model
        
        # Try to get component memory
        if hasattr(actual_model, 'reconstruction_backbone'):
            rb = actual_model.reconstruction_backbone
            if hasattr(rb, 'da3_model'):
                # Estimate DA3 memory from parameters
                da3_params = sum(p.numel() for p in rb.da3_model.parameters())
                da3_buffers = sum(b.numel() for b in rb.da3_model.buffers())
                da3_total = da3_params + da3_buffers
                da3_mb = (da3_total * 4) / 1024**2  # 4 bytes per float32
            
            if hasattr(rb, 'refinement') and rb.refinement is not None:
                # Estimate Refinement memory from parameters
                refinement_params = sum(p.numel() for p in rb.refinement.parameters())
                refinement_buffers = sum(b.numel() for b in rb.refinement.buffers())
                refinement_total = refinement_params + refinement_buffers
                refinement_mb = (refinement_total * 4) / 1024**2
        
        return {
            'da3_mb': da3_mb,
            'refinement_mb': refinement_mb,
            'total_mb': total_mb
        }
    
    def before_train_iter(self, runner):
        """Reset peak memory tracking before each iteration."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def after_train_iter(self, runner):
        """Add component memory directly to log output.
        
        This is called after each training iteration. We add component memory
        as separate fields to the log buffer, which will be automatically
        formatted by the parent TextLoggerHook.
        """
        # Get component memory
        memory_info = self._get_component_memory(runner.model)
        
        # Get current memory and peak memory for this iteration
        # (peak was reset in before_train_iter)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory_mb = torch.cuda.memory_allocated() / 1024**2
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            current_memory_mb = 0
            peak_memory_mb = 0
        
        # Add component memory directly to log buffer output
        # These will appear as separate fields in the log
        if hasattr(runner, 'log_buffer') and hasattr(runner.log_buffer, 'output'):
            runner.log_buffer.output['memory_current'] = current_memory_mb
            runner.log_buffer.output['memory_peak'] = peak_memory_mb
            runner.log_buffer.output['memory_da3'] = memory_info['da3_mb']
            runner.log_buffer.output['memory_refinement'] = memory_info['refinement_mb']
        
        # Call parent to handle the actual logging
        super().after_train_iter(runner)

