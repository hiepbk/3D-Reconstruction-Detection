import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class OccupancyLoss(nn.Module):
    """Enhanced Loss for occupancy maps with multiple options.
    
    Class Imbalance Explanation:
    - Occupancy maps are (B, 256, 180, 180) where each channel is a height level
    - Within each 180×180 feature map: MOST voxels are empty (0) vs FEW occupied (>0)
    - This is the class imbalance: many negative examples (empty) vs few positive (occupied)
    - Example: In a 180×180 map, ~30,000 empty voxels vs ~2,400 occupied voxels
    
    Supports:
    - Binary Cross Entropy (BCE): Standard for probability prediction
    - Focal Loss: Handles class imbalance within each channel's spatial map
      (down-weights easy empty voxels, focuses on occupied and hard examples)
    - Dice Loss: Good for segmentation-like tasks, handles imbalanced data
    - Combined losses: BCE + Dice for better convergence
    
    Args:
        loss_type (str): Type of loss. Options: 'bce', 'focal', 'dice', 'bce_dice'.
            Defaults to 'bce'.
        reduction (str): Method to reduce losses. Options: 'none', 'sum', 'mean'.
            Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
        focal_alpha (float): Alpha parameter for focal loss (weight for positive class).
            Defaults to 0.25.
        focal_gamma (float): Gamma parameter for focal loss (focusing parameter).
            Higher gamma = more focus on hard examples. Defaults to 2.0.
        dice_weight (float): Weight for dice loss when using 'bce_dice'. Defaults to 0.5.
        pos_weight (float, optional): Weight for positive class in BCE. 
            Currently not used (removed due to tensor shape issues).
        channel_weights (list, optional): Per-channel weights for different height levels.
            If None, all channels weighted equally. Defaults to None.
    """
    
    def __init__(
        self, 
        loss_type='bce',
        reduction='mean', 
        loss_weight=1.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        dice_weight=0.5,
        pos_weight=None,
        channel_weights=None,
    ):
        super(OccupancyLoss, self).__init__()
        assert loss_type in ['bce', 'focal', 'dice', 'bce_dice']
        assert reduction in ['none', 'sum', 'mean']
        
        self.loss_type = loss_type
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.channel_weights = channel_weights
        
        if channel_weights is not None:
            self.register_buffer('channel_weights_tensor', 
                               torch.tensor(channel_weights, dtype=torch.float32))
        else:
            self.channel_weights_tensor = None
    
    def _compute_bce_loss(self, pred, target):
        """Compute Binary Cross Entropy loss.
        
        Note: Class imbalance (many empty vs few occupied voxels within each channel)
        is handled implicitly by the loss. For severe imbalance, use focal loss instead.
        """
        loss = F.binary_cross_entropy(
            pred, target, 
            reduction='none'
        )
        return loss
    
    def _compute_focal_loss(self, pred, target):
        """Compute Focal Loss for handling class imbalance.
        
        Focal loss down-weights easy examples (the many empty voxels) and focuses
        on hard examples (occupied voxels and hard empty voxels near occupied regions).
        This helps when there's severe class imbalance within each channel's spatial map.
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Compute p_t: probability of true class
        p_t = pred * target + (1 - pred) * (1 - target)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Alpha weighting: alpha for positive, (1-alpha) for negative
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        
        loss = alpha_t * focal_weight * bce
        return loss
    
    def _compute_dice_loss(self, pred, target, smooth=1e-6):
        """Compute Dice Loss (good for segmentation/imbalanced data)."""
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)  # (B, C)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # (B, C)
        
        # Dice coefficient: 2 * intersection / union
        dice = (2.0 * intersection + smooth) / (union + smooth)  # (B, C)
        
        # Dice loss: 1 - dice
        loss = 1.0 - dice  # (B, C)
        
        return loss
    
    def _apply_channel_weights(self, loss):
        """Apply per-channel weights if specified."""
        if self.channel_weights_tensor is not None:
            # loss shape: (B, C, H, W) or (B, C)
            if loss.dim() == 4:
                # (B, C, H, W) -> apply weights to C dimension
                weights = self.channel_weights_tensor.view(1, -1, 1, 1)
            elif loss.dim() == 3:
                # (B, C, H*W) -> apply weights to C dimension
                weights = self.channel_weights_tensor.view(1, -1, 1)
            elif loss.dim() == 2:
                # (B, C) -> apply weights to C dimension
                weights = self.channel_weights_tensor.view(1, -1)
            else:
                weights = 1.0
            
            loss = loss * weights
        return loss
    
    def forward(self, pred_occupancy, gt_occupancy, reduction_override=None, **kwargs):
        """Forward function of loss calculation.
        
        Args:
            pred_occupancy (torch.Tensor): Predicted occupancy map with shape [B, C, H, W].
                Values should be in [0, 1] (already sigmoid-activated).
            gt_occupancy (torch.Tensor): Ground truth occupancy map with shape [B, C, H, W].
                Values should be in [0, 1] (soft probabilities from SoftVoxelOccupancyVFE).
            reduction_override (str, optional): Method to reduce losses.
                Options: 'none', 'sum', 'mean'. Defaults to None.
        
        Returns:
            torch.Tensor: Occupancy loss value.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        # Clamp predictions to avoid numerical issues
        pred_occupancy = pred_occupancy.clamp(min=1e-6, max=1.0 - 1e-6)
        
        if self.loss_type == 'bce':
            loss = self._compute_bce_loss(pred_occupancy, gt_occupancy)
            
        elif self.loss_type == 'focal':
            loss = self._compute_focal_loss(pred_occupancy, gt_occupancy)
            
        elif self.loss_type == 'dice':
            loss = self._compute_dice_loss(pred_occupancy, gt_occupancy)
            # Dice loss returns (B, C), expand to (B, C, H, W) for consistency
            loss = loss.unsqueeze(-1).unsqueeze(-1).expand_as(pred_occupancy)
            
        elif self.loss_type == 'bce_dice':
            bce_loss = self._compute_bce_loss(pred_occupancy, gt_occupancy)
            dice_loss = self._compute_dice_loss(pred_occupancy, gt_occupancy)
            # Expand dice_loss to match bce_loss shape
            dice_loss = dice_loss.unsqueeze(-1).unsqueeze(-1).expand_as(pred_occupancy)
            loss = bce_loss + self.dice_weight * dice_loss
        
        # Apply channel weights if specified
        loss = self._apply_channel_weights(loss)
        
        # Apply reduction
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()
        # 'none' case: return as-is
        
        return loss * self.loss_weight

