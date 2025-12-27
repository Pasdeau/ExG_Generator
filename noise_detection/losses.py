"""
Focal Loss for imbalanced binary segmentation.
Helps the model focus on hard-to-classify samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification/segmentation.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (after sigmoid), shape (B, 1, L)
            target: Ground truth labels (0 or 1), shape (B, 1, L)
        """
        # Clamp predictions to avoid log(0)
        pred = pred.clamp(min=1e-7, max=1 - 1e-7)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Probability of correct class
        pt = torch.where(target == 1, pred, 1 - pred)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting (higher weight for positive class)
        alpha_weight = torch.where(target == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce
        return focal_loss.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + BCE Loss for better boundary detection.
    """
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE
        bce = F.binary_cross_entropy(pred, target)
        
        # Dice
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        dice_loss = 1 - dice
        
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss
