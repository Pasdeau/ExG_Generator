import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(pt) = -alpha_t * (1-pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # dict or tensor of class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        inputs: (B, C) logits
        targets: (B,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, dict):
                # Convert dict to tensor
                alpha_tensor = torch.ones(inputs.size(1))
                for k, v in self.alpha.items():
                    alpha_tensor[k] = v
                alpha_tensor = alpha_tensor.to(inputs.device)
            else:
                alpha_tensor = self.alpha.to(inputs.device)
            
            alpha_t = alpha_tensor.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """Combined loss for gesture + subject classification"""
    def __init__(self, gesture_weight=1.0, subject_weight=0.3, 
                 class_weights=None, gamma=2.0):
        super().__init__()
        self.gesture_weight = gesture_weight
        self.subject_weight = subject_weight
        self.gesture_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        self.subject_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, gesture_targets, subject_targets):
        """
        outputs: dict with 'gesture' and 'subject' logits
        """
        loss_gesture = self.gesture_loss(outputs['gesture'], gesture_targets)
        loss_subject = self.subject_loss(outputs['subject'], subject_targets)
        
        total_loss = self.gesture_weight * loss_gesture + self.subject_weight * loss_subject
        
        return total_loss, loss_gesture, loss_subject
