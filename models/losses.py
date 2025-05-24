import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for twin verification"""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, output, target):
        # output: similarity scores, target: 1 for twins, 0 for non-twins
        distances = 1 - torch.sigmoid(output)  # Convert to distance
        
        positive_loss = target * torch.pow(distances, 2)
        negative_loss = (1 - target) * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, output, target):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(output, target.float(), reduction='none')
        p_t = probs * target + (1 - probs) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        focal_loss = alpha_t * torch.pow(1 - p_t, self.gamma) * ce_loss
        
        return torch.mean(focal_loss)


class CombinedLoss(nn.Module):
    """Combined loss function for twin verification"""
    
    def __init__(self, contrastive_margin=1.0, focal_alpha=0.25, focal_gamma=2.0,
                 lambda_contrastive=1.0, lambda_focal=0.5):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.lambda_contrastive = lambda_contrastive
        self.lambda_focal = lambda_focal
        
    def forward(self, output, target):
        contrastive = self.contrastive_loss(output, target)
        focal = self.focal_loss(output, target)
        
        total_loss = (self.lambda_contrastive * contrastive + 
                     self.lambda_focal * focal)
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive,
            'focal_loss': focal
        }
