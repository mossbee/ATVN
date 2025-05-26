import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class TripletLoss(nn.Module):
    """Triplet loss with online hard negative mining."""
    
    def __init__(self, margin=0.5, hard_factor=0.0):
        super().__init__()
        self.margin = margin
        self.hard_factor = hard_factor
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, embedding_dim)
            labels: (B,) - binary labels for pairs
        """
        # Compute pairwise distances
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Get positive and negative pairs
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
        negative_mask = ~positive_mask
        
        # Remove diagonal (self-similarity)
        eye_mask = torch.eye(embeddings.size(0), device=embeddings.device).bool()
        positive_mask = positive_mask & ~eye_mask
        negative_mask = negative_mask & ~eye_mask
        
        if positive_mask.sum() == 0 or negative_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Hard positive mining (furthest positive)
        positive_dist = pairwise_dist * positive_mask.float()
        positive_dist[~positive_mask] = -float('inf')
        hard_positive, _ = torch.max(positive_dist, dim=1)
        
        # Hard negative mining (closest negative)
        negative_dist = pairwise_dist * negative_mask.float()
        negative_dist[~negative_mask] = float('inf')
        hard_negative, _ = torch.min(negative_dist, dim=1)
        
        # Only consider valid triplets
        valid_triplets = (hard_positive > -float('inf')) & (hard_negative < float('inf'))
        
        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Compute triplet loss
        loss = F.relu(hard_positive - hard_negative + self.margin)
        loss = loss[valid_triplets].mean()
        
        return loss


class CenterLoss(nn.Module):
    """Center loss for feature learning."""
    
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    
    def forward(self, x, labels):
        """
        Args:
            x: (B, feat_dim)
            labels: (B,)
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: 
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for siamese networks."""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, embeddings1, embeddings2, labels):
        """
        Args:
            embeddings1: (B, embedding_dim)
            embeddings2: (B, embedding_dim) 
            labels: (B,) - 1 for similar (twins), 0 for dissimilar
        """
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(embeddings1, embeddings2)
        
        # Contrastive loss
        loss_contrastive = torch.mean(
            labels * torch.pow(euclidean_distance, 2) +
            (1 - labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B,) - predicted probabilities
            targets: (B,) - binary labels
        """
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ArcFaceLoss(nn.Module):
    """ArcFace loss for face recognition."""
    
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(self, input, label):
        # cos(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss."""
    
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: (B, embedding_dim) - normalized features
            labels: (B,) - ground truth labels
            mask: (B, B) - contrastive mask
        """
        device = features.device
        
        if len(features.shape) < 3:
            raise ValueError('features needs to be [bsz, n_views, feature_dim]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
            
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
            
        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function for twin verification."""
    
    def __init__(self, 
                 triplet_weight=1.0,
                 contrastive_weight=0.5,
                 focal_weight=0.5,
                 center_weight=0.1,
                 margin=0.5,
                 focal_alpha=1.0,
                 focal_gamma=2.0):
        super().__init__()
        
        self.triplet_weight = triplet_weight
        self.contrastive_weight = contrastive_weight
        self.focal_weight = focal_weight
        self.center_weight = center_weight
        
        self.triplet_loss = TripletLoss(margin=margin)
        self.contrastive_loss = ContrastiveLoss(margin=margin)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = nn.BCELoss()
        
    def forward(self, 
                similarity_scores, 
                embeddings1, 
                embeddings2, 
                labels,
                use_focal=False):
        """
        Args:
            similarity_scores: (B,) - predicted similarity scores
            embeddings1: (B, embedding_dim) - embeddings for first image
            embeddings2: (B, embedding_dim) - embeddings for second image
            labels: (B,) - binary labels (1 for twins, 0 for non-twins)
            use_focal: Whether to use focal loss instead of BCE
        """
        losses = {}
        
        # Classification loss
        if use_focal:
            cls_loss = self.focal_loss(similarity_scores, labels)
            losses['focal_loss'] = cls_loss
        else:
            cls_loss = self.bce_loss(similarity_scores, labels)
            losses['bce_loss'] = cls_loss
        
        # Contrastive loss
        if self.contrastive_weight > 0:
            cont_loss = self.contrastive_loss(embeddings1, embeddings2, labels)
            losses['contrastive_loss'] = cont_loss
            cls_loss += self.contrastive_weight * cont_loss
        
        # Triplet loss (if we have enough samples)
        if self.triplet_weight > 0 and len(embeddings1) >= 2:
            # Combine embeddings for triplet loss
            all_embeddings = torch.cat([embeddings1, embeddings2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            
            trip_loss = self.triplet_loss(all_embeddings, all_labels)
            losses['triplet_loss'] = trip_loss
            cls_loss += self.triplet_weight * trip_loss
        
        losses['total_loss'] = cls_loss
        
        return losses


def get_loss_function(loss_config):
    """Factory function to create loss functions."""
    loss_type = loss_config.get('type', 'combined')
    
    if loss_type == 'triplet':
        return TripletLoss(
            margin=loss_config.get('margin', 0.5),
            hard_factor=loss_config.get('hard_factor', 0.0)
        )
    elif loss_type == 'contrastive':
        return ContrastiveLoss(
            margin=loss_config.get('margin', 1.0)
        )
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('alpha', 1.0),
            gamma=loss_config.get('gamma', 2.0)
        )
    elif loss_type == 'arcface':
        return ArcFaceLoss(
            in_features=loss_config.get('in_features', 512),
            out_features=loss_config.get('out_features', 2),
            scale=loss_config.get('scale', 30.0),
            margin=loss_config.get('margin', 0.5)
        )
    elif loss_type == 'supcon':
        return SupConLoss(
            temperature=loss_config.get('temperature', 0.07)
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            triplet_weight=loss_config.get('triplet_weight', 1.0),
            contrastive_weight=loss_config.get('contrastive_weight', 0.5),
            focal_weight=loss_config.get('focal_weight', 0.5),
            center_weight=loss_config.get('center_weight', 0.1),
            margin=loss_config.get('margin', 0.5)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Import math for ArcFace
import math