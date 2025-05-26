import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math

from .backbone import BackboneFactory, FeatureExtractor
from .attention import TwinAttentionModule, AdaptiveFeatureFusion


class ATVNModel(nn.Module):
    """
    Attention-Enhanced Twin Verification Network (ATVN).
    
    This model combines pre-trained face recognition backbones with attention mechanisms
    to distinguish between identical twins by focusing on fine-grained differences.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.embedding_dim = config['model']['embedding_dim']
        self.attention_dim = config['model']['attention_dim']
        
        # Create backbone
        self.backbone = BackboneFactory.create_backbone(config)
        
        # Feature extractor with multi-scale support
        self.feature_extractor = FeatureExtractor(self.backbone)
        
        # Twin attention module
        self.twin_attention = TwinAttentionModule(
            in_channels=self._get_feature_channels(),
            attention_dim=self.attention_dim,
            num_heads=config['model']['num_attention_heads'],
            dropout=config['model']['dropout']
        )
        
        # Global average pooling for attention features
        self.attention_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature fusion module
        self.feature_fusion = AdaptiveFeatureFusion(
            global_dim=self.embedding_dim,
            attention_dim=self.attention_dim,
            output_dim=self.embedding_dim
        )
        
        # Similarity computation
        self.similarity_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(self.embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # L2 normalization for embeddings
        self.l2_norm = nn.functional.normalize
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_feature_channels(self) -> int:
        """Get the number of channels in feature maps from backbone."""
        # This is a simple heuristic - in practice, you might want to
        # run a forward pass to determine this automatically
        backbone_type = self.config['model']['backbone'].lower()
        if 'facenet' in backbone_type:
            return 1792  # InceptionResnetV1
        elif 'resnet' in backbone_type:
            return 2048  # ResNet50/101
        elif 'efficientnet' in backbone_type:
            return 1280  # EfficientNet-B0
        else:
            return 512  # Default
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract global and local features from input image.
        
        Args:
            x: Input image tensor (B, 3, H, W)
        Returns:
            global_embeddings: Global face embeddings (B, embedding_dim)
            feature_maps: Feature maps for attention (B, C, H', W')
        """
        # Extract features using backbone
        global_embeddings, feature_maps, _ = self.feature_extractor(x)
        
        # L2 normalize global embeddings
        global_embeddings = self.l2_norm(global_embeddings, p=2, dim=1)
        
        return global_embeddings, feature_maps
    
    def compute_attention_features(
        self, 
        feature_maps1: torch.Tensor, 
        feature_maps2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute attention-enhanced features for twin comparison.
        
        Args:
            feature_maps1: Feature maps from first image (B, C, H, W)
            feature_maps2: Feature maps from second image (B, C, H, W)
        Returns:
            attention_features: Attention-enhanced features (B, attention_dim)
            attention_info: Dictionary containing attention maps
        """
        # Apply twin attention module
        (enhanced_f1, enhanced_f2), attention_info = self.twin_attention(
            feature_maps1, feature_maps2
        )
        
        # Global pooling of attention features
        attn_f1 = self.attention_pool(enhanced_f1).squeeze(-1).squeeze(-1)
        attn_f2 = self.attention_pool(enhanced_f2).squeeze(-1).squeeze(-1)
        
        # Combine attention features (element-wise absolute difference)
        attention_features = torch.abs(attn_f1 - attn_f2)
        
        return attention_features, attention_info
    
    def forward(
        self, 
        image1: torch.Tensor, 
        image2: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for twin verification.
        
        Args:
            image1: First image (B, 3, H, W)
            image2: Second image (B, 3, H, W)
            return_attention: Whether to return attention maps
        Returns:
            Dictionary containing:
                - similarity: Similarity score (B, 1)
                - embeddings1: Embeddings for image1 (B, embedding_dim)
                - embeddings2: Embeddings for image2 (B, embedding_dim)
                - fused_features: Fused global and attention features (B, embedding_dim)
                - attention_info: Attention maps (if return_attention=True)
        """
        # Extract features for both images
        global_emb1, feature_maps1 = self.extract_features(image1)
        global_emb2, feature_maps2 = self.extract_features(image2)
        
        # Compute attention features
        attention_features, attention_info = self.compute_attention_features(
            feature_maps1, feature_maps2
        )
        
        # Combine global embeddings (element-wise absolute difference)
        global_diff = torch.abs(global_emb1 - global_emb2)
        
        # Fuse global and attention features
        fused_features = self.feature_fusion(global_diff, attention_features)
        
        # Compute similarity score
        similarity = self.similarity_head(fused_features)
        
        # Prepare output
        output = {
            'similarity': similarity,
            'embeddings1': global_emb1,
            'embeddings2': global_emb2,
            'fused_features': fused_features,
            'global_diff': global_diff,
            'attention_features': attention_features
        }
        
        if return_attention:
            output['attention_info'] = attention_info
        
        return output
    
    def compute_similarity(
        self, 
        image1: torch.Tensor, 
        image2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity score between two images.
        
        Args:
            image1: First image (B, 3, H, W)
            image2: Second image (B, 3, H, W)
        Returns:
            similarity: Similarity score (B, 1)
        """
        with torch.no_grad():
            output = self.forward(image1, image2, return_attention=False)
            return output['similarity']
    
    def get_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        """
        Get face embeddings for a single image.
        
        Args:
            image: Input image (B, 3, H, W)
        Returns:
            embeddings: Face embeddings (B, embedding_dim)
        """
        with torch.no_grad():
            embeddings, _ = self.extract_features(image)
            return embeddings
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def get_attention_maps(
        self, 
        image1: torch.Tensor, 
        image2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            image1: First image (1, 3, H, W)
            image2: Second image (1, 3, H, W)
        Returns:
            Dictionary of attention maps
        """
        with torch.no_grad():
            output = self.forward(image1, image2, return_attention=True)
            return output['attention_info']
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_from_checkpoint(cls, filepath: str, map_location: str = 'cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Create model
        model = cls(checkpoint['config'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint.get('epoch', 0)


class ATVNLoss(nn.Module):
    """Combined loss function for ATVN training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Loss weights
        self.triplet_weight = config['loss']['triplet_weight']
        self.center_weight = config['loss']['center_weight']
        self.attention_reg_weight = config['loss']['attention_reg_weight']
        
        # Individual losses
        self.bce_loss = nn.BCELoss()
        self.triplet_loss = nn.TripletMarginLoss(
            margin=config['loss']['triplet_margin']
        )
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        labels: torch.Tensor,
        embeddings1: torch.Tensor = None,
        embeddings2: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels (B,)
            embeddings1: Embeddings for first image (for triplet loss)
            embeddings2: Embeddings for second image (for triplet loss)
        Returns:
            Dictionary of losses
        """
        # Binary cross-entropy loss for similarity
        similarity = predictions['similarity'].squeeze()
        bce_loss = self.bce_loss(similarity, labels)
        
        # Center loss (encourage similar embeddings for same twins)
        center_loss = torch.tensor(0.0, device=labels.device)
        if self.center_weight > 0 and embeddings1 is not None and embeddings2 is not None:
            positive_mask = labels == 1
            if positive_mask.sum() > 0:
                pos_emb1 = embeddings1[positive_mask]
                pos_emb2 = embeddings2[positive_mask]
                center_loss = self.mse_loss(pos_emb1, pos_emb2)
        
        # Attention regularization (encourage sparse attention)
        attention_reg_loss = torch.tensor(0.0, device=labels.device)
        if 'attention_info' in predictions and self.attention_reg_weight > 0:
            attention_info = predictions['attention_info']
            for key, attn_maps in attention_info.items():
                if isinstance(attn_maps, torch.Tensor):
                    # L1 regularization on attention maps
                    attention_reg_loss += torch.mean(torch.abs(attn_maps))
                elif isinstance(attn_maps, list):
                    for attn_map in attn_maps:
                        attention_reg_loss += torch.mean(torch.abs(attn_map))
        
        # Total loss
        total_loss = (bce_loss + 
                     self.center_weight * center_loss + 
                     self.attention_reg_weight * attention_reg_loss)
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'center_loss': center_loss,
            'attention_reg_loss': attention_reg_loss
        }


def create_model(config: Dict[str, Any]) -> Tuple[ATVNModel, ATVNLoss]:
    """Create ATVN model and loss function."""
    model = ATVNModel(config)
    loss_fn = ATVNLoss(config)
    
    return model, loss_fn