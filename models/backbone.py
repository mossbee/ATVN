import torch
import torch.nn as nn
import torchvision.models as models
from facenet_pytorch import InceptionResnetV1
import timm
from typing import Dict, Any


class FaceNetBackbone(nn.Module):
    """FaceNet backbone using InceptionResnetV1."""
    
    def __init__(self, pretrained=True, embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load pre-trained FaceNet
        self.facenet = InceptionResnetV1(
            pretrained='vggface2' if pretrained else None,
            classify=False,
            num_classes=None
        )
        
        # Get the dimension of FaceNet features
        facenet_dim = 512
        
        # Feature extraction layers (before final classification)
        self.feature_extractor = nn.Sequential(*list(self.facenet.children())[:-1])
        
        # Adaptive pooling to get consistent feature maps
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Projection to desired embedding dimension
        if facenet_dim != embedding_dim:
            self.projection = nn.Linear(facenet_dim, embedding_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x, return_feature_maps=False):
        """
        Args:
            x: (B, 3, H, W)
            return_feature_maps: Whether to return intermediate feature maps
        Returns:
            embeddings: (B, embedding_dim)
            feature_maps: (B, C, H', W') if return_feature_maps=True
        """
        # Extract features through FaceNet layers
        features = x
        feature_maps = None
        
        for i, layer in enumerate(self.feature_extractor):
            features = layer(features)
            # Capture feature maps from a middle layer for attention
            if i == len(self.feature_extractor) - 3:  # Third last layer
                feature_maps = features
        
        # Global average pooling
        embeddings = torch.mean(features.view(features.size(0), features.size(1), -1), dim=2)
        
        # Project to desired dimension
        embeddings = self.projection(embeddings)
        
        if return_feature_maps:
            return embeddings, feature_maps
        return embeddings


class ArcFaceBackbone(nn.Module):
    """ArcFace backbone using ResNet."""
    
    def __init__(self, pretrained=True, embedding_dim=512, backbone='resnet50'):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x, return_feature_maps=False):
        """
        Args:
            x: (B, 3, H, W)
            return_feature_maps: Whether to return intermediate feature maps
        Returns:
            embeddings: (B, embedding_dim)
            feature_maps: (B, C, H', W') if return_feature_maps=True
        """
        # Extract features
        feature_maps = self.backbone(x)  # (B, C, H', W')
        
        # Global pooling
        pooled_features = self.global_pool(feature_maps)  # (B, C, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # (B, C)
        
        # Project to embedding space
        embeddings = self.projection(pooled_features)
        
        if return_feature_maps:
            return embeddings, feature_maps
        return embeddings


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for face recognition."""
    
    def __init__(self, pretrained=True, embedding_dim=512, model_name='efficientnet_b0'):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Load EfficientNet
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='',  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            backbone_dim = dummy_output.shape[1]
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, x, return_feature_maps=False):
        """
        Args:
            x: (B, 3, H, W)
            return_feature_maps: Whether to return intermediate feature maps
        Returns:
            embeddings: (B, embedding_dim)
            feature_maps: (B, C, H', W') if return_feature_maps=True
        """
        # Extract features
        feature_maps = self.backbone(x)  # (B, C, H', W')
        
        # Global pooling
        pooled_features = self.global_pool(feature_maps)  # (B, C, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # (B, C)
        
        # Project to embedding space
        embeddings = self.projection(pooled_features)
        
        if return_feature_maps:
            return embeddings, feature_maps
        return embeddings


class BackboneFactory:
    """Factory for creating backbone models."""
    
    @staticmethod
    def create_backbone(config: Dict[str, Any]) -> nn.Module:
        """Create backbone model based on configuration."""
        backbone_type = config['model']['backbone'].lower()
        embedding_dim = config['model']['embedding_dim']
        pretrained = config['model']['pretrained']
        
        if backbone_type == 'facenet':
            return FaceNetBackbone(
                pretrained=pretrained,
                embedding_dim=embedding_dim
            )
        elif backbone_type == 'arcface':
            return ArcFaceBackbone(
                pretrained=pretrained,
                embedding_dim=embedding_dim,
                backbone='resnet50'
            )
        elif backbone_type == 'resnet50':
            return ArcFaceBackbone(
                pretrained=pretrained,
                embedding_dim=embedding_dim,
                backbone='resnet50'
            )
        elif backbone_type == 'resnet101':
            return ArcFaceBackbone(
                pretrained=pretrained,
                embedding_dim=embedding_dim,
                backbone='resnet101'
            )
        elif backbone_type.startswith('efficientnet'):
            return EfficientNetBackbone(
                pretrained=pretrained,
                embedding_dim=embedding_dim,
                model_name=backbone_type
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")


class FeatureExtractor(nn.Module):
    """Feature extractor with multiple scale outputs."""
    
    def __init__(self, backbone, feature_scales=[1, 2, 4]):
        super().__init__()
        self.backbone = backbone
        self.feature_scales = feature_scales
        
    def forward(self, x):
        """
        Extract features at multiple scales.
        
        Args:
            x: (B, 3, H, W)
        Returns:
            embeddings: (B, embedding_dim)
            feature_maps: (B, C, H', W')
            multi_scale_features: List of feature maps at different scales
        """
        # Get embeddings and feature maps
        embeddings, feature_maps = self.backbone(x, return_feature_maps=True)
        
        # Extract multi-scale features
        multi_scale_features = []
        B, C, H, W = feature_maps.shape
        
        for scale in self.feature_scales:
            if scale == 1:
                scale_features = feature_maps
            else:
                scale_h, scale_w = H // scale, W // scale
                scale_features = torch.nn.functional.adaptive_avg_pool2d(
                    feature_maps, (scale_h, scale_w)
                )
            multi_scale_features.append(scale_features)
        
        return embeddings, feature_maps, multi_scale_features


def freeze_backbone(model, freeze=True):
    """Freeze or unfreeze backbone parameters."""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
    elif hasattr(model, 'facenet'):
        for param in model.facenet.parameters():
            param.requires_grad = not freeze
    else:
        print("Warning: Could not find backbone to freeze/unfreeze")


def get_backbone_info(backbone_type: str) -> Dict[str, Any]:
    """Get information about a backbone model."""
    info = {
        'facenet': {
            'input_size': (160, 160),
            'embedding_dim': 512,
            'pretrained_weights': 'vggface2',
            'description': 'FaceNet InceptionResnetV1 pre-trained on VGGFace2'
        },
        'arcface': {
            'input_size': (224, 224),
            'embedding_dim': 512,
            'pretrained_weights': 'imagenet',
            'description': 'ResNet50 backbone for ArcFace'
        },
        'resnet50': {
            'input_size': (224, 224),
            'embedding_dim': 2048,
            'pretrained_weights': 'imagenet',
            'description': 'ResNet50 pre-trained on ImageNet'
        },
        'efficientnet_b0': {
            'input_size': (224, 224),
            'embedding_dim': 1280,
            'pretrained_weights': 'imagenet',
            'description': 'EfficientNet-B0 pre-trained on ImageNet'
        }
    }
    
    return info.get(backbone_type.lower(), {
        'input_size': (224, 224),
        'embedding_dim': 512,
        'pretrained_weights': 'unknown',
        'description': f'Unknown backbone: {backbone_type}'
    })
