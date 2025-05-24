import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import timm
from einops import rearrange


class MultiScaleAttention(nn.Module):
    """Multi-scale attention module for fine-grained feature extraction"""
    
    def __init__(self, in_channels, scales=[1, 2, 4], dropout=0.1):
        super().__init__()
        self.scales = scales
        self.in_channels = in_channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for each scale
        self.spatial_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            ) for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Conv2d(
            in_channels * len(scales), in_channels, 1
        )
        
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # Multi-scale spatial attention
        scale_features = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                scale_x = x_channel
            else:
                # Downsample
                scale_x = F.adaptive_avg_pool2d(x_channel, 
                                              (height // scale, width // scale))
            
            # Spatial attention
            avg_pool = torch.mean(scale_x, dim=1, keepdim=True)
            max_pool, _ = torch.max(scale_x, dim=1, keepdim=True)
            spatial_input = torch.cat([avg_pool, max_pool], dim=1)
            spatial_att = self.spatial_attentions[i](spatial_input)
            
            # Apply spatial attention
            scale_x = scale_x * spatial_att
            
            # Upsample back to original size
            if scale != 1:
                scale_x = F.interpolate(scale_x, size=(height, width), 
                                      mode='bilinear', align_corners=False)
            
            scale_features.append(scale_x)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_features, dim=1)
        fused = self.scale_fusion(fused)
        fused = self.dropout(fused)
        
        return fused


class AdaptiveFusion(nn.Module):
    """Adaptive feature fusion module"""
    
    def __init__(self, attention_dim, pretrained_dim, output_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.pretrained_dim = pretrained_dim
        self.output_dim = output_dim
        
        # Feature projections
        self.attention_proj = nn.Linear(attention_dim, output_dim)
        self.pretrained_proj = nn.Linear(pretrained_dim, output_dim)
        
        # Fusion weights network
        self.fusion_gate = nn.Sequential(
            nn.Linear(attention_dim + pretrained_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, attention_feat, pretrained_feat):
        # Project features to same dimension
        att_proj = self.attention_proj(attention_feat)
        pre_proj = self.pretrained_proj(pretrained_feat)
        
        # Compute fusion weights
        concat_feat = torch.cat([attention_feat, pretrained_feat], dim=-1)
        weights = self.fusion_gate(concat_feat)
        
        # Weighted fusion
        fused = weights[:, 0:1] * att_proj + weights[:, 1:2] * pre_proj
        
        # Output projection
        output = self.output_proj(fused)
        
        return output


class ATVN(nn.Module):
    """Attention-Enhanced Twin Verification Network"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone selection
        if config['model']['backbone'] == 'facenet':
            self.backbone = InceptionResnetV1(pretrained='vggface2')
            backbone_dim = 512
        else:
            # Alternative: Use ResNet backbone
            self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
            backbone_dim = 2048
            
        # Freeze backbone initially (can be unfrozen later)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Attention branch
        self.attention_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),
        )
        
        # Multi-scale attention
        self.attention_module = MultiScaleAttention(
            in_channels=512,
            scales=config['model']['attention']['scales'],
            dropout=config['model']['attention']['dropout']
        )
        
        # Global average pooling for attention features
        self.attention_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature fusion
        self.fusion = AdaptiveFusion(
            attention_dim=512,
            pretrained_dim=backbone_dim,
            output_dim=config['model']['fusion']['output_dim']
        )
        
        # Verification head
        head_dims = config['model']['head']['hidden_dims']
        layers = []
        in_dim = config['model']['fusion']['output_dim']
        
        for hidden_dim in head_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config['model']['head']['dropout'])
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 1))  # Binary classification
        self.verification_head = nn.Sequential(*layers)
        
    def _make_layer(self, in_channels, out_channels, stride):
        """Create a ResNet-like layer"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def extract_features(self, x):
        """Extract features from a single image"""
        # Attention branch
        att_feat = self.attention_conv(x)
        att_feat = self.attention_module(att_feat)
        att_feat = self.attention_pool(att_feat).flatten(1)
        
        # Pre-trained branch
        if self.config['model']['backbone'] == 'facenet':
            pre_feat = self.backbone(x)
        else:
            pre_feat = self.backbone(x)
            
        # Feature fusion
        fused_feat = self.fusion(att_feat, pre_feat)
        
        return fused_feat
        
    def forward(self, x1, x2):
        """Forward pass for twin verification"""
        # Extract features for both images
        feat1 = self.extract_features(x1)
        feat2 = self.extract_features(x2)
        
        # Compute similarity features
        # Concatenate features along with their difference and element-wise product
        diff = torch.abs(feat1 - feat2)
        prod = feat1 * feat2
        
        # Combine all similarity features
        similarity_feat = torch.cat([feat1, feat2, diff, prod], dim=1)
        
        # Expand verification head to handle concatenated features
        if not hasattr(self, '_head_adjusted'):
            input_dim = similarity_feat.shape[1]
            first_layer = self.verification_head[0]
            if first_layer.in_features != input_dim:
                # Recreate first layer with correct input dimension
                new_first_layer = nn.Linear(input_dim, first_layer.out_features)
                nn.init.xavier_uniform_(new_first_layer.weight)
                nn.init.zeros_(new_first_layer.bias)
                self.verification_head[0] = new_first_layer
            self._head_adjusted = True
        
        # Verification prediction
        similarity_score = self.verification_head(similarity_feat)
        
        return similarity_score.squeeze(-1)
        
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
