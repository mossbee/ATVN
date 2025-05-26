import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x, attn


class CrossAttention(nn.Module):
    """Cross-attention module for comparing two feature maps."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2):
        """
        Args:
            x1: (B, N, C) - query features
            x2: (B, N, C) - key and value features
        """
        B, N, C = x1.shape
        
        q = self.q_proj(x1).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x2).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x, attn


class SpatialAttention(nn.Module):
    """Spatial attention module for highlighting important regions."""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attended_x: (B, C, H, W)
            attention_map: (B, 1, H, W)
        """
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention_map = self.sigmoid(attention)
        
        attended_x = x * attention_map
        
        return attended_x, attention_map


class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention, attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(in_channels, reduction)
        
    def forward(self, x):
        x, channel_attn = self.channel_attention(x)
        x, spatial_attn = self.spatial_attention(x)
        
        return x, (channel_attn, spatial_attn)


class DifferenceAttention(nn.Module):
    """Attention module that focuses on differences between two features."""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels // reduction, 1)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: (B, C, H, W)
        Returns:
            diff_features: (B, C, H, W)
            attention_map: (B, C, H, W)
        """
        # Compute absolute difference
        diff = torch.abs(x1 - x2)
        
        # Concatenate original features with difference
        combined = torch.cat([diff, x1], dim=1)
        
        # Generate attention
        attention = F.relu(self.conv1(combined))
        attention_map = self.sigmoid(self.conv2(attention))
        
        # Apply attention to difference
        diff_features = diff * attention_map
        
        return diff_features, attention_map


class MultiScaleAttention(nn.Module):
    """Multi-scale attention module."""
    
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            SpatialAttention(in_channels) for _ in scales
        ])
        self.fusion = nn.Conv2d(in_channels * len(scales), in_channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        scale_features = []
        attention_maps = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1:
                scale_x = x
            else:
                # Downsample
                scale_h, scale_w = H // scale, W // scale
                scale_x = F.adaptive_avg_pool2d(x, (scale_h, scale_w))
                
            # Apply attention
            attended_x, attn_map = self.attentions[i](scale_x)
            
            # Upsample back to original size
            if scale != 1:
                attended_x = F.interpolate(attended_x, size=(H, W), mode='bilinear', align_corners=False)
                attn_map = F.interpolate(attn_map, size=(H, W), mode='bilinear', align_corners=False)
            
            scale_features.append(attended_x)
            attention_maps.append(attn_map)
        
        # Fuse multi-scale features
        fused_features = torch.cat(scale_features, dim=1)
        fused_features = self.fusion(fused_features)
        
        return fused_features, attention_maps


class TwinAttentionModule(nn.Module):
    """Specialized attention module for twin verification."""
    
    def __init__(self, in_channels, attention_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        
        # Feature projection
        self.feature_proj = nn.Conv2d(in_channels, attention_dim, 1)
        
        # Multi-scale spatial attention
        self.multiscale_attention = MultiScaleAttention(attention_dim)
        
        # Cross-attention for comparing features
        self.cross_attention = CrossAttention(attention_dim, num_heads, dropout)
        
        # Difference attention
        self.diff_attention = DifferenceAttention(attention_dim)
        
        # Final feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(attention_dim * 3, attention_dim, 1),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, in_channels, 1)
        )
        
    def forward(self, x1, x2):
        """
        Args:
            x1, x2: (B, C, H, W) - feature maps from two images
        Returns:
            enhanced_features: (B, C, H, W)
            attention_info: dict containing attention maps
        """
        B, C, H, W = x1.shape
        
        # Project features
        f1 = self.feature_proj(x1)  # (B, attention_dim, H, W)
        f2 = self.feature_proj(x2)
        
        # Multi-scale attention
        ms_f1, ms_attn1 = self.multiscale_attention(f1)
        ms_f2, ms_attn2 = self.multiscale_attention(f2)
        
        # Flatten for cross-attention
        f1_flat = rearrange(ms_f1, 'b c h w -> b (h w) c')
        f2_flat = rearrange(ms_f2, 'b c h w -> b (h w) c')
        
        # Cross-attention
        cross_f1, cross_attn1 = self.cross_attention(f1_flat, f2_flat)
        cross_f2, cross_attn2 = self.cross_attention(f2_flat, f1_flat)
        
        # Reshape back
        cross_f1 = rearrange(cross_f1, 'b (h w) c -> b c h w', h=H, w=W)
        cross_f2 = rearrange(cross_f2, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Difference attention
        diff_features, diff_attn = self.diff_attention(cross_f1, cross_f2)
        
        # Combine features
        combined_f1 = torch.cat([ms_f1, cross_f1, diff_features], dim=1)
        combined_f2 = torch.cat([ms_f2, cross_f2, diff_features], dim=1)
        
        # Fuse features
        enhanced_f1 = self.fusion(combined_f1)
        enhanced_f2 = self.fusion(combined_f2)
        
        # Residual connection
        enhanced_f1 = x1 + enhanced_f1
        enhanced_f2 = x2 + enhanced_f2
        
        attention_info = {
            'multiscale_attn1': ms_attn1,
            'multiscale_attn2': ms_attn2,
            'cross_attn1': cross_attn1,
            'cross_attn2': cross_attn2,
            'diff_attn': diff_attn
        }
        
        return (enhanced_f1, enhanced_f2), attention_info


class AdaptiveFeatureFusion(nn.Module):
    """Adaptive fusion of global and attention features."""
    
    def __init__(self, global_dim, attention_dim, output_dim):
        super().__init__()
        self.global_proj = nn.Linear(global_dim, output_dim)
        self.attention_proj = nn.Linear(attention_dim, output_dim)
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(global_dim + attention_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, global_features, attention_features):
        """
        Args:
            global_features: (B, global_dim)
            attention_features: (B, attention_dim)
        Returns:
            fused_features: (B, output_dim)
        """
        global_proj = self.global_proj(global_features)
        attention_proj = self.attention_proj(attention_features)
        
        # Compute fusion gate
        concat_features = torch.cat([global_features, attention_features], dim=1)
        gate_weights = self.gate(concat_features)
        
        # Adaptive fusion
        fused = gate_weights * global_proj + (1 - gate_weights) * attention_proj
        
        # Final projection
        output = self.output_proj(fused)
        
        return output
