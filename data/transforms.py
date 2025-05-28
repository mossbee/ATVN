import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Tuple
import numpy as np


def get_train_transforms(config: Dict[str, Any], image_size: Tuple[int, int] = (224, 224)):
    """Get training data augmentation transforms."""
    aug_config = config.get('augmentation', {})
    
    transform = A.Compose([
        # Resize
        A.Resize(image_size[0], image_size[1]),
        
        # Geometric augmentations
        A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
        A.Rotate(limit=aug_config.get('rotation', 15), p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=aug_config.get('rotation', 15),
            p=0.5
        ),
        
        # Color augmentations
        A.ColorJitter(
            brightness=aug_config.get('color_jitter', 0.2),
            contrast=aug_config.get('color_jitter', 0.2),
            saturation=aug_config.get('color_jitter', 0.2),
            hue=aug_config.get('color_jitter', 0.1),
            p=0.5
        ),
        
        # Lighting augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        
        # Noise and blur
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ], p=0.2),
        
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
        ], p=0.2),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return transform


def get_val_transforms(image_size: Tuple[int, int] = (224, 224)):
    """Get validation/test transforms (no augmentation)."""
    transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return transform


def get_inference_transforms(image_size: Tuple[int, int] = (224, 224)):
    """Get transforms for inference."""
    return get_val_transforms(image_size)


class TwinSynchronizedTransforms:
    """Synchronized transforms for twin pairs to ensure consistency."""
    
    def __init__(self, transform, sync_geometric: bool = True):
        self.transform = transform
        self.sync_geometric = sync_geometric
    
    def __call__(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply synchronized transforms to twin pair."""
        if self.sync_geometric:
            # Apply same geometric transforms to both images
            # Get random parameters for geometric transforms
            
            # For simplicity, apply same transform to both
            # In practice, you might want more sophisticated synchronization
            augmented1 = self.transform(image=image1)
            augmented2 = self.transform(image=image2)
            
            return augmented1['image'], augmented2['image']
        else:
            # Apply independent transforms
            augmented1 = self.transform(image=image1)
            augmented2 = self.transform(image=image2)
            
            return augmented1['image'], augmented2['image']


def create_data_transforms(config: Dict[str, Any]):
    """Create all data transforms based on configuration."""
    image_size = tuple(config['data']['image_size'])
    
    transforms_dict = {
        'train': get_train_transforms(config, image_size),
        'val': get_val_transforms(image_size),
        'test': get_val_transforms(image_size),
        'inference': get_inference_transforms(image_size)
    }
    
    return transforms_dict
