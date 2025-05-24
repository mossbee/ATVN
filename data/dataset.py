import os
import json
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


class TwinDataset(Dataset):
    """Dataset for identical twin verification"""
    
    def __init__(self, data_root, pairs_file, mode='train', config=None, transform=None):
        self.data_root = data_root
        self.mode = mode
        self.config = config
        self.transform = transform
        
        # Load pairs
        with open(pairs_file, 'r') as f:
            self.twin_pairs = json.load(f)
            
        # Get all folder names
        self.all_folders = []
        for root, dirs, files in os.walk(data_root):
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                folder_name = os.path.basename(root)
                self.all_folders.append(folder_name)
        
        # Create twin pair mapping
        self.twin_dict = {}
        for pair in self.twin_pairs:
            self.twin_dict[pair[0]] = pair[1]
            self.twin_dict[pair[1]] = pair[0]
            
        # Generate training pairs
        self.pairs = self._generate_pairs()
        
    def _generate_pairs(self):
        """Generate positive and negative pairs"""
        pairs = []
        
        # Generate positive pairs (twins)
        for twin_pair in self.twin_pairs:
            folder1, folder2 = twin_pair
            
            # Get images from both folders
            images1 = self._get_images_from_folder(folder1)
            images2 = self._get_images_from_folder(folder2)
            
            # Create positive pairs
            for img1 in images1:
                for img2 in images2:
                    pairs.append((img1, img2, 1))  # 1 for twins
                    
        # Generate negative pairs (non-twins)
        non_twin_folders = [f for f in self.all_folders 
                           if f not in [item for pair in self.twin_pairs for item in pair]]
        
        # Add negative pairs from twin folders with non-twin folders
        for twin_pair in self.twin_pairs:
            for twin_folder in twin_pair:
                twin_images = self._get_images_from_folder(twin_folder)
                
                # Sample random non-twin folders
                sampled_folders = random.sample(non_twin_folders, 
                                              min(5, len(non_twin_folders)))
                
                for non_twin_folder in sampled_folders:
                    non_twin_images = self._get_images_from_folder(non_twin_folder)
                    
                    # Create negative pairs
                    for twin_img in twin_images[:2]:  # Limit to avoid too many pairs
                        for non_twin_img in non_twin_images[:2]:
                            pairs.append((twin_img, non_twin_img, 0))  # 0 for non-twins
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def _get_images_from_folder(self, folder_name):
        """Get all image paths from a folder"""
        folder_path = os.path.join(self.data_root, folder_name)
        images = []
        
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(folder_path, file))
                    
        return images
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path1, img_path2, label = self.pairs[idx]
        
        # Load images
        image1 = self._load_image(img_path1)
        image2 = self._load_image(img_path2)
        
        # Apply transforms
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        
        return {
            'image1': image1,
            'image2': image2,
            'label': torch.tensor(label, dtype=torch.long),
            'path1': img_path1,
            'path2': img_path2
        }
    
    def _load_image(self, img_path):
        """Load and preprocess image"""
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)


def get_transforms(config, mode='train'):
    """Get data transforms based on configuration"""
    
    if mode == 'train':
        transform = A.Compose([
            A.Resize(config['data']['image_size'][0], config['data']['image_size'][1]),
            A.HorizontalFlip(p=config['data']['augmentation']['horizontal_flip']),
            A.Rotate(limit=config['data']['augmentation']['rotation'], p=0.5),
            A.ColorJitter(
                brightness=config['data']['augmentation']['color_jitter']['brightness'],
                contrast=config['data']['augmentation']['color_jitter']['contrast'],
                saturation=config['data']['augmentation']['color_jitter']['saturation'],
                hue=config['data']['augmentation']['color_jitter']['hue'],
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=config['data']['augmentation']['gaussian_blur']),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                p=config['data']['augmentation']['random_erasing']
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std']
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(config['data']['image_size'][0], config['data']['image_size'][1]),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std']
            ),
            ToTensorV2()
        ])
    
    return transform


def create_dataloaders(config, data_root, pairs_file):
    """Create train and validation dataloaders"""
    
    # Get transforms
    train_transform = get_transforms(config, 'train')
    val_transform = get_transforms(config, 'val')
    
    # Create full dataset
    full_dataset = TwinDataset(
        data_root=data_root,
        pairs_file=pairs_file,
        mode='train',
        config=config,
        transform=train_transform
    )
    
    # Split dataset
    val_split = config['validation']['split_ratio']
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update transform for validation dataset
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader
