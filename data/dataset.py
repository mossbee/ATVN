import os
import json
import random
from typing import List, Tuple, Dict, Any
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class TwinDataset(Dataset):
    """Dataset for identical twin verification."""
    
    def __init__(
        self,
        dataset_root: str,
        pairs_file: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        augmentation_config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            dataset_root: Root directory containing image folders
            pairs_file: JSON file containing twin pairs
            split: 'train', 'val', or 'test'
            image_size: Target image size (H, W)
            augmentation_config: Data augmentation configuration
        """
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.split = split
        
        # Load twin pairs
        with open(pairs_file, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Get all image folders
        self.image_folders = [d for d in os.listdir(dataset_root) 
                             if os.path.isdir(os.path.join(dataset_root, d))]
        
        # Create folder to images mapping
        self.folder_images = {}
        for folder in self.image_folders:
            folder_path = os.path.join(dataset_root, folder)
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.folder_images[folder] = images
        
        # Create positive and negative pairs for training
        self.pairs = self._create_pairs()
        
        # Setup transforms
        self.transform = self._get_transforms(augmentation_config)
        
    def _create_pairs(self) -> List[Tuple[str, str, int]]:
        """Create positive and negative pairs."""
        pairs = []
        
        # Positive pairs (twins)
        for twin_pair in self.twin_pairs:
            folder1, folder2 = twin_pair
            if folder1 in self.folder_images and folder2 in self.folder_images:
                # Create multiple positive pairs
                for _ in range(min(5, len(self.folder_images[folder1]), len(self.folder_images[folder2]))):
                    img1 = random.choice(self.folder_images[folder1])
                    img2 = random.choice(self.folder_images[folder2])
                    pairs.append((f"{folder1}/{img1}", f"{folder2}/{img2}", 1))
        
        # Negative pairs (non-twins)
        twin_folders = set()
        for twin_pair in self.twin_pairs:
            twin_folders.update(twin_pair)
        
        non_twin_folders = [f for f in self.image_folders if f not in twin_folders]
        
        # Create negative pairs (equal number to positive pairs)
        num_negatives = len(pairs)
        for _ in range(num_negatives):
            # Random two different folders
            folder1, folder2 = random.sample(self.image_folders, 2)
            
            # Ensure they're not twins
            is_twin_pair = any(set([folder1, folder2]) == set(twin_pair) 
                             for twin_pair in self.twin_pairs)
            if not is_twin_pair:
                img1 = random.choice(self.folder_images[folder1])
                img2 = random.choice(self.folder_images[folder2])
                pairs.append((f"{folder1}/{img1}", f"{folder2}/{img2}", 0))
        
        # Shuffle pairs
        random.shuffle(pairs)
        return pairs
    
    def _get_transforms(self, aug_config: Dict[str, Any] = None):
        """Get image transforms based on split and config."""
        if aug_config is None:
            aug_config = {}
        
        if self.split == 'train':
            transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
                A.Rotate(limit=aug_config.get('rotation', 15), p=0.5),
                A.ColorJitter(
                    brightness=aug_config.get('color_jitter', 0.2),
                    contrast=aug_config.get('color_jitter', 0.2),
                    saturation=aug_config.get('color_jitter', 0.2),
                    hue=aug_config.get('color_jitter', 0.1),
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=3, p=aug_config.get('gaussian_blur', 0.1)),
                A.Normalize(
                    mean=aug_config.get('normalize_mean', [0.485, 0.456, 0.406]),
                    std=aug_config.get('normalize_std', [0.229, 0.224, 0.225])
                ),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=aug_config.get('normalize_mean', [0.485, 0.456, 0.406]),
                    std=aug_config.get('normalize_std', [0.229, 0.224, 0.225])
                ),
                ToTensorV2()
            ])
        
        return transform
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = Image.open(os.path.join(self.dataset_root, img1_path)).convert('RGB')
        img2 = Image.open(os.path.join(self.dataset_root, img2_path)).convert('RGB')
        
        # Convert to numpy for albumentations
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(image=img1)['image']
            img2 = self.transform(image=img2)['image']
        
        return {
            'image1': img1,
            'image2': img2,
            'label': torch.tensor(label, dtype=torch.float32),
            'path1': img1_path,
            'path2': img2_path
        }


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    
    # Split twin pairs for train/val/test
    with open(config['data']['pairs_file'], 'r') as f:
        all_pairs = json.load(f)
    
    random.shuffle(all_pairs)
    
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    
    n_total = len(all_pairs)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]
    
    # Create temporary pair files for each split
    import tempfile
    train_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    val_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    
    json.dump(train_pairs, train_file)
    json.dump(val_pairs, val_file)
    json.dump(test_pairs, test_file)
    
    train_file.close()
    val_file.close() 
    test_file.close()
    
    # Create datasets
    train_dataset = TwinDataset(
        dataset_root=config['data']['dataset_root'],
        pairs_file=train_file.name,
        split='train',
        image_size=config['data']['image_size'],
        augmentation_config=config.get('augmentation', {})
    )
    
    val_dataset = TwinDataset(
        dataset_root=config['data']['dataset_root'],
        pairs_file=val_file.name,
        split='val',
        image_size=config['data']['image_size'],
        augmentation_config=config.get('augmentation', {})
    )
    
    test_dataset = TwinDataset(
        dataset_root=config['data']['dataset_root'],
        pairs_file=test_file.name,
        split='test',
        image_size=config['data']['image_size'],
        augmentation_config=config.get('augmentation', {})
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Clean up temporary files
    os.unlink(train_file.name)
    os.unlink(val_file.name)
    os.unlink(test_file.name)
    
    return train_loader, val_loader, test_loader


class TripletDataset(Dataset):
    """Dataset for triplet loss training."""
    
    def __init__(
        self,
        dataset_root: str,
        pairs_file: str,
        image_size: Tuple[int, int] = (224, 224),
        augmentation_config: Dict[str, Any] = None,
        triplets_per_epoch: int = 10000
    ):
        """
        Args:
            dataset_root: Root directory containing image folders
            pairs_file: JSON file containing twin pairs
            image_size: Target image size (H, W)
            augmentation_config: Data augmentation configuration
            triplets_per_epoch: Number of triplets to generate per epoch
        """
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.triplets_per_epoch = triplets_per_epoch
        
        # Load twin pairs
        with open(pairs_file, 'r') as f:
            self.twin_pairs = json.load(f)
        
        # Get all image folders
        self.image_folders = [d for d in os.listdir(dataset_root) 
                             if os.path.isdir(os.path.join(dataset_root, d))]
        
        # Create folder to images mapping
        self.folder_images = {}
        for folder in self.image_folders:
            folder_path = os.path.join(dataset_root, folder)
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            self.folder_images[folder] = images
        
        # Setup transforms
        self.transform = self._get_transforms(augmentation_config)
        
        # Generate triplets
        self.triplets = self._generate_triplets()
    
    def _generate_triplets(self) -> List[Tuple[str, str, str]]:
        """Generate anchor-positive-negative triplets."""
        triplets = []
        
        for _ in range(self.triplets_per_epoch):
            # Choose a random twin pair for anchor and positive
            twin_pair = random.choice(self.twin_pairs)
            anchor_folder, positive_folder = twin_pair
            
            if (anchor_folder not in self.folder_images or 
                positive_folder not in self.folder_images):
                continue
            
            # Select random images from anchor and positive folders
            anchor_img = random.choice(self.folder_images[anchor_folder])
            positive_img = random.choice(self.folder_images[positive_folder])
            
            # Select negative image (not from twin pair)
            negative_folders = [f for f in self.image_folders 
                              if f not in twin_pair]
            negative_folder = random.choice(negative_folders)
            negative_img = random.choice(self.folder_images[negative_folder])
            
            triplets.append((
                f"{anchor_folder}/{anchor_img}",
                f"{positive_folder}/{positive_img}",
                f"{negative_folder}/{negative_img}"
            ))
        
        return triplets
    
    def _get_transforms(self, aug_config: Dict[str, Any] = None):
        """Get image transforms."""
        if aug_config is None:
            aug_config = {}
        
        transform = A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
            A.Rotate(limit=aug_config.get('rotation', 15), p=0.5),
            A.ColorJitter(
                brightness=aug_config.get('color_jitter', 0.2),
                contrast=aug_config.get('color_jitter', 0.2),
                saturation=aug_config.get('color_jitter', 0.2),
                hue=aug_config.get('color_jitter', 0.1),
                p=0.5
            ),
            A.Normalize(
                mean=aug_config.get('normalize_mean', [0.485, 0.456, 0.406]),
                std=aug_config.get('normalize_std', [0.229, 0.224, 0.225])
            ),
            ToTensorV2()
        ])
        
        return transform
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        # Load images
        anchor = Image.open(os.path.join(self.dataset_root, anchor_path)).convert('RGB')
        positive = Image.open(os.path.join(self.dataset_root, positive_path)).convert('RGB')
        negative = Image.open(os.path.join(self.dataset_root, negative_path)).convert('RGB')
        
        # Convert to numpy for albumentations
        anchor = np.array(anchor)
        positive = np.array(positive)
        negative = np.array(negative)
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(image=anchor)['image']
            positive = self.transform(image=positive)['image']
            negative = self.transform(image=negative)['image']
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'anchor_path': anchor_path,
            'positive_path': positive_path,
            'negative_path': negative_path
        }