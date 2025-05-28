import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.atvn import ATVNModel
from models.losses import CombinedTwinLoss
from data.dataset import TwinDataset, TripletDataset
from data.transforms import create_data_transforms
from utils.metrics import TwinMetrics
from utils.inference import verify_twin_pair


def setup_logging(log_dir: str):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: Dict[str, Any]):
    """Create train, validation, and test data loaders."""
    # Get transforms
    transforms = create_data_transforms(config)
    
    # Create datasets
    train_dataset = TripletDataset(
        dataset_root=config['data']['dataset_root'],
        pairs_file=config['data']['pairs_file'],
        split='train',
        image_size=tuple(config['data']['image_size']),
        transform=transforms['train']
    )
    
    val_dataset = TwinDataset(
        dataset_root=config['data']['dataset_root'],
        pairs_file=config['data']['pairs_file'],
        split='val',
        image_size=tuple(config['data']['image_size']),
        transform=transforms['val']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: Dict[str, Any]):
    """Create optimizer based on configuration."""
    opt_config = config['optimizer']
    params = model.parameters()
    
    if opt_config['type'].lower() == 'adam':
        optimizer = optim.Adam(
            params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=opt_config['betas']
        )
    elif opt_config['type'].lower() == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=opt_config['betas']
        )
    elif opt_config['type'].lower() == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            momentum=opt_config['momentum']
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    return optimizer


def create_scheduler(optimizer, config: Dict[str, Any]):
    """Create learning rate scheduler."""
    scheduler_type = config['training']['lr_scheduler']
    
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs']
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['epochs'] // 3,
            gamma=0.1
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger,
    writer: SummaryWriter,
    config: Dict[str, Any]
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        # Get batch data - handle both triplet and pair formats
        if len(batch) == 3:  # Triplet format (anchor, positive, negative)
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            # Forward pass for triplet
            anchor_out = model.get_embeddings(anchor)
            positive_out = model.get_embeddings(positive)
            negative_out = model.get_embeddings(negative)
            
            # Compute loss
            loss = criterion(anchor_out, positive_out, negative_out)
            
        else:  # Pair format (image1, image2, label)
            image1, image2, labels = batch
            image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(image1, image2)
            similarities = outputs['similarity']
            
            # Compute loss
            loss = criterion(similarities, labels.float())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        # Log to tensorboard
        if batch_idx % 100 == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    logger.info(f'Epoch {epoch} - Average training loss: {avg_loss:.4f}')
    writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
    
    return avg_loss


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger,
    writer: SummaryWriter
):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_similarities = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            image1, image2, labels = batch
            image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(image1, image2)
            similarities = outputs['similarity']
            
            # Compute loss
            loss = criterion(similarities, labels.float())
            total_loss += loss.item()
            
            # Collect predictions and labels
            all_similarities.extend(similarities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    metrics = TwinMetrics()
    metrics_dict = metrics.compute_metrics(
        np.array(all_similarities),
        np.array(all_labels)
    )
    
    avg_loss = total_loss / len(val_loader)
    
    # Log metrics
    logger.info(f'Epoch {epoch} - Validation Results:')
    logger.info(f'  Loss: {avg_loss:.4f}')
    logger.info(f'  Accuracy: {metrics_dict["accuracy"]:.4f}')
    logger.info(f'  F1-Score: {metrics_dict["f1_score"]:.4f}')
    logger.info(f'  AUC: {metrics_dict["auc"]:.4f}')
    logger.info(f'  EER: {metrics_dict["eer"]:.4f}')
    
    # Log to tensorboard
    writer.add_scalar('Loss/Val', avg_loss, epoch)
    writer.add_scalar('Metrics/Accuracy', metrics_dict['accuracy'], epoch)
    writer.add_scalar('Metrics/F1_Score', metrics_dict['f1_score'], epoch)
    writer.add_scalar('Metrics/AUC', metrics_dict['auc'], epoch)
    writer.add_scalar('Metrics/EER', metrics_dict['eer'], epoch)
    
    return avg_loss, metrics_dict


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ATVN model')
    parser.add_argument('--config', type=str, default='configs/atvn_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # Setup logging
    logger = setup_logging('logs')
    writer = SummaryWriter('runs/atvn_training')
    
    logger.info(f'Starting training with config: {args.config}')
    logger.info(f'Device: {device}')
    
    # Create data loaders
    logger.info('Creating data loaders...')
    train_loader, val_loader = create_data_loaders(config)
    logger.info(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Create model
    logger.info('Creating model...')
    model = ATVNModel(config)
    model.to(device)
    
    # Create loss function
    criterion = CombinedTwinLoss(config['loss'])
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        model, start_epoch = ATVNModel.load_from_checkpoint(args.resume, map_location=device)
        model.to(device)
    
    # Training loop
    logger.info('Starting training...')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, logger, writer, config
        )
        
        # Validate
        val_loss, metrics = validate_epoch(
            model, val_loader, criterion,
            device, epoch, logger, writer
        )
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = metrics['accuracy'] > best_acc
        if is_best:
            best_acc = metrics['accuracy']
            logger.info(f'New best accuracy: {best_acc:.4f}')
        
        # Save regular checkpoint
        checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pth'
        model.save_checkpoint(checkpoint_path, epoch, optimizer.state_dict())
        
        # Save best model
        if is_best:
            best_path = 'checkpoints/best_model.pth'
            model.save_checkpoint(best_path, epoch, optimizer.state_dict())
        
        # Save latest model
        latest_path = 'checkpoints/latest_model.pth'
        model.save_checkpoint(latest_path, epoch, optimizer.state_dict())
    
    logger.info('Training completed!')
    logger.info(f'Best validation accuracy: {best_acc:.4f}')
    
    writer.close()


if __name__ == '__main__':
    main()