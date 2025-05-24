import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from models.atvn import ATVN
from models.losses import CombinedLoss
from data.dataset import create_dataloaders
from utils.metrics import calculate_eer, plot_roc_curve
from utils.logger import Logger


class Trainer:
    """Trainer class for ATVN model"""
    
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self._create_directories()
        
        # Initialize model
        self.model = ATVN(self.config).to(self.device)
        
        # Initialize loss function
        loss_config = self.config['training']['loss']
        self.criterion = CombinedLoss(
            contrastive_margin=loss_config['contrastive_margin'],
            focal_alpha=loss_config['focal_alpha'],
            focal_gamma=loss_config['focal_gamma'],
            lambda_contrastive=loss_config['lambda_contrastive'],
            lambda_focal=loss_config['lambda_focal']
        )
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._get_scheduler()
        
        # Initialize logger
        self.logger = Logger(self.config)
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        
    def _create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config['paths']['output_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
        
    def _get_optimizer(self):
        """Get optimizer"""
        optimizer_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            
    def _get_scheduler(self):
        """Get learning rate scheduler"""
        scheduler_name = self.config['training']['scheduler'].lower()
        epochs = self.config['training']['epochs']
        warmup_epochs = self.config['training']['warmup_epochs']
        
        if scheduler_name == 'cosine':
            main_scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs - warmup_epochs)
            if warmup_epochs > 0:
                warmup_scheduler = LinearLR(
                    self.optimizer, start_factor=0.1, total_iters=warmup_epochs
                )
                return SequentialLR(
                    self.optimizer, 
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs]
                )
            return main_scheduler
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(image1, image2)
            
            # Compute loss
            loss_dict = self.criterion(outputs, labels)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / total_samples:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log batch metrics
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.logger.log_batch_metrics({
                    'train_loss': loss.item(),
                    'train_contrastive_loss': loss_dict['contrastive_loss'].item(),
                    'train_focal_loss': loss_dict['focal_loss'].item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, self.current_epoch * len(train_loader) + batch_idx)
        
        return total_loss / total_samples
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move data to device
                image1 = batch['image1'].to(self.device)
                image2 = batch['image2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(image1, image2)
                
                # Compute loss
                loss_dict = self.criterion(outputs, labels)
                total_loss += loss_dict['total_loss'].item() * len(labels)
                
                # Store predictions and labels
                all_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader.dataset)
        metrics = self._calculate_metrics(all_outputs, all_labels)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, outputs, labels):
        """Calculate validation metrics"""
        outputs = np.array(outputs)
        labels = np.array(labels)
        
        # Convert probabilities to predictions
        predictions = (outputs > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        try:
            auc = roc_auc_score(labels, outputs)
        except ValueError:
            auc = 0.0
            
        try:
            eer = calculate_eer(labels, outputs)
        except:
            eer = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'eer': eer
        }
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            self.logger.log_epoch_metrics({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_scores.append(val_metrics['f1'])
            
            # Save checkpoint
            is_best = val_metrics['f1'] > self.best_score
            if is_best:
                self.best_score = val_metrics['f1']
                
            if epoch % self.config['logging']['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch results
            print(f"Epoch {epoch + 1}/{self.config['training']['epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")
            print("-" * 50)
        
        # Save final plots
        self._save_training_plots()
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['paths']['checkpoint_dir'], 
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config['paths']['checkpoint_dir'], 
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"New best model saved with F1 score: {self.best_score:.4f}")
    
    def _save_training_plots(self):
        """Save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Validation F1 score
        axes[0, 1].plot(epochs, self.val_scores, 'g-', label='Val F1 Score')
        axes[0, 1].set_title('Validation F1 Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [self.scheduler.get_last_lr()[0] for _ in epochs]
            axes[1, 0].plot(epochs, lrs, 'm-', label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        else:
            axes[1, 0].axis('off')
        
        # Remove empty subplot
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(self.config['paths']['output_dir'], 'training_plots.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Training plots saved to {plot_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ATVN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--pairs_file', type=str, default='pairs.json', help='Pairs file name')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    # Create dataloaders
    pairs_file = os.path.join(args.data_path, args.pairs_file)
    train_loader, val_loader = create_dataloaders(
        trainer.config, args.data_path, pairs_file
    )
    
    print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
