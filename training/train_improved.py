"""
Training script for improved face verification model with ResNet50 backbone.
Implements training loop with triplet loss for better performance.
"""

import os
import sys
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.improved_model import create_improved_model
from models.siamese_network import create_siamese_network
from dataset.dataset_loader import create_data_loaders, create_triplet_loader
from utils.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedTrainer:
    """Trainer for improved face verification model."""
    
    def __init__(self, config=None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config if config else get_config()
        self.device = self.config.device
        
        # Initialize model
        self.encoder = create_improved_model(
            model_type=self.config.model.improved_model_type,
            embedding_dim=self.config.model.improved_embedding_dim,
            pretrained=self.config.model.pretrained,
            freeze_backbone=self.config.model.freeze_backbone
        )
        
        self.siamese_net = create_siamese_network(
            encoder=self.encoder,
            loss_type=self.config.model.loss_type,
            improved=True,
            margin=self.config.model.triplet_margin
        )
        
        self.siamese_net.to(self.device)
        
        # Initialize optimizer with different learning rates for backbone and new layers
        backbone_params = []
        new_params = []
        
        for name, param in self.siamese_net.named_parameters():
            if 'encoder.backbone' in name:
                backbone_params.append(param)
            else:
                new_params.append(param)
        
        self.optimizer = optim.Adam([
            {'params': backbone_params, 'lr': self.config.training.improved_lr * 0.1},
            {'params': new_params, 'lr': self.config.training.improved_lr}
        ], weight_decay=self.config.training.improved_weight_decay)
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config.training.step_size,
            gamma=self.config.training.gamma
        )
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        logger.info("Improved trainer initialized")
    
    def load_data(self):
        """Load training and validation data."""
        logger.info("Loading data...")
        
        if self.config.model.loss_type == 'triplet':
            # Load triplet data for triplet loss
            self.train_loader = create_triplet_loader(
                data_dir=self.config.data.data_dir,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
                image_size=self.config.data.image_size
            )
            
            # For validation, we still need pairs
            data_loaders = create_data_loaders(
                data_dir=self.config.data.data_dir,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
                image_size=self.config.data.image_size
            )
            self.val_loader = data_loaders['test']
            
        else:
            # Load pair data for contrastive loss
            data_loaders = create_data_loaders(
                data_dir=self.config.data.data_dir,
                batch_size=self.config.data.batch_size,
                num_workers=self.config.data.num_workers,
                image_size=self.config.data.image_size
            )
            self.train_loader = data_loaders['train']
            self.val_loader = data_loaders['test']
        
        logger.info(f"Training batches: {len(self.train_loader)}")
        logger.info(f"Validation batches: {len(self.val_loader)}")
    
    def train_epoch_triplet(self):
        """Train for one epoch with triplet loss."""
        self.siamese_net.train()
        
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training (Triplet)")
        
        for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # Forward pass
            outputs = self.siamese_net(anchor, positive, negative)
            loss = self.siamese_net.compute_loss(outputs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.siamese_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_samples += anchor.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, 0.0  # Accuracy not meaningful for triplet loss
    
    def train_epoch_contrastive(self):
        """Train for one epoch with contrastive loss."""
        self.siamese_net.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training (Contrastive)")
        
        for batch_idx, (img1, img2, labels) in enumerate(progress_bar):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.siamese_net(img1, img2)
            loss = self.siamese_net.compute_loss(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.siamese_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                similarity, prediction = self.siamese_net.predict_similarity(img1, img2)
                correct_predictions += (prediction.squeeze() == labels).sum().item()
                total_samples += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train_epoch(self):
        """Train for one epoch."""
        if self.config.model.loss_type == 'triplet':
            return self.train_epoch_triplet()
        else:
            return self.train_epoch_contrastive()
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.siamese_net.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for img1, img2, labels in progress_bar:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.siamese_net(img1, img2)
                loss = self.siamese_net.compute_loss(outputs, labels)
                
                # Calculate accuracy
                similarity, prediction = self.siamese_net.predict_similarity(img1, img2)
                correct_predictions += (prediction.squeeze() == labels).sum().item()
                total_samples += labels.size(0)
                
                total_loss += loss.item()
                
                # Update progress bar
                accuracy = correct_predictions / total_samples
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop."""
        logger.info("Starting improved model training...")
        
        # Load data
        self.load_data()
        
        # Training loop
        for epoch in range(self.config.training.improved_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.training.improved_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_model("best_improved_model.pth")
                logger.info(f"New best model saved at epoch {epoch + 1}")
            
            # Unfreeze backbone after a few epochs if it was frozen
            if (self.config.model.freeze_backbone and 
                epoch == 3 and 
                hasattr(self.encoder, 'unfreeze_backbone')):
                self.encoder.unfreeze_backbone()
                logger.info("Backbone unfrozen at epoch 4")
            
            # Early stopping
            if self.config.training.early_stopping:
                if epoch - self.best_epoch >= self.config.training.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        logger.info("Training completed!")
        logger.info(f"Best epoch: {self.best_epoch + 1}, Best val loss: {self.best_val_loss:.4f}")
        
        # Save training history
        self.save_training_history()
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_model(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': len(self.train_losses),
            'model_state_dict': self.siamese_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }
        
        filepath = os.path.join(self.config.training.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filename):
        """Load model checkpoint."""
        filepath = os.path.join(self.config.training.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint not found: {filepath}")
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.siamese_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Model loaded from {filepath}")
        return True
    
    def save_training_history(self):
        """Save training history to JSON file."""
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses),
            'config': self.config.to_dict()
        }
        
        output_file = os.path.join(self.config.evaluation.results_dir, "improved_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {output_file}")
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        if not self.config.evaluation.save_plots:
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy curves (if available)
        if any(acc > 0 for acc in self.train_accuracies):
            ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
            ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, 'Accuracy not available\nfor triplet loss', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Training and Validation Accuracy')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.config.evaluation.plot_dir, "improved_training_curves.png")
        plt.savefig(plot_file, dpi=self.config.evaluation.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {plot_file}")

def main():
    """Main training function."""
    logger.info("Starting improved model training...")
    
    # Create trainer
    trainer = ImprovedTrainer()
    
    # Start training
    trainer.train()
    
    logger.info("Improved model training completed successfully!")

if __name__ == "__main__":
    main()
