"""
Training script for baseline face verification model.
Implements training loop with contrastive loss for Siamese network.
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

from models.baseline_cnn import create_baseline_model
from models.siamese_network import create_siamese_network
from dataset.dataset_loader import create_data_loaders
from utils.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineTrainer:
    """Trainer for baseline face verification model."""
    
    def __init__(self, config=None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config if config else get_config()
        self.device = self.config.device
        
        # Initialize model
        self.encoder = create_baseline_model(
            model_type=self.config.model.baseline_model_type,
            embedding_dim=self.config.model.baseline_embedding_dim
        )
        
        self.siamese_net = create_siamese_network(
            encoder=self.encoder,
            loss_type=self.config.model.loss_type
        )
        
        self.siamese_net.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.siamese_net.parameters(),
            lr=self.config.training.baseline_lr,
            weight_decay=self.config.training.baseline_weight_decay
        )
        
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
        
        logger.info("Baseline trainer initialized")
    
    def load_data(self):
        """Load training and validation data."""
        logger.info("Loading data...")
        
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
    
    def train_epoch(self):
        """Train for one epoch."""
        self.siamese_net.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
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
            self.optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                similarity, prediction = self.siamese_net.predict_similarity(img1, img2)
                correct_predictions += (prediction.squeeze() == labels).sum().item()
                total_samples += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
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
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct_predictions/total_samples:.4f}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop."""
        logger.info("Starting baseline model training...")
        
        # Load data
        self.load_data()
        
        # Training loop
        for epoch in range(self.config.training.baseline_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.training.baseline_epochs}")
            
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
                self.save_model("best_baseline_model.pth")
                logger.info(f"New best model saved at epoch {epoch + 1}")
            
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
        
        output_file = os.path.join(self.config.evaluation.results_dir, "baseline_results.json")
        
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
        
        # Plot accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.config.evaluation.plot_dir, "baseline_training_curves.png")
        plt.savefig(plot_file, dpi=self.config.evaluation.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {plot_file}")

def main():
    """Main training function."""
    logger.info("Starting baseline model training...")
    
    # Create trainer
    trainer = BaselineTrainer()
    
    # Start training
    trainer.train()
    
    logger.info("Baseline model training completed successfully!")

if __name__ == "__main__":
    main()
