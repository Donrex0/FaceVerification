"""
Configuration file for Face Verification Project
Contains all hyperparameters, paths, and settings.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Data configuration settings."""
    
    # Dataset paths
    data_dir: str = "data/lfw"
    pairs_file: str = "data/lfw/lfw_pairs.pkl"
    metadata_file: str = "data/lfw/metadata.pkl"
    
    # Image settings
    image_size: tuple = (160, 160)
    batch_size: int = 32
    num_workers: int = 4
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 10
    color_jitter: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.color_jitter is None:
            self.color_jitter = {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1
            }

@dataclass
class ModelConfig:
    """Model configuration settings."""
    
    # Baseline model
    baseline_model_type: str = "standard"  # 'standard' or 'simple'
    baseline_embedding_dim: int = 128
    
    # Improved model
    improved_model_type: str = "resnet50"  # 'resnet50', 'improved_resnet50', 'efficientnet_b0'
    improved_embedding_dim: int = 128
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Siamese network
    loss_type: str = "contrastive"  # 'contrastive' or 'triplet'
    contrastive_margin: float = 2.0
    triplet_margin: float = 0.2

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    
    # Baseline training
    baseline_epochs: int = 10
    baseline_lr: float = 0.001
    baseline_weight_decay: float = 1e-4
    
    # Improved training
    improved_epochs: int = 15
    improved_lr: float = 0.0001
    improved_weight_decay: float = 1e-4
    
    # Common settings
    optimizer: str = "adam"
    scheduler: str = "step"
    step_size: int = 5
    gamma: float = 0.1
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 3
    min_delta: float = 1e-4
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "checkpoints"

@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    
    # Metrics
    compute_all_metrics: bool = True
    threshold_range: tuple = (0.0, 1.0)
    threshold_steps: int = 100
    
    # Visualization
    save_plots: bool = True
    plot_dir: str = "experiments/plots"
    dpi: int = 300
    
    # Results
    save_results: bool = True
    results_dir: str = "experiments"

@dataclass
class ExperimentConfig:
    """Experiment configuration settings."""
    
    # Experiment tracking
    experiment_name: str = "face_verification_experiment"
    log_dir: str = "logs"
    seed: int = 42
    
    # Device settings
    device: str = "auto"  # 'auto', 'cpu', 'cuda'
    mixed_precision: bool = False
    
    # Reproducibility
    deterministic: bool = True
    benchmark: bool = False

class Config:
    """Main configuration class that combines all configurations."""
    
    def __init__(self):
        """Initialize all configurations."""
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.experiment = ExperimentConfig()
        
        # Set up directories
        self._setup_directories()
        
        # Set up device
        self._setup_device()
        
        # Set up reproducibility
        self._setup_reproducibility()
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.data.data_dir,
            self.training.checkpoint_dir,
            self.evaluation.plot_dir,
            self.evaluation.results_dir,
            self.experiment.log_dir,
            "models"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_device(self):
        """Set up computation device."""
        if self.experiment.device == "auto":
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.experiment.device)
        
        logger.info(f"Using device: {self.device}")
    
    def _setup_reproducibility(self):
        """Set up reproducibility settings."""
        if self.experiment.deterministic:
            import random
            import numpy as np
            import torch
            
            random.seed(self.experiment.seed)
            np.random.seed(self.experiment.seed)
            torch.manual_seed(self.experiment.seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.experiment.seed)
                torch.cuda.manual_seed_all(self.experiment.seed)
            
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = self.experiment.benchmark
            
            logger.info(f"Set random seed to {self.experiment.seed}")
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                        logger.info(f"Updated {section_name}.{key} = {value}")
                    else:
                        logger.warning(f"Unknown config key: {section_name}.{key}")
            else:
                logger.warning(f"Unknown config section: {section_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        import json
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """Load configuration from file."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.update_from_dict(config_dict)
        logger.info(f"Configuration loaded from {filepath}")
    
    def print_config(self):
        """Print current configuration."""
        print("=" * 50)
        print("FACE VERIFICATION CONFIGURATION")
        print("=" * 50)
        
        for section_name, section in [
            ("DATA", self.data),
            ("MODEL", self.model),
            ("TRAINING", self.training),
            ("EVALUATION", self.evaluation),
            ("EXPERIMENT", self.experiment)
        ]:
            print(f"\n{section_name}:")
            for key, value in section.__dict__.items():
                print(f"  {key}: {value}")
        
        print("=" * 50)

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def update_config(config_dict: Dict[str, Any]):
    """Update the global configuration."""
    config.update_from_dict(config_dict)

def create_experiment_config(
    experiment_name: str,
    model_type: str = "resnet50",
    loss_type: str = "contrastive",
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Create a configuration dictionary for a specific experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        model_type (str): Type of model to use
        loss_type (str): Type of loss function
        epochs (int): Number of training epochs
        lr (float): Learning rate
        batch_size (int): Batch size
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return {
        'experiment': {
            'experiment_name': experiment_name
        },
        'model': {
            'improved_model_type': model_type,
            'loss_type': loss_type
        },
        'training': {
            'improved_epochs': epochs,
            'improved_lr': lr
        },
        'data': {
            'batch_size': batch_size
        }
    }

def test_config():
    """Test configuration functionality."""
    # Create config
    cfg = Config()
    
    # Print config
    cfg.print_config()
    
    # Test update
    update_dict = {
        'model': {
            'improved_model_type': 'efficientnet_b0',
            'embedding_dim': 256
        },
        'training': {
            'improved_epochs': 20,
            'improved_lr': 0.0005
        }
    }
    
    cfg.update_from_dict(update_dict)
    
    print("\nAfter update:")
    cfg.print_config()
    
    # Test save/load
    cfg.save_config("test_config.json")
    
    new_cfg = Config()
    new_cfg.load_config("test_config.json")
    
    print("\nAfter loading:")
    new_cfg.print_config()

if __name__ == "__main__":
    test_config()
