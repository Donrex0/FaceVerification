"""
Model Evaluation Script for Face Verification
Evaluates trained models and saves comprehensive results.
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_loader import create_data_loaders
from models.baseline_cnn import create_baseline_model
from models.improved_model import create_improved_model
from models.siamese_network import create_siamese_network
from evaluation.metrics import VerificationEvaluator
from utils.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluator."""
    
    def __init__(self, config=None):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config if config else get_config()
        self.device = self.config.device
        
        # Load test data
        self.load_test_data()
    
    def load_test_data(self):
        """Load test dataset for evaluation."""
        logger.info("Loading test data...")
        
        data_loaders = create_data_loaders(
            data_dir=self.config.data.data_dir,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size
        )
        
        self.test_loader = data_loaders['test']
        logger.info(f"Test batches: {len(self.test_loader)}")
    
    def load_baseline_model(self, checkpoint_path=None):
        """
        Load trained baseline model.
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            
        Returns:
            Trained model
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config.training.checkpoint_dir, "best_baseline_model.pth")
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Baseline model checkpoint not found: {checkpoint_path}")
            return None
        
        # Create model architecture
        encoder = create_baseline_model(
            model_type=self.config.model.baseline_model_type,
            embedding_dim=self.config.model.baseline_embedding_dim
        )
        
        siamese_net = create_siamese_network(
            encoder=encoder,
            loss_type=self.config.model.loss_type
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        siamese_net.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Baseline model loaded from {checkpoint_path}")
        return siamese_net
    
    def load_improved_model(self, checkpoint_path=None):
        """
        Load trained improved model.
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            
        Returns:
            Trained model
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.config.training.checkpoint_dir, "best_improved_model.pth")
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Improved model checkpoint not found: {checkpoint_path}")
            return None
        
        # Create model architecture
        encoder = create_improved_model(
            model_type=self.config.model.improved_model_type,
            embedding_dim=self.config.model.improved_embedding_dim,
            pretrained=self.config.model.pretrained,
            freeze_backbone=self.config.model.freeze_backbone
        )
        
        siamese_net = create_siamese_network(
            encoder=encoder,
            loss_type=self.config.model.loss_type,
            improved=True,
            margin=self.config.model.triplet_margin
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        siamese_net.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Improved model loaded from {checkpoint_path}")
        return siamese_net
    
    def evaluate_model(self, model, model_name, threshold=0.5, find_optimal_threshold=True):
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            threshold (float): Similarity threshold
            find_optimal_threshold (bool): Whether to find optimal threshold
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Create evaluator
        evaluator = VerificationEvaluator(model, self.device)
        
        # Find optimal threshold if requested
        if find_optimal_threshold:
            logger.info("Finding optimal threshold...")
            optimal_threshold = evaluator.find_optimal_threshold(self.test_loader, metric='f1')
            logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
            threshold = optimal_threshold
        
        # Evaluate model
        results = evaluator.evaluate_dataset(self.test_loader, threshold=threshold)
        
        # Add metadata
        results['model_name'] = model_name
        results['threshold_used'] = threshold
        results['evaluation_date'] = datetime.now().isoformat()
        results['test_samples'] = len(self.test_loader.dataset)
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all available models."""
        results = {}
        
        # Evaluate baseline model
        baseline_model = self.load_baseline_model()
        if baseline_model is not None:
            results['baseline'] = self.evaluate_model(baseline_model, "Baseline CNN")
        
        # Evaluate improved model
        improved_model = self.load_improved_model()
        if improved_model is not None:
            results['improved'] = self.evaluate_model(improved_model, "Improved ResNet50")
        
        return results
    
    def save_results(self, results, output_file=None):
        """
        Save evaluation results to file.
        
        Args:
            results (dict): Evaluation results
            output_file (str): Output file path
        """
        if output_file is None:
            output_file = os.path.join(self.config.evaluation.results_dir, "model_evaluation_results.json")
        
        # Prepare results for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def print_summary(self, results):
        """
        Print evaluation summary.
        
        Args:
            results (dict): Evaluation results
        """
        print("=" * 80)
        print("FACE VERIFICATION MODEL EVALUATION SUMMARY")
        print("=" * 80)
        
        for model_name, model_results in results.items():
            print(f"\n{model_name.upper()} MODEL:")
            print("-" * 40)
            
            if 'basic' in model_results:
                basic = model_results['basic']
                print(f"Accuracy:  {basic.get('accuracy', 0):.4f}")
                print(f"Precision: {basic.get('precision', 0):.4f}")
                print(f"Recall:    {basic.get('recall', 0):.4f}")
                print(f"F1 Score:  {basic.get('f1_score', 0):.4f}")
            
            if 'roc' in model_results:
                roc = model_results['roc']
                print(f"ROC AUC:   {roc.get('roc_auc', 0):.4f}")
                print(f"Optimal Threshold: {roc.get('optimal_threshold', 0):.4f}")
            
            if 'embedding_stats' in model_results:
                stats = model_results['embedding_stats']
                print(f"Total Pairs: {stats.get('total_pairs', 0)}")
                print(f"Positive Pairs: {stats.get('positive_pairs', 0)}")
                print(f"Negative Pairs: {stats.get('negative_pairs', 0)}")
        
        print("\n" + "=" * 80)
    
    def generate_comparison_table(self, results):
        """
        Generate comparison table for models.
        
        Args:
            results (dict): Evaluation results
            
        Returns:
            dict: Comparison data
        """
        comparison = {
            'models': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': [],
            'optimal_threshold': []
        }
        
        for model_name, model_results in results.items():
            comparison['models'].append(model_name)
            
            if 'basic' in model_results:
                basic = model_results['basic']
                comparison['accuracy'].append(basic.get('accuracy', 0))
                comparison['precision'].append(basic.get('precision', 0))
                comparison['recall'].append(basic.get('recall', 0))
                comparison['f1_score'].append(basic.get('f1_score', 0))
            else:
                comparison['accuracy'].append(0)
                comparison['precision'].append(0)
                comparison['recall'].append(0)
                comparison['f1_score'].append(0)
            
            if 'roc' in model_results:
                roc = model_results['roc']
                comparison['roc_auc'].append(roc.get('roc_auc', 0))
                comparison['optimal_threshold'].append(roc.get('optimal_threshold', 0))
            else:
                comparison['roc_auc'].append(0)
                comparison['optimal_threshold'].append(0)
        
        return comparison
    
    def save_comparison_table(self, comparison, output_file=None):
        """
        Save comparison table to CSV file.
        
        Args:
            comparison (dict): Comparison data
            output_file (str): Output file path
        """
        if output_file is None:
            output_file = os.path.join(self.config.evaluation.results_dir, "comparison_table.csv")
        
        import csv
        
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Model', 'Accuracy', 'AUC', 'F1 Score', 'Precision', 'Recall', 'Optimal Threshold']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(comparison['models'])):
                writer.writerow({
                    'Model': comparison['models'][i],
                    'Accuracy': f"{comparison['accuracy'][i]:.4f}",
                    'AUC': f"{comparison['roc_auc'][i]:.4f}",
                    'F1 Score': f"{comparison['f1_score'][i]:.4f}",
                    'Precision': f"{comparison['precision'][i]:.4f}",
                    'Recall': f"{comparison['recall'][i]:.4f}",
                    'Optimal Threshold': f"{comparison['optimal_threshold'][i]:.4f}"
                })
        
        logger.info(f"Comparison table saved to {output_file}")

def main():
    """Main evaluation function."""
    logger.info("Starting model evaluation...")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results = evaluator.evaluate_all_models()
    
    if not results:
        logger.error("No models found for evaluation!")
        return
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_results(results)
    
    # Generate and save comparison table
    comparison = evaluator.generate_comparison_table(results)
    evaluator.save_comparison_table(comparison)
    
    logger.info("Model evaluation completed successfully!")

if __name__ == "__main__":
    main()
