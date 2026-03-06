"""
ROC Curve Generation and Visualization for Face Verification
Creates ROC curves, precision-recall curves, and other performance visualizations.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ROCVisualizer:
    """Visualizer for ROC curves and related performance metrics."""
    
    def __init__(self, config=None):
        """
        Initialize the ROC visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config if config else get_config()
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.dpi'] = self.config.evaluation.dpi
        plt.rcParams['savefig.dpi'] = self.config.evaluation.dpi
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, model_name="Model", save_path=None):
        """
        Plot ROC curve for a single model.
        
        Args:
            fpr (array): False positive rates
            tpr (array): True positive rates
            roc_auc (float): Area under ROC curve
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_multiple_roc_curves(self, roc_data, save_path=None):
        """
        Plot ROC curves for multiple models.
        
        Args:
            roc_data (dict): Dictionary with model names as keys and (fpr, tpr, roc_auc) as values
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown']
        
        for i, (model_name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Multiple ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, precision, recall, avg_precision, model_name="Model", save_path=None):
        """
        Plot precision-recall curve for a single model.
        
        Args:
            precision (array): Precision values
            recall (array): Recall values
            avg_precision (float): Average precision score
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Plot precision-recall curve
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'{model_name} (AP = {avg_precision:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_multiple_precision_recall_curves(self, pr_data, save_path=None):
        """
        Plot precision-recall curves for multiple models.
        
        Args:
            pr_data (dict): Dictionary with model names as keys and (precision, recall, avg_precision) as values
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (model_name, (precision, recall, avg_precision)) in enumerate(pr_data.items()):
            color = colors[i % len(colors)]
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{model_name} (AP = {avg_precision:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Multiple precision-recall curves saved to {save_path}")
        
        plt.show()
    
    def plot_threshold_analysis(self, thresholds, metrics, save_path=None):
        """
        Plot metrics vs threshold analysis.
        
        Args:
            thresholds (array): Threshold values
            metrics (dict): Dictionary with metric names as keys and values as arrays
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot each metric
        for metric_name, values in metrics.items():
            plt.plot(thresholds, values, label=metric_name, lw=2)
        
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_similarity_distribution(self, similarities, labels, model_name="Model", save_path=None):
        """
        Plot distribution of similarity scores for positive and negative pairs.
        
        Args:
            similarities (array): Similarity scores
            labels (array): Ground truth labels
            model_name (str): Name of the model
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Separate positive and negative pairs
        pos_similarities = similarities[labels == 1]
        neg_similarities = similarities[labels == 0]
        
        # Plot histograms
        plt.hist(neg_similarities, bins=50, alpha=0.7, label='Different Persons', 
                color='red', density=True)
        plt.hist(pos_similarities, bins=50, alpha=0.7, label='Same Person', 
                color='green', density=True)
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title(f'Similarity Score Distribution - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Similarity distribution plot saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_evaluation_plot(self, results, save_path=None):
        """
        Create a comprehensive evaluation plot with multiple subplots.
        
        Args:
            results (dict): Evaluation results for multiple models
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curves
        ax1 = axes[0, 0]
        colors = ['darkorange', 'blue', 'green', 'red']
        
        for i, (model_name, model_results) in enumerate(results.items()):
            if 'roc' in model_results:
                roc = model_results['roc']
                color = colors[i % len(colors)]
                ax1.plot(roc['fpr'], roc['tpr'], color=color, lw=2,
                        label=f'{model_name} (AUC = {roc["roc_auc"]:.4f})')
        
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax2 = axes[0, 1]
        for i, (model_name, model_results) in enumerate(results.items()):
            if 'precision_recall' in model_results:
                pr = model_results['precision_recall']
                color = colors[i % len(colors)]
                ax2.plot(pr['recall'], pr['precision'], color=color, lw=2,
                        label=f'{model_name} (AP = {pr["average_precision"]:.4f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy Comparison
        ax3 = axes[1, 0]
        model_names = list(results.keys())
        accuracies = []
        precisions = []
        f1_scores = []
        
        for model_name in model_names:
            if 'basic' in results[model_name]:
                basic = results[model_name]['basic']
                accuracies.append(basic.get('accuracy', 0))
                precisions.append(basic.get('precision', 0))
                f1_scores.append(basic.get('f1_score', 0))
            else:
                accuracies.append(0)
                precisions.append(0)
                f1_scores.append(0)
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax3.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
        ax3.bar(x, precisions, width, label='Precision', alpha=0.8)
        ax3.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ROC AUC Comparison
        ax4 = axes[1, 1]
        roc_aucs = []
        
        for model_name in model_names:
            if 'roc' in results[model_name]:
                roc_aucs.append(results[model_name]['roc'].get('roc_auc', 0))
            else:
                roc_aucs.append(0)
        
        bars = ax4.bar(model_names, roc_aucs, alpha=0.8, color=['darkorange', 'blue', 'green', 'red'])
        ax4.set_ylabel('ROC AUC')
        ax4.set_title('ROC AUC Comparison')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, roc_auc_val in zip(bars, roc_aucs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{roc_auc_val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Comprehensive evaluation plot saved to {save_path}")
        
        plt.show()
    
    def save_roc_data(self, roc_data, output_file):
        """
        Save ROC data to JSON file.
        
        Args:
            roc_data (dict): ROC data for multiple models
            output_file (str): Output file path
        """
        serializable_data = {}
        
        for model_name, (fpr, tpr, roc_auc) in roc_data.items():
            serializable_data[model_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'roc_auc': float(roc_auc)
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"ROC data saved to {output_file}")

def create_roc_curves_from_results(results_file, output_dir=None):
    """
    Create ROC curves from evaluation results file.
    
    Args:
        results_file (str): Path to evaluation results JSON file
        output_dir (str): Directory to save plots
    """
    if output_dir is None:
        output_dir = os.path.join(get_config().evaluation.plot_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    visualizer = ROCVisualizer()
    
    # Extract ROC data
    roc_data = {}
    pr_data = {}
    
    for model_name, model_results in results.items():
        if 'roc' in model_results:
            roc = model_results['roc']
            roc_data[model_name] = (
                np.array(roc['fpr']),
                np.array(roc['tpr']),
                roc['roc_auc']
            )
        
        if 'precision_recall' in model_results:
            pr = model_results['precision_recall']
            pr_data[model_name] = (
                np.array(pr['precision']),
                np.array(pr['recall']),
                pr['average_precision']
            )
    
    # Create plots
    if roc_data:
        visualizer.plot_multiple_roc_curves(
            roc_data, 
            save_path=os.path.join(output_dir, "roc_curves_comparison.png")
        )
    
    if pr_data:
        visualizer.plot_multiple_precision_recall_curves(
            pr_data,
            save_path=os.path.join(output_dir, "pr_curves_comparison.png")
        )
    
    # Create comprehensive plot
    visualizer.create_comprehensive_evaluation_plot(
        results,
        save_path=os.path.join(output_dir, "comprehensive_evaluation.png")
    )
    
    # Save ROC data
    if roc_data:
        visualizer.save_roc_data(
            roc_data,
            os.path.join(output_dir, "roc_data.json")
        )

def main():
    """Main function to create ROC curves from evaluation results."""
    logger.info("Creating ROC curves from evaluation results...")
    
    # Default paths
    results_file = os.path.join(get_config().evaluation.results_dir, "model_evaluation_results.json")
    output_dir = get_config().evaluation.plot_dir
    
    if not os.path.exists(results_file):
        logger.error(f"Evaluation results file not found: {results_file}")
        logger.info("Please run evaluation first: python evaluation/evaluate_models.py")
        return
    
    create_roc_curves_from_results(results_file, output_dir)
    logger.info("ROC curves created successfully!")

if __name__ == "__main__":
    main()
