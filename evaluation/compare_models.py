"""
Model Comparison Script for Face Verification
Compares performance of baseline and improved models with statistical analysis.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparator:
    """Comprehensive model comparison with statistical analysis."""
    
    def __init__(self, config=None):
        """
        Initialize the model comparator.
        
        Args:
            config: Configuration object
        """
        self.config = config if config else get_config()
    
    def load_evaluation_results(self, results_file=None):
        """
        Load evaluation results from file.
        
        Args:
            results_file (str): Path to evaluation results file
            
        Returns:
            dict: Evaluation results
        """
        if results_file is None:
            results_file = os.path.join(self.config.evaluation.results_dir, "model_evaluation_results.json")
        
        if not os.path.exists(results_file):
            logger.error(f"Evaluation results file not found: {results_file}")
            return None
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded evaluation results from {results_file}")
        return results
    
    def create_comparison_table(self, results):
        """
        Create detailed comparison table.
        
        Args:
            results (dict): Evaluation results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            
            # Basic metrics
            if 'basic' in model_results:
                basic = model_results['basic']
                row.update({
                    'Accuracy': basic.get('accuracy', 0),
                    'Precision': basic.get('precision', 0),
                    'Recall': basic.get('recall', 0),
                    'F1 Score': basic.get('f1_score', 0)
                })
            else:
                row.update({
                    'Accuracy': 0,
                    'Precision': 0,
                    'Recall': 0,
                    'F1 Score': 0
                })
            
            # ROC metrics
            if 'roc' in model_results:
                roc = model_results['roc']
                row.update({
                    'ROC AUC': roc.get('roc_auc', 0),
                    'Optimal Threshold': roc.get('optimal_threshold', 0),
                    'Optimal Sensitivity': roc.get('optimal_sensitivity', 0),
                    'Optimal Specificity': roc.get('optimal_specificity', 0)
                })
            else:
                row.update({
                    'ROC AUC': 0,
                    'Optimal Threshold': 0,
                    'Optimal Sensitivity': 0,
                    'Optimal Specificity': 0
                })
            
            # Precision-Recall metrics
            if 'precision_recall' in model_results:
                pr = model_results['precision_recall']
                row.update({
                    'Average Precision': pr.get('average_precision', 0)
                })
            else:
                row.update({
                    'Average Precision': 0
                })
            
            # Embedding statistics
            if 'embedding_stats' in model_results:
                stats = model_results['embedding_stats']
                row.update({
                    'Total Pairs': stats.get('total_pairs', 0),
                    'Positive Pairs': stats.get('positive_pairs', 0),
                    'Negative Pairs': stats.get('negative_pairs', 0)
                })
                
                # Similarity statistics
                if 'similarity_stats' in stats:
                    sim_stats = stats['similarity_stats']
                    row.update({
                        'Mean Similarity': sim_stats.get('mean', 0),
                        'Std Similarity': sim_stats.get('std', 0)
                    })
                
                # Positive pair statistics
                if 'positive_similarity' in stats:
                    pos_sim = stats['positive_similarity']
                    row.update({
                        'Pos Mean Similarity': pos_sim.get('mean', 0),
                        'Pos Std Similarity': pos_sim.get('std', 0)
                    })
                
                # Negative pair statistics
                if 'negative_similarity' in stats:
                    neg_sim = stats['negative_similarity']
                    row.update({
                        'Neg Mean Similarity': neg_sim.get('mean', 0),
                        'Neg Std Similarity': neg_sim.get('std', 0)
                    })
            else:
                row.update({
                    'Total Pairs': 0,
                    'Positive Pairs': 0,
                    'Negative Pairs': 0,
                    'Mean Similarity': 0,
                    'Std Similarity': 0,
                    'Pos Mean Similarity': 0,
                    'Pos Std Similarity': 0,
                    'Neg Mean Similarity': 0,
                    'Neg Std Similarity': 0
                })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def save_comparison_table(self, df, output_file=None):
        """
        Save comparison table to CSV file.
        
        Args:
            df (pd.DataFrame): Comparison table
            output_file (str): Output file path
        """
        if output_file is None:
            output_file = os.path.join(self.config.evaluation.results_dir, "detailed_comparison_table.csv")
        
        # Round numeric columns for better readability
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_rounded = df.copy()
        df_rounded[numeric_columns] = df_rounded[numeric_columns].round(4)
        
        df_rounded.to_csv(output_file, index=False)
        logger.info(f"Detailed comparison table saved to {output_file}")
    
    def create_performance_summary(self, df):
        """
        Create performance summary with best model identification.
        
        Args:
            df (pd.DataFrame): Comparison table
            
        Returns:
            dict: Performance summary
        """
        summary = {}
        
        # Find best model for each metric
        metrics_to_compare = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Average Precision']
        
        for metric in metrics_to_compare:
            if metric in df.columns:
                best_idx = df[metric].idxmax()
                best_model = df.loc[best_idx, 'Model']
                best_value = df.loc[best_idx, metric]
                
                summary[f'Best_{metric.replace(" ", "_")}'] = {
                    'model': best_model,
                    'value': best_value
                }
        
        # Calculate improvements
        if len(df) >= 2:
            baseline_row = df[df['Model'] == 'baseline']
            improved_row = df[df['Model'] == 'improved']
            
            if not baseline_row.empty and not improved_row.empty:
                baseline_metrics = baseline_row.iloc[0]
                improved_metrics = improved_row.iloc[0]
                
                improvements = {}
                for metric in metrics_to_compare:
                    if metric in baseline_row.columns:
                        baseline_val = baseline_metrics[metric]
                        improved_val = improved_metrics[metric]
                        
                        if baseline_val > 0:
                            improvement = ((improved_val - baseline_val) / baseline_val) * 100
                            improvements[metric] = {
                                'baseline': baseline_val,
                                'improved': improved_val,
                                'improvement_percent': improvement
                            }
                
                summary['improvements'] = improvements
        
        return summary
    
    def plot_model_comparison(self, df, save_path=None):
        """
        Create comparison plots for models.
        
        Args:
            df (pd.DataFrame): Comparison table
            save_path (str): Path to save the plot
        """
        # Select key metrics for comparison
        key_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        if not available_metrics:
            logger.warning("No key metrics available for plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(available_metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Create bar plot
            bars = ax.bar(df['Model'], df[metric], color=colors[:len(df)], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel('Score')
            ax.set_ylim(0, max(df[metric]) * 1.1)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if needed
            if len(df['Model'].iloc[0]) > 10:
                ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.evaluation.dpi)
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_improvement_chart(self, summary, save_path=None):
        """
        Plot improvement chart showing performance gains.
        
        Args:
            summary (dict): Performance summary
            save_path (str): Path to save the plot
        """
        if 'improvements' not in summary:
            logger.warning("No improvement data available")
            return
        
        improvements = summary['improvements']
        metrics = list(improvements.keys())
        improvement_percentages = [improvements[m]['improvement_percent'] for m in metrics]
        
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        colors = ['green' if imp > 0 else 'red' for imp in improvement_percentages]
        bars = plt.barh(metrics, improvement_percentages, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, imp in zip(bars, improvement_percentages):
            width = bar.get_width()
            plt.text(width + (0.5 if width >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                    f'{imp:+.2f}%', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')
        
        plt.xlabel('Improvement Percentage')
        plt.title('Model Performance Improvements (Improved vs Baseline)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=self.config.evaluation.dpi)
            logger.info(f"Improvement chart saved to {save_path}")
        
        plt.show()
    
    def generate_comparison_report(self, results, output_dir=None):
        """
        Generate comprehensive comparison report.
        
        Args:
            results (dict): Evaluation results
            output_dir (str): Directory to save report
        """
        if output_dir is None:
            output_dir = self.config.evaluation.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison table
        df = self.create_comparison_table(results)
        
        # Save detailed comparison table
        self.save_comparison_table(df, os.path.join(output_dir, "detailed_comparison_table.csv"))
        
        # Create performance summary
        summary = self.create_performance_summary(df)
        
        # Save summary
        summary_file = os.path.join(output_dir, "performance_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Performance summary saved to {summary_file}")
        
        # Create plots
        self.plot_model_comparison(df, os.path.join(output_dir, "model_comparison_plot.png"))
        
        if 'improvements' in summary:
            self.plot_improvement_chart(summary, os.path.join(output_dir, "improvement_chart.png"))
        
        # Generate text report
        self.generate_text_report(df, summary, os.path.join(output_dir, "comparison_report.txt"))
        
        logger.info("Comparison report generated successfully!")
    
    def generate_text_report(self, df, summary, output_file):
        """
        Generate text-based comparison report.
        
        Args:
            df (pd.DataFrame): Comparison table
            summary (dict): Performance summary
            output_file (str): Output file path
        """
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FACE VERIFICATION MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Model overview
            f.write("MODEL OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Number of models compared: {len(df)}\n")
            f.write(f"Models: {', '.join(df['Model'].tolist())}\n\n")
            
            # Best performance
            f.write("BEST PERFORMANCE BY METRIC\n")
            f.write("-" * 40 + "\n")
            
            for key, value in summary.items():
                if key.startswith('Best_'):
                    metric = key.replace('Best_', '').replace('_', ' ').title()
                    f.write(f"{metric}: {value['model']} ({value['value']:.4f})\n")
            f.write("\n")
            
            # Detailed comparison table
            f.write("DETAILED COMPARISON TABLE\n")
            f.write("-" * 40 + "\n")
            
            # Round for display
            df_display = df.copy()
            numeric_columns = df_display.select_dtypes(include=[np.number]).columns
            df_display[numeric_columns] = df_display[numeric_columns].round(4)
            
            f.write(df_display.to_string(index=False))
            f.write("\n\n")
            
            # Improvements
            if 'improvements' in summary:
                f.write("PERFORMANCE IMPROVEMENTS\n")
                f.write("-" * 40 + "\n")
                f.write("Improved Model vs Baseline:\n\n")
                
                for metric, data in summary['improvements'].items():
                    f.write(f"{metric}:\n")
                    f.write(f"  Baseline: {data['baseline']:.4f}\n")
                    f.write(f"  Improved: {data['improved']:.4f}\n")
                    f.write(f"  Improvement: {data['improvement_percent']:+.2f}%\n\n")
            
            # Conclusions
            f.write("CONCLUSIONS\n")
            f.write("-" * 40 + "\n")
            
            if 'improvements' in summary:
                avg_improvement = np.mean([data['improvement_percent'] 
                                         for data in summary['improvements'].values()])
                f.write(f"Average improvement across all metrics: {avg_improvement:+.2f}%\n\n")
            
            # Find overall best model
            best_models = {}
            for key, value in summary.items():
                if key.startswith('Best_'):
                    metric = key.replace('Best_', '')
                    model = value['model']
                    if model not in best_models:
                        best_models[model] = 0
                    best_models[model] += 1
            
            if best_models:
                overall_best = max(best_models, key=best_models.get)
                f.write(f"Overall best model: {overall_best} (best in {best_models[overall_best]} metrics)\n")
        
        logger.info(f"Text report saved to {output_file}")

def main():
    """Main comparison function."""
    logger.info("Starting model comparison...")
    
    # Create comparator
    comparator = ModelComparator()
    
    # Load evaluation results
    results = comparator.load_evaluation_results()
    
    if results is None:
        logger.error("No evaluation results found. Please run evaluation first.")
        return
    
    # Generate comparison report
    comparator.generate_comparison_report(results)
    
    logger.info("Model comparison completed successfully!")

if __name__ == "__main__":
    main()
