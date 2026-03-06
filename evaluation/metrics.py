"""
Evaluation Metrics for Face Verification
Implements comprehensive metrics including accuracy, precision, recall, F1, ROC, AUC.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)

class FaceVerificationMetrics:
    """Comprehensive metrics for face verification evaluation."""
    
    def __init__(self, threshold=0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold (float): Similarity threshold for classification
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset all stored metrics."""
        self.predictions = []
        self.labels = []
        self.similarities = []
        self.embeddings1 = []
        self.embeddings2 = []
    
    def update(self, embeddings1, embeddings2, labels):
        """
        Update metrics with new batch.
        
        Args:
            embeddings1 (torch.Tensor): First embeddings
            embeddings2 (torch.Tensor): Second embeddings
            labels (torch.Tensor): True labels
        """
        # Convert to numpy if needed
        if isinstance(embeddings1, torch.Tensor):
            embeddings1 = embeddings1.detach().cpu().numpy()
        if isinstance(embeddings2, torch.Tensor):
            embeddings2 = embeddings2.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Compute similarities
        similarities = self._compute_cosine_similarity(embeddings1, embeddings2)
        
        # Make predictions
        predictions = (similarities >= self.threshold).astype(int)
        
        # Store results
        self.embeddings1.extend(embeddings1)
        self.embeddings2.extend(embeddings2)
        self.similarities.extend(similarities)
        self.predictions.extend(predictions)
        self.labels.extend(labels)
    
    def _compute_cosine_similarity(self, embeddings1, embeddings2):
        """Compute cosine similarity between embeddings."""
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(similarity)
        return np.array(similarities)
    
    def compute_basic_metrics(self):
        """Compute basic classification metrics."""
        if len(self.labels) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1_score': f1_score(labels, predictions, zero_division=0)
        }
        
        return metrics
    
    def compute_roc_metrics(self):
        """Compute ROC curve and AUC metrics."""
        if len(self.labels) == 0:
            return {}
        
        similarities = np.array(self.similarities)
        labels = np.array(self.labels)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'roc_auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'optimal_sensitivity': tpr[optimal_idx],
            'optimal_specificity': 1 - fpr[optimal_idx]
        }
    
    def compute_precision_recall_metrics(self):
        """Compute precision-recall curve and average precision."""
        if len(self.labels) == 0:
            return {}
        
        similarities = np.array(self.similarities)
        labels = np.array(self.labels)
        
        # Compute precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels, similarities)
        avg_precision = average_precision_score(labels, similarities)
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': avg_precision
        }
    
    def compute_confusion_matrix(self):
        """Compute confusion matrix."""
        if len(self.labels) == 0:
            return {}
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        cm = confusion_matrix(labels, predictions)
        
        return {
            'confusion_matrix': cm,
            'true_negatives': cm[0, 0],
            'false_positives': cm[0, 1],
            'false_negatives': cm[1, 0],
            'true_positives': cm[1, 1]
        }
    
    def compute_threshold_analysis(self, threshold_range=None, num_thresholds=100):
        """Analyze performance across different thresholds."""
        if len(self.labels) == 0:
            return {}
        
        similarities = np.array(self.similarities)
        labels = np.array(self.labels)
        
        if threshold_range is None:
            threshold_range = (similarities.min(), similarities.max())
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        
        results = {
            'thresholds': thresholds,
            'accuracies': [],
            'precisions': [],
            'recalls': [],
            'f1_scores': []
        }
        
        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            
            results['accuracies'].append(accuracy_score(labels, predictions))
            results['precisions'].append(precision_score(labels, predictions, zero_division=0))
            results['recalls'].append(recall_score(labels, predictions, zero_division=0))
            results['f1_scores'].append(f1_score(labels, predictions, zero_division=0))
        
        # Find best threshold for each metric
        best_acc_idx = np.argmax(results['accuracies'])
        best_f1_idx = np.argmax(results['f1_scores'])
        
        results['best_accuracy'] = {
            'threshold': thresholds[best_acc_idx],
            'value': results['accuracies'][best_acc_idx]
        }
        
        results['best_f1'] = {
            'threshold': thresholds[best_f1_idx],
            'value': results['f1_scores'][best_f1_idx]
        }
        
        return results
    
    def compute_embedding_statistics(self):
        """Compute statistics about the embeddings."""
        if len(self.embeddings1) == 0:
            return {}
        
        embeddings1 = np.array(self.embeddings1)
        embeddings2 = np.array(self.embeddings2)
        similarities = np.array(self.similarities)
        
        # Compute distances
        euclidean_distances = np.linalg.norm(embeddings1 - embeddings2, axis=1)
        
        # Separate positive and negative pairs
        labels = np.array(self.labels)
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        stats = {
            'total_pairs': len(labels),
            'positive_pairs': pos_mask.sum(),
            'negative_pairs': neg_mask.sum(),
            'similarity_stats': {
                'mean': similarities.mean(),
                'std': similarities.std(),
                'min': similarities.min(),
                'max': similarities.max()
            },
            'distance_stats': {
                'mean': euclidean_distances.mean(),
                'std': euclidean_distances.std(),
                'min': euclidean_distances.min(),
                'max': euclidean_distances.max()
            }
        }
        
        # Positive pair statistics
        if pos_mask.sum() > 0:
            pos_similarities = similarities[pos_mask]
            pos_distances = euclidean_distances[pos_mask]
            
            stats['positive_similarity'] = {
                'mean': pos_similarities.mean(),
                'std': pos_similarities.std(),
                'min': pos_similarities.min(),
                'max': pos_similarities.max()
            }
            
            stats['positive_distance'] = {
                'mean': pos_distances.mean(),
                'std': pos_distances.std(),
                'min': pos_distances.min(),
                'max': pos_distances.max()
            }
        
        # Negative pair statistics
        if neg_mask.sum() > 0:
            neg_similarities = similarities[neg_mask]
            neg_distances = euclidean_distances[neg_mask]
            
            stats['negative_similarity'] = {
                'mean': neg_similarities.mean(),
                'std': neg_similarities.std(),
                'min': neg_similarities.min(),
                'max': neg_similarities.max()
            }
            
            stats['negative_distance'] = {
                'mean': neg_distances.mean(),
                'std': neg_distances.std(),
                'min': neg_distances.min(),
                'max': neg_distances.max()
            }
        
        return stats
    
    def compute_all_metrics(self):
        """Compute all available metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['basic'] = self.compute_basic_metrics()
        
        # ROC metrics
        metrics['roc'] = self.compute_roc_metrics()
        
        # Precision-recall metrics
        metrics['precision_recall'] = self.compute_precision_recall_metrics()
        
        # Confusion matrix
        metrics['confusion_matrix'] = self.compute_confusion_matrix()
        
        # Threshold analysis
        metrics['threshold_analysis'] = self.compute_threshold_analysis()
        
        # Embedding statistics
        metrics['embedding_stats'] = self.compute_embedding_statistics()
        
        return metrics
    
    def get_classification_report(self):
        """Get detailed classification report."""
        if len(self.labels) == 0:
            return ""
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        return classification_report(labels, predictions, target_names=['Different', 'Same'])
    
    def set_threshold(self, threshold):
        """Update the threshold and recompute predictions."""
        self.threshold = threshold
        
        if len(self.similarities) > 0:
            similarities = np.array(self.similarities)
            self.predictions = (similarities >= threshold).astype(int)

class VerificationEvaluator:
    """High-level evaluator for face verification models."""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained face verification model
            device (str): Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate_dataset(self, data_loader, threshold=0.5):
        """
        Evaluate model on entire dataset.
        
        Args:
            data_loader: DataLoader with test data
            threshold (float): Similarity threshold
            
        Returns:
            dict: Comprehensive evaluation results
        """
        metrics_calculator = FaceVerificationMetrics(threshold=threshold)
        
        logger.info("Starting evaluation...")
        
        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(data_loader):
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                # Get embeddings
                if hasattr(self.model, 'encoder'):
                    # Siamese network
                    embeddings1 = self.model.encoder(img1)
                    embeddings2 = self.model.encoder(img2)
                else:
                    # Direct encoder
                    embeddings1 = self.model(img1)
                    embeddings2 = self.model(img2)
                
                # Update metrics
                metrics_calculator.update(embeddings1, embeddings2, labels)
                
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        # Compute all metrics
        results = metrics_calculator.compute_all_metrics()
        results['classification_report'] = metrics_calculator.get_classification_report()
        
        logger.info("Evaluation completed!")
        
        return results
    
    def find_optimal_threshold(self, data_loader, metric='f1'):
        """
        Find optimal threshold based on specified metric.
        
        Args:
            data_loader: DataLoader with validation data
            metric (str): Metric to optimize ('accuracy', 'f1', 'precision', 'recall')
            
        Returns:
            float: Optimal threshold
        """
        # Evaluate with a range of thresholds
        metrics_calculator = FaceVerificationMetrics()
        
        with torch.no_grad():
            for img1, img2, labels in data_loader:
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                if hasattr(self.model, 'encoder'):
                    embeddings1 = self.model.encoder(img1)
                    embeddings2 = self.model.encoder(img2)
                else:
                    embeddings1 = self.model(img1)
                    embeddings2 = self.model(img2)
                
                metrics_calculator.update(embeddings1, embeddings2, labels)
        
        # Analyze thresholds
        threshold_analysis = metrics_calculator.compute_threshold_analysis()
        
        if metric == 'accuracy':
            return threshold_analysis['best_accuracy']['threshold']
        elif metric == 'f1':
            return threshold_analysis['best_f1']['threshold']
        else:
            # Find best threshold for specified metric
            metric_values = threshold_analysis[f'{metric}s']
            best_idx = np.argmax(metric_values)
            return threshold_analysis['thresholds'][best_idx]

def test_metrics():
    """Test metrics functionality with dummy data."""
    # Create dummy data
    np.random.seed(42)
    
    # Generate random embeddings
    n_samples = 1000
    embedding_dim = 128
    
    embeddings1 = np.random.randn(n_samples, embedding_dim)
    embeddings2 = np.random.randn(n_samples, embedding_dim)
    
    # Create labels (0: different, 1: same)
    labels = np.random.randint(0, 2, n_samples)
    
    # Make same-person pairs more similar
    for i in range(n_samples):
        if labels[i] == 1:
            # Make second embedding similar to first
            embeddings2[i] = embeddings1[i] + np.random.randn(embedding_dim) * 0.1
    
    # Test metrics calculator
    calculator = FaceVerificationMetrics()
    calculator.update(embeddings1, embeddings2, labels)
    
    # Compute all metrics
    metrics = calculator.compute_all_metrics()
    
    print("Metrics Test Results:")
    print(f"Basic metrics: {metrics['basic']}")
    print(f"ROC AUC: {metrics['roc']['roc_auc']:.4f}")
    print(f"Optimal threshold: {metrics['roc']['optimal_threshold']:.4f}")
    print(f"Best accuracy: {metrics['threshold_analysis']['best_accuracy']}")
    print(f"Best F1: {metrics['threshold_analysis']['best_f1']}")

if __name__ == "__main__":
    test_metrics()
