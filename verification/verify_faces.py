"""
Face Verification Demo Script
Demonstrates real-time face verification using trained models.
"""

import os
import sys
import torch
import argparse
from PIL import Image
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.preprocessing import FaceImagePreprocessor
from models.baseline_cnn import create_baseline_model
from models.improved_model import create_improved_model
from models.siamese_network import create_siamese_network
from utils.config import get_config
from utils.image_utils import ImageVisualizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceVerifier:
    """Face verification system for real-time inference."""
    
    def __init__(self, model_path=None, model_type='improved', config=None):
        """
        Initialize the face verifier.
        
        Args:
            model_path (str): Path to trained model checkpoint
            model_type (str): Type of model ('baseline' or 'improved')
            config: Configuration object
        """
        self.config = config if config else get_config()
        self.device = self.config.device
        self.model_type = model_type
        
        # Initialize preprocessor
        self.preprocessor = FaceImagePreprocessor(
            image_size=self.config.data.image_size,
            normalize_mean=[0.485, 0.456, 0.406],
            normalize_std=[0.229, 0.224, 0.225]
        )
        
        # Load model
        self.model = self.load_model(model_path, model_type)
        
        # Initialize visualizer
        self.visualizer = ImageVisualizer()
        
        logger.info(f"Face verifier initialized with {model_type} model")
    
    def load_model(self, model_path, model_type):
        """
        Load trained model.
        
        Args:
            model_path (str): Path to model checkpoint
            model_type (str): Type of model
            
        Returns:
            Trained model
        """
        if model_path is None:
            if model_type == 'baseline':
                model_path = os.path.join(self.config.training.checkpoint_dir, "best_baseline_model.pth")
            else:
                model_path = os.path.join(self.config.training.checkpoint_dir, "best_improved_model.pth")
        
        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint not found: {model_path}")
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Create model architecture
        if model_type == 'baseline':
            encoder = create_baseline_model(
                model_type=self.config.model.baseline_model_type,
                embedding_dim=self.config.model.baseline_embedding_dim
            )
            siamese_net = create_siamese_network(
                encoder=encoder,
                loss_type=self.config.model.loss_type
            )
        else:
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
        checkpoint = torch.load(model_path, map_location=self.device)
        siamese_net.load_state_dict(checkpoint['model_state_dict'])
        siamese_net.to(self.device)
        siamese_net.eval()
        
        logger.info(f"Model loaded from {model_path}")
        return siamese_net
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for verification.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Preprocess
            tensor = self.preprocessor.preprocess_image(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0).to(self.device)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def verify_faces(self, image1_path, image2_path, threshold=0.5):
        """
        Verify if two faces belong to the same person.
        
        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
            threshold (float): Similarity threshold
            
        Returns:
            dict: Verification results
        """
        try:
            # Preprocess images
            img1_tensor = self.preprocess_image(image1_path)
            img2_tensor = self.preprocess_image(image2_path)
            
            # Get embeddings and similarity
            with torch.no_grad():
                similarity, prediction = self.model.predict_similarity(img1_tensor, img2_tensor, threshold)
            
            # Convert to numpy
            similarity_score = similarity.cpu().item()
            prediction_label = prediction.cpu().item()
            
            # Determine confidence level
            if similarity_score >= 0.8:
                confidence = "HIGH"
            elif similarity_score >= 0.6:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            # Determine result
            result = "SAME PERSON" if prediction_label == 1 else "DIFFERENT PERSONS"
            
            return {
                'similarity_score': similarity_score,
                'prediction': prediction_label,
                'result': result,
                'confidence': confidence,
                'threshold_used': threshold,
                'image1_path': image1_path,
                'image2_path': image2_path
            }
            
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            raise
    
    def verify_and_visualize(self, image1_path, image2_path, threshold=0.5, save_result=False):
        """
        Verify faces and visualize the result.
        
        Args:
            image1_path (str): Path to first image
            image2_path (str): Path to second image
            threshold (float): Similarity threshold
            save_result (bool): Whether to save the comparison image
            
        Returns:
            dict: Verification results
        """
        # Perform verification
        results = self.verify_faces(image1_path, image2_path, threshold)
        
        # Load images for visualization
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)
        
        # Display comparison
        self.visualizer.show_image_pair(
            img1, img2, 
            label=results['prediction'], 
            similarity=results['similarity_score'],
            title=f"Face Verification - {results['result']}"
        )
        
        # Save result if requested
        if save_result:
            output_path = f"verification_result_{os.path.basename(image1_path)}_{os.path.basename(image2_path)}.png"
            self.visualizer.show_image_pair(
                img1, img2,
                label=results['prediction'],
                similarity=results['similarity_score'],
                title=f"Face Verification - {results['result']}"
            )
            # Note: In a real implementation, you'd save the figure here
            logger.info(f"Result visualization would be saved to {output_path}")
        
        return results
    
    def batch_verify(self, image_pairs, threshold=0.5):
        """
        Verify multiple image pairs.
        
        Args:
            image_pairs (list): List of tuples (image1_path, image2_path)
            threshold (float): Similarity threshold
            
        Returns:
            list: List of verification results
        """
        results = []
        
        logger.info(f"Processing {len(image_pairs)} image pairs...")
        
        for i, (img1_path, img2_path) in enumerate(image_pairs):
            try:
                result = self.verify_faces(img1_path, img2_path, threshold)
                result['pair_index'] = i
                results.append(result)
                
                logger.info(f"Pair {i+1}: {result['result']} (similarity: {result['similarity_score']:.4f})")
                
            except Exception as e:
                logger.error(f"Error processing pair {i+1}: {e}")
                results.append({
                    'pair_index': i,
                    'error': str(e),
                    'image1_path': img1_path,
                    'image2_path': img2_path
                })
        
        return results
    
    def find_optimal_threshold(self, validation_pairs, validation_labels):
        """
        Find optimal threshold for verification.
        
        Args:
            validation_pairs (list): List of image pairs for validation
            validation_labels (list): Ground truth labels
            
        Returns:
            float: Optimal threshold
        """
        logger.info("Finding optimal threshold...")
        
        similarities = []
        
        for img1_path, img2_path in validation_pairs:
            result = self.verify_faces(img1_path, img2_path, threshold=0.0)  # Use 0.0 to get raw similarity
            similarities.append(result['similarity_score'])
        
        # Find optimal threshold using Youden's J statistic
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(validation_labels, similarities)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        return optimal_threshold

def main():
    """Main function for face verification demo."""
    parser = argparse.ArgumentParser(description='Face Verification Demo')
    parser.add_argument('image1', help='Path to first image')
    parser.add_argument('image2', help='Path to second image')
    parser.add_argument('--model', choices=['baseline', 'improved'], default='improved',
                       help='Model type to use')
    parser.add_argument('--model-path', help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold')
    parser.add_argument('--visualize', action='store_true',
                       help='Show visualization')
    parser.add_argument('--save', action='store_true',
                       help='Save result visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize verifier
        verifier = FaceVerifier(model_path=args.model_path, model_type=args.model)
        
        # Perform verification
        if args.visualize:
            results = verifier.verify_and_visualize(
                args.image1, args.image2, 
                threshold=args.threshold, 
                save_result=args.save
            )
        else:
            results = verifier.verify_faces(args.image1, args.image2, threshold=args.threshold)
        
        # Print results
        print("\n" + "="*60)
        print("FACE VERIFICATION RESULTS")
        print("="*60)
        print(f"Image 1: {results['image1_path']}")
        print(f"Image 2: {results['image2_path']}")
        print(f"Similarity Score: {results['similarity_score']:.4f}")
        print(f"Prediction: {results['result']}")
        print(f"Confidence: {results['confidence']}")
        print(f"Threshold Used: {results['threshold_used']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

def demo_batch_verification():
    """Demonstrate batch verification functionality."""
    # This would be used for testing multiple pairs
    verifier = FaceVerifier(model_type='improved')
    
    # Example pairs (in practice, these would be real image paths)
    image_pairs = [
        ("path/to/image1a.jpg", "path/to/image1b.jpg"),
        ("path/to/image2a.jpg", "path/to/image2b.jpg"),
    ]
    
    results = verifier.batch_verify(image_pairs)
    
    for result in results:
        if 'error' not in result:
            print(f"Pair {result['pair_index']}: {result['result']} "
                  f"(similarity: {result['similarity_score']:.4f})")
        else:
            print(f"Pair {result['pair_index']}: Error - {result['error']}")

if __name__ == "__main__":
    exit(main())
