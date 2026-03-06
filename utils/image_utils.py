"""
Image Utilities for Face Verification
Contains helper functions for image processing, visualization, and manipulation.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class ImageVisualizer:
    """Utility class for visualizing images and results."""
    
    def __init__(self, figsize=(12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize (tuple): Figure size for matplotlib plots
        """
        self.figsize = figsize
        plt.style.use('default')
    
    def show_image_pair(self, img1, img2, label=None, similarity=None, title="Image Pair"):
        """
        Display a pair of images with optional label and similarity score.
        
        Args:
            img1: First image (PIL, numpy, or tensor)
            img2: Second image (PIL, numpy, or tensor)
            label (int): Label (0=different, 1=same)
            similarity (float): Similarity score
            title (str): Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Convert images to displayable format
        img1_display = self._prepare_image_for_display(img1)
        img2_display = self._prepare_image_for_display(img2)
        
        # Display images
        ax1.imshow(img1_display)
        ax1.set_title("Image 1")
        ax1.axis('off')
        
        ax2.imshow(img2_display)
        ax2.set_title("Image 2")
        ax2.axis('off')
        
        # Add title with label and similarity
        title_text = title
        if label is not None:
            title_text += f" - Label: {'Same' if label == 1 else 'Different'}"
        if similarity is not None:
            title_text += f" - Similarity: {similarity:.4f}"
        
        fig.suptitle(title_text)
        plt.tight_layout()
        plt.show()
    
    def show_triplet(self, anchor, positive, negative, title="Triplet"):
        """
        Display a triplet of images.
        
        Args:
            anchor: Anchor image
            positive: Positive image
            negative: Negative image
            title (str): Plot title
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert images to displayable format
        anchor_display = self._prepare_image_for_display(anchor)
        positive_display = self._prepare_image_for_display(positive)
        negative_display = self._prepare_image_for_display(negative)
        
        # Display images
        ax1.imshow(anchor_display)
        ax1.set_title("Anchor")
        ax1.axis('off')
        
        ax2.imshow(positive_display)
        ax2.set_title("Positive")
        ax2.axis('off')
        
        ax3.imshow(negative_display)
        ax3.set_title("Negative")
        ax3.axis('off')
        
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def show_image_grid(self, images, titles=None, cols=4, figsize=(15, 10)):
        """
        Display a grid of images.
        
        Args:
            images (list): List of images
            titles (list): List of titles for each image
            cols (int): Number of columns in the grid
            figsize (tuple): Figure size
        """
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, img in enumerate(images):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            img_display = self._prepare_image_for_display(img)
            ax.imshow(img_display)
            ax.axis('off')
            
            if titles and i < len(titles):
                ax.set_title(titles[i])
        
        # Hide empty subplots
        for i in range(n_images, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _prepare_image_for_display(self, img):
        """Convert image to displayable format."""
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy
            if img.dim() == 4:
                img = img.squeeze(0)
            
            # Denormalize if needed (assuming ImageNet normalization)
            if img.min() < 0:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
            
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).cpu().numpy()
        
        elif isinstance(img, np.ndarray):
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
        elif isinstance(img, Image.Image):
            img = np.array(img)
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
        
        return np.clip(img, 0, 1)

class FaceDetector:
    """Face detection utilities."""
    
    def __init__(self):
        """Initialize face detector."""
        try:
            # Load OpenCV's face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            self.detector_available = True
            logger.info("Face detector initialized successfully")
        except Exception as e:
            logger.warning(f"Face detector not available: {e}")
            self.detector_available = False
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image.
        
        Args:
            image: Input image (PIL, numpy, or tensor)
            scale_factor (float): Scale factor for detection
            min_neighbors (int): Minimum neighbors for detection
            min_size (tuple): Minimum face size
            
        Returns:
            list: List of face bounding boxes [(x, y, w, h), ...]
        """
        if not self.detector_available:
            logger.warning("Face detector not available")
            return []
        
        # Convert image to numpy array
        if isinstance(image, torch.Tensor):
            img = self._tensor_to_numpy(image)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size
        )
        
        return faces.tolist()
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image: Input image
            faces (list): List of face bounding boxes
            color (tuple): Box color (B, G, R)
            thickness (int): Box thickness
            
        Returns:
            numpy.array: Image with drawn faces
        """
        # Convert image to numpy array
        if isinstance(image, torch.Tensor):
            img = self._tensor_to_numpy(image)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Draw faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        return img
    
    def crop_faces(self, image, faces, padding=20):
        """
        Crop faces from image.
        
        Args:
            image: Input image
            faces (list): List of face bounding boxes
            padding (int): Padding around faces
            
        Returns:
            list: List of cropped face images
        """
        # Convert image to numpy array
        if isinstance(image, torch.Tensor):
            img = self._tensor_to_numpy(image)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        cropped_faces = []
        
        for (x, y, w, h) in faces:
            # Add padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            
            # Crop face
            face = img[y1:y2, x1:x2]
            cropped_faces.append(face)
        
        return cropped_faces
    
    def _tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize if needed
        if tensor.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean
        
        tensor = torch.clamp(tensor, 0, 1)
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        
        return (tensor * 255).astype(np.uint8)

class ImageAugmentation:
    """Image augmentation utilities for training."""
    
    def __init__(self):
        """Initialize augmentation transforms."""
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def augment_image(self, image):
        """
        Apply augmentation to an image.
        
        Args:
            image: Input image (PIL Image)
            
        Returns:
            torch.Tensor: Augmented image tensor
        """
        return self.train_transform(image)
    
    def preprocess_image(self, image):
        """
        Preprocess image without augmentation.
        
        Args:
            image: Input image (PIL Image)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        return self.val_transform(image)

def save_image_comparison(img1, img2, label, similarity, output_path):
    """
    Save a comparison of two images with labels and similarity.
    
    Args:
        img1: First image
        img2: Second image
        label (int): Label (0=different, 1=same)
        similarity (float): Similarity score
        output_path (str): Output file path
    """
    visualizer = ImageVisualizer()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convert images to displayable format
    img1_display = visualizer._prepare_image_for_display(img1)
    img2_display = visualizer._prepare_image_for_display(img2)
    
    # Display images
    ax1.imshow(img1_display)
    ax1.set_title("Image 1")
    ax1.axis('off')
    
    ax2.imshow(img2_display)
    ax2.set_title("Image 2")
    ax2.axis('off')
    
    # Add title
    label_text = 'Same' if label == 1 else 'Different'
    fig.suptitle(f"Label: {label_text} | Similarity: {similarity:.4f}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def compute_embedding_similarity(embedding1, embedding2, metric='cosine'):
    """
    Compute similarity between two embeddings.
    
    Args:
        embedding1 (torch.Tensor): First embedding
        embedding2 (torch.Tensor): Second embedding
        metric (str): Similarity metric ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        float: Similarity score
    """
    if metric == 'cosine':
        similarity = torch.cosine_similarity(embedding1, embedding2, dim=1)
    elif metric == 'euclidean':
        distance = torch.pairwise_distance(embedding1, embedding2, p=2)
        similarity = 1 / (1 + distance)  # Convert distance to similarity
    elif metric == 'manhattan':
        distance = torch.pairwise_distance(embedding1, embedding2, p=1)
        similarity = 1 / (1 + distance)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    
    return similarity.item() if similarity.numel() == 1 else similarity

def create_similarity_histogram(similarities, labels, bins=50, title="Similarity Distribution"):
    """
    Create histogram of similarity scores.
    
    Args:
        similarities (list): List of similarity scores
        labels (list): List of corresponding labels
        bins (int): Number of bins for histogram
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Separate positive and negative pairs
    pos_similarities = [sim for sim, label in zip(similarities, labels) if label == 1]
    neg_similarities = [sim for sim, label in zip(similarities, labels) if label == 0]
    
    # Plot histograms
    plt.hist(neg_similarities, bins=bins, alpha=0.7, label='Different Persons', color='red')
    plt.hist(pos_similarities, bins=bins, alpha=0.7, label='Same Person', color='green')
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def test_image_utils():
    """Test image utilities with dummy data."""
    # Create dummy images
    dummy_img1 = torch.randn(3, 160, 160)
    dummy_img2 = torch.randn(3, 160, 160)
    
    # Test visualizer
    visualizer = ImageVisualizer()
    visualizer.show_image_pair(dummy_img1, dummy_img2, label=1, similarity=0.85)
    
    # Test similarity computation
    similarity = compute_embedding_similarity(
        dummy_img1.unsqueeze(0), 
        dummy_img2.unsqueeze(0), 
        metric='cosine'
    )
    print(f"Cosine similarity: {similarity:.4f}")
    
    # Test face detector
    detector = FaceDetector()
    if detector.detector_available:
        print("Face detector is available")
    else:
        print("Face detector is not available")

if __name__ == "__main__":
    test_image_utils()
