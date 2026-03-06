"""
Image Preprocessing Utilities for Face Verification
Handles image normalization, resizing, and tensor conversion.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class FaceImagePreprocessor:
    """Preprocessor for face images in verification tasks."""
    
    def __init__(self, image_size=(160, 160), normalize_mean=[0.485, 0.456, 0.406], 
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Initialize the preprocessor.
        
        Args:
            image_size (tuple): Target image size (height, width)
            normalize_mean (list): Mean values for normalization
            normalize_std (list): Standard deviation values for normalization
        """
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Define transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
        
        # Transform for validation/testing (no augmentation)
        self.val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess a single image.
        
        Args:
            image: PIL Image, numpy array, or tensor
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # RGB numpy array
                    image = Image.fromarray(image.astype(np.uint8))
                elif len(image.shape) == 2:
                    # Grayscale numpy array
                    image = Image.fromarray(image.astype(np.uint8), mode='L')
                    image = image.convert('RGB')
                else:
                    raise ValueError(f"Unsupported numpy array shape: {image.shape}")
            
            # Convert tensor to PIL Image if needed
            elif isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = transforms.ToPILImage()(image)
                else:
                    raise ValueError(f"Unsupported tensor shape: {image.shape}")
            
            # Ensure PIL Image is RGB
            elif isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Apply transformations
            processed = self.transform(image)
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def preprocess_batch(self, images):
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images (PIL, numpy, or tensor)
            
        Returns:
            torch.Tensor: Batch of preprocessed images
        """
        try:
            processed_images = []
            for img in images:
                processed_img = self.preprocess_image(img)
                processed_images.append(processed_img)
            
            # Stack into batch
            batch = torch.stack(processed_images, dim=0)
            return batch
            
        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            raise
    
    def denormalize(self, tensor):
        """
        Denormalize a tensor for visualization.
        
        Args:
            tensor (torch.Tensor): Normalized tensor
            
        Returns:
            torch.Tensor: Denormalized tensor
        """
        mean = torch.tensor(self.normalize_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.normalize_std).view(1, 3, 1, 1)
        
        if tensor.device != mean.device:
            mean = mean.to(tensor.device)
            std = std.to(tensor.device)
        
        denormalized = tensor * std + mean
        return torch.clamp(denormalized, 0, 1)
    
    def tensor_to_pil(self, tensor):
        """
        Convert tensor to PIL Image for visualization.
        
        Args:
            tensor (torch.Tensor): Image tensor
            
        Returns:
            PIL.Image: PIL Image
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Denormalize if needed
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = self.denormalize(tensor.unsqueeze(0)).squeeze(0)
        
        # Convert to PIL
        tensor = torch.clamp(tensor, 0, 1)
        return transforms.ToPILImage()(tensor)

class FaceDetector:
    """Simple face detection for preprocessing."""
    
    def __init__(self):
        """Initialize face detector."""
        try:
            # Try to load OpenCV's face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detector_available = True
        except Exception as e:
            logger.warning(f"Face detector not available: {e}")
            self.detector_available = False
    
    def detect_face(self, image):
        """
        Detect and crop face from image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy.array: Cropped face image or original if no face detected
        """
        if not self.detector_available:
            logger.warning("Face detector not available, returning original image")
            if isinstance(image, Image.Image):
                return np.array(image)
            return image
        
        try:
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the first detected face
                x, y, w, h = faces[0]
                
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image_array.shape[1] - x, w + 2 * padding)
                h = min(image_array.shape[0] - y, h + 2 * padding)
                
                # Crop face
                face_crop = image_array[y:y+h, x:x+w]
                logger.info(f"Face detected and cropped: {face_crop.shape}")
                return face_crop
            else:
                logger.warning("No face detected, returning original image")
                return image_array
                
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            if isinstance(image, Image.Image):
                return np.array(image)
            return image

def remove_corrupted_images(image_paths):
    """
    Remove corrupted images from a list of image paths.
    
    Args:
        image_paths (list): List of image file paths
        
    Returns:
        list: List of valid image paths
    """
    valid_paths = []
    
    for img_path in image_paths:
        try:
            # Try to open the image
            with Image.open(img_path) as img:
                img.verify()
            
            # Try to load the image
            with Image.open(img_path) as img:
                img.load()
            
            valid_paths.append(img_path)
            
        except Exception as e:
            logger.warning(f"Corrupted image removed: {img_path}, Error: {e}")
    
    logger.info(f"Removed {len(image_paths) - len(valid_paths)} corrupted images")
    return valid_paths

def create_data_loaders(preprocessor, batch_size=32, num_workers=4):
    """
    Create data loaders with preprocessing.
    
    Args:
        preprocessor (FaceImagePreprocessor): Preprocessor instance
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        
    Returns:
        dict: Dictionary with train and val loaders
    """
    # This will be implemented when we create the dataset loader
    pass

# Utility functions for common preprocessing tasks
def resize_image(image, target_size):
    """Resize image to target size."""
    if isinstance(image, np.ndarray):
        return cv2.resize(image, target_size)
    elif isinstance(image, Image.Image):
        return image.resize(target_size[::-1])  # PIL uses (width, height)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def normalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize image with given mean and std."""
    if isinstance(image, np.ndarray):
        image = image.astype(np.float32) / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    elif isinstance(image, torch.Tensor):
        normalize = transforms.Normalize(mean=mean, std=std)
        return normalize(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
