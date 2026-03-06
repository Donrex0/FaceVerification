"""
Dataset Loader for Face Verification
Handles loading and batching of face image pairs for Siamese networks.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from PIL import Image
import logging
from .preprocessing import FaceImagePreprocessor

logger = logging.getLogger(__name__)

class LFWPairsDataset(Dataset):
    """Dataset class for LFW face pairs."""
    
    def __init__(self, data_file, split='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_file (str): Path to the pickle file containing LFW pairs
            split (str): 'train' or 'test'
            transform: Transform to apply to images
        """
        self.data_file = data_file
        self.split = split
        self.transform = transform
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load LFW pairs data."""
        try:
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            
            # Handle both original LFW format and our simplified format
            if isinstance(data, dict) and self.split in data and isinstance(data[self.split], dict):
                # Original LFW format
                self.pairs = data[self.split]['pairs']
                self.targets = data[self.split]['targets']
                self.raw_data = data[self.split]['data']
            elif isinstance(data, dict) and self.split in data:
                # Our simplified format - data[split] is a list of tuples
                pairs_data = data[self.split]
                self.pairs = [(pair[0], pair[1]) for pair in pairs_data]
                self.targets = [pair[2] for pair in pairs_data]
                self.raw_data = None
            else:
                raise ValueError(f"Split '{self.split}' not found in data or invalid format")
            
            logger.info(f"Loaded {len(self.pairs)} {self.split} pairs")
            logger.info(f"Positive pairs: {sum(self.targets)}")
            logger.info(f"Negative pairs: {len(self.targets) - sum(self.targets)}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def __len__(self):
        """Return the number of pairs."""
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a pair of images and their label.
        
        Args:
            idx (int): Index of the pair
            
        Returns:
            tuple: (image1, image2, label)
        """
        try:
            # Get pair data
            pair_data = self.pairs[idx]
            label = self.targets[idx]
            
            # Handle both file path format and raw data format
            if isinstance(pair_data[0], str) and isinstance(pair_data[1], str):
                # Our simplified format - file paths
                img1_path = pair_data[0]
                img2_path = pair_data[1]
                
                # Load images from file paths
                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')
            else:
                # Original LFW format - raw data indices
                img1_data = self._extract_image_data(pair_data[0])
                img2_data = self._extract_image_data(pair_data[1])
                
                # Convert to PIL Images
                img1 = self._array_to_pil(img1_data)
                img2 = self._array_to_pil(img2_data)
            
            # Apply transforms if provided
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            return img1, img2, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            raise
    
    def _extract_image_data(self, pair_info):
        """Extract image data from pair information."""
        # LFW pairs data structure: (image_index, person_name)
        image_idx = pair_info[0]
        
        # Get the image from raw data
        if hasattr(self.raw_data, 'shape'):
            # If raw_data is a numpy array
            if len(self.raw_data.shape) == 4:
                # Shape: (n_pairs, 2, height, width)
                return self.raw_data[image_idx]
            else:
                # Handle other formats
                return self.raw_data[image_idx]
        else:
            # If raw_data is a list or other format
            return self.raw_data[image_idx]
    
    def _array_to_pil(self, img_array):
        """Convert numpy array to PIL Image."""
        if isinstance(img_array, np.ndarray):
            if len(img_array.shape) == 2:
                # Grayscale
                return Image.fromarray(img_array, mode='L').convert('RGB')
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 3:
                    # RGB
                    return Image.fromarray(img_array.astype(np.uint8))
                elif img_array.shape[2] == 1:
                    # Single channel
                    return Image.fromarray(img_array.squeeze().astype(np.uint8), mode='L').convert('RGB')
        
        raise ValueError(f"Unsupported image array shape: {img_array.shape}")

class SiameseDataset(Dataset):
    """Dataset class for Siamese network training."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize Siamese dataset.
        
        Args:
            image_paths (list): List of pairs of image paths [(img1_path, img2_path), ...]
            labels (list): List of labels (0 or 1)
            transform: Transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        assert len(image_paths) == len(labels), "Number of image pairs must match number of labels"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class TripletDataset(Dataset):
    """Dataset class for triplet loss training."""
    
    def __init__(self, image_paths, person_ids, transform=None):
        """
        Initialize triplet dataset.
        
        Args:
            image_paths (list): List of image paths
            person_ids (list): List of corresponding person IDs
            transform: Transform to apply to images
        """
        self.image_paths = image_paths
        self.person_ids = person_ids
        self.transform = transform
        
        # Group images by person
        self.person_to_images = {}
        for img_path, person_id in zip(image_paths, person_ids):
            if person_id not in self.person_to_images:
                self.person_to_images[person_id] = []
            self.person_to_images[person_id].append(img_path)
        
        self.person_ids_list = list(self.person_to_images.keys())
        
        logger.info(f"Created triplet dataset with {len(self.person_ids_list)} people")
        logger.info(f"Total images: {len(image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Anchor
        anchor_path = self.image_paths[idx]
        anchor_person = self.person_ids[idx]
        
        # Positive (same person as anchor)
        positive_paths = [p for p in self.person_to_images[anchor_person] if p != anchor_path]
        if not positive_paths:
            # If no other image for this person, use anchor as positive
            positive_path = anchor_path
        else:
            positive_path = np.random.choice(positive_paths)
        
        # Negative (different person)
        negative_person = np.random.choice([p for p in self.person_ids_list if p != anchor_person])
        negative_path = np.random.choice(self.person_to_images[negative_person])
        
        # Load images
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative

def create_data_loaders(data_dir, batch_size=32, num_workers=4, image_size=(160, 160)):
    """
    Create train and test data loaders.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        image_size (tuple): Target image size
        
    Returns:
        dict: Dictionary with train and test loaders
    """
    # Initialize preprocessor
    preprocessor = FaceImagePreprocessor(image_size=image_size)
    
    # Data file path
    data_file = os.path.join(data_dir, "lfw_pairs.pkl")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Create datasets
    train_dataset = LFWPairsDataset(
        data_file=data_file,
        split='train',
        transform=preprocessor.transform
    )
    
    test_dataset = LFWPairsDataset(
        data_file=data_file,
        split='test',
        transform=preprocessor.val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'test': test_loader,
        'preprocessor': preprocessor
    }

def create_triplet_loader(data_dir, batch_size=32, num_workers=4, image_size=(160, 160)):
    """
    Create triplet data loader for improved model training.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        image_size (tuple): Target image size
        
    Returns:
        DataLoader: Triplet data loader
    """
    # Load metadata to get individual images
    metadata_file = os.path.join(data_dir, "metadata.pkl")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    # Create image paths and person IDs
    image_paths = []
    person_ids = []
    
    # For LFW, we'll create synthetic paths since the data is already loaded
    for i, target in enumerate(metadata['targets']):
        image_paths.append(f"lfw_image_{i}")  # Synthetic path
        person_ids.append(target)
    
    # Initialize preprocessor
    preprocessor = FaceImagePreprocessor(image_size=image_size)
    
    # Create triplet dataset
    triplet_dataset = TripletDataset(
        image_paths=image_paths,
        person_ids=person_ids,
        transform=preprocessor.transform
    )
    
    # Create data loader
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return triplet_loader

def get_dataset_statistics(data_loader):
    """
    Get statistics about the dataset.
    
    Args:
        data_loader (DataLoader): Data loader
        
    Returns:
        dict: Dataset statistics
    """
    total_pairs = 0
    positive_pairs = 0
    negative_pairs = 0
    
    for _, _, labels in data_loader:
        total_pairs += len(labels)
        positive_pairs += (labels == 1).sum().item()
        negative_pairs += (labels == 0).sum().item()
    
    return {
        'total_pairs': total_pairs,
        'positive_pairs': positive_pairs,
        'negative_pairs': negative_pairs,
        'positive_ratio': positive_pairs / total_pairs if total_pairs > 0 else 0
    }
