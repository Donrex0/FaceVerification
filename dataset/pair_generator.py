"""
Face Pair Generator for Verification Tasks
Generates pairs of face images with labels for training and evaluation.
"""

import numpy as np
import random
import os
import pickle
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class FacePairGenerator:
    """Generates face pairs for verification tasks."""
    
    def __init__(self, seed=42):
        """
        Initialize the pair generator.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_pairs_from_lfw(self, lfw_data, num_positive_pairs=None, num_negative_pairs=None):
        """
        Generate pairs from LFW dataset structure.
        
        Args:
            lfw_data (dict): LFW dataset data
            num_positive_pairs (int): Number of positive pairs to generate
            num_negative_pairs (int): Number of negative pairs to generate
            
        Returns:
            tuple: (pairs, labels)
        """
        # Extract images and targets from LFW data
        if isinstance(lfw_data, dict) and 'train' in lfw_data:
            # Use training data for pair generation
            targets = lfw_data['train']['targets']
            pairs_info = lfw_data['train']['pairs']
        else:
            # Assume data is in format: (images, targets)
            targets = lfw_data[1] if isinstance(lfw_data, (list, tuple)) else None
            pairs_info = None
        
        # Group images by person
        person_to_images = defaultdict(list)
        
        if pairs_info is not None:
            # Use pairs_info if available
            for i, (pair, target) in enumerate(zip(pairs_info, targets)):
                if target == 1:  # Positive pair
                    person_id = f"person_{pair[0][1]}"  # Use person name
                    if person_id not in person_to_images:
                        person_to_images[person_id] = []
                    person_to_images[person_id].append(i)
        else:
            # Group by targets if no pairs_info
            for i, target in enumerate(targets):
                person_id = f"person_{target}"
                person_to_images[person_id].append(i)
        
        # Generate pairs
        positive_pairs = []
        negative_pairs = []
        
        # Generate positive pairs (same person)
        for person_id, image_indices in person_to_images.items():
            if len(image_indices) >= 2:
                # Generate all possible pairs for this person
                for i in range(len(image_indices)):
                    for j in range(i + 1, len(image_indices)):
                        if pairs_info is not None:
                            # Use actual pair data
                            pair_data = (pairs_info[image_indices[i]], pairs_info[image_indices[j]])
                        else:
                            # Use indices
                            pair_data = (image_indices[i], image_indices[j])
                        
                        positive_pairs.append((pair_data, 1))
        
        # Generate negative pairs (different people)
        person_ids = list(person_to_images.keys())
        
        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                person1 = person_ids[i]
                person2 = person_ids[j]
                
                images1 = person_to_images[person1]
                images2 = person_to_images[person2]
                
                # Generate pairs between these two people
                for img1 in images1:
                    for img2 in images2:
                        if pairs_info is not None:
                            pair_data = (pairs_info[img1], pairs_info[img2])
                        else:
                            pair_data = (img1, img2)
                        
                        negative_pairs.append((pair_data, 0))
        
        # Sample pairs if limits are specified
        if num_positive_pairs and len(positive_pairs) > num_positive_pairs:
            positive_pairs = random.sample(positive_pairs, num_positive_pairs)
        
        if num_negative_pairs and len(negative_pairs) > num_negative_pairs:
            negative_pairs = random.sample(negative_pairs, num_negative_pairs)
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Separate pairs and labels
        pairs = [pair[0] for pair in all_pairs]
        labels = [pair[1] for pair in all_pairs]
        
        logger.info(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        
        return pairs, labels
    
    def generate_pairs_from_directory(self, image_dir, num_pairs_per_person=10, 
                                    num_negative_pairs_multiplier=2):
        """
        Generate pairs from directory structure where each subfolder is a person.
        
        Args:
            image_dir (str): Directory containing person subfolders
            num_pairs_per_person (int): Number of positive pairs per person
            num_negative_pairs_multiplier (int): Multiplier for negative pairs
            
        Returns:
            tuple: (pairs, labels)
        """
        # Scan directory structure
        person_to_images = defaultdict(list)
        
        for person_name in os.listdir(image_dir):
            person_dir = os.path.join(image_dir, person_name)
            if os.path.isdir(person_dir):
                for img_file in os.listdir(person_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(person_dir, img_file)
                        person_to_images[person_name].append(img_path)
        
        logger.info(f"Found {len(person_to_images)} people")
        total_images = sum(len(imgs) for imgs in person_to_images.values())
        logger.info(f"Total images: {total_images}")
        
        # Generate positive pairs
        positive_pairs = []
        for person_name, images in person_to_images.items():
            if len(images) >= 2:
                # Generate random pairs
                for _ in range(min(num_pairs_per_person, len(images) * (len(images) - 1) // 2)):
                    img1, img2 = random.sample(images, 2)
                    positive_pairs.append(((img1, img2), 1))
        
        # Generate negative pairs
        negative_pairs = []
        person_names = list(person_to_images.keys())
        
        for _ in range(len(positive_pairs) * num_negative_pairs_multiplier):
            # Select two different people
            person1, person2 = random.sample(person_names, 2)
            
            # Select random images from each person
            img1 = random.choice(person_to_images[person1])
            img2 = random.choice(person_to_images[person2])
            
            negative_pairs.append(((img1, img2), 0))
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Separate pairs and labels
        pairs = [pair[0] for pair in all_pairs]
        labels = [pair[1] for pair in all_pairs]
        
        logger.info(f"Generated {len(positive_pairs)} positive pairs and {len(negative_pairs)} negative pairs")
        
        return pairs, labels
    
    def generate_triplets(self, pairs, labels, num_triplets=None):
        """
        Generate triplets (anchor, positive, negative) from pairs.
        
        Args:
            pairs (list): List of image pairs
            labels (list): List of labels
            num_triplets (int): Number of triplets to generate
            
        Returns:
            list: List of triplets
        """
        # Separate positive and negative pairs
        positive_pairs = [pair for pair, label in zip(pairs, labels) if label == 1]
        negative_pairs = [pair for pair, label in zip(pairs, labels) if label == 0]
        
        if len(positive_pairs) == 0 or len(negative_pairs) == 0:
            raise ValueError("Need both positive and negative pairs to generate triplets")
        
        triplets = []
        
        # Generate triplets
        for _ in range(num_triplets or len(positive_pairs)):
            # Select anchor-positive pair
            anchor, positive = random.choice(positive_pairs)
            
            # Select negative example
            negative_pair = random.choice(negative_pairs)
            negative = negative_pair[0]  # Use first image of negative pair
            
            triplets.append((anchor, positive, negative))
        
        logger.info(f"Generated {len(triplets)} triplets")
        return triplets
    
    def balance_dataset(self, pairs, labels):
        """
        Balance the dataset by undersampling the majority class.
        
        Args:
            pairs (list): List of image pairs
            labels (list): List of labels
            
        Returns:
            tuple: (balanced_pairs, balanced_labels)
        """
        # Separate positive and negative pairs
        positive_indices = [i for i, label in enumerate(labels) if label == 1]
        negative_indices = [i for i, label in enumerate(labels) if label == 0]
        
        # Find minority class size
        min_size = min(len(positive_indices), len(negative_indices))
        
        # Sample from majority class
        if len(positive_indices) > min_size:
            positive_indices = random.sample(positive_indices, min_size)
        if len(negative_indices) > min_size:
            negative_indices = random.sample(negative_indices, min_size)
        
        # Combine balanced indices
        balanced_indices = positive_indices + negative_indices
        random.shuffle(balanced_indices)
        
        # Create balanced dataset
        balanced_pairs = [pairs[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        logger.info(f"Balanced dataset: {len(positive_indices)} positive, {len(negative_indices)} negative")
        
        return balanced_pairs, balanced_labels
    
    def save_pairs(self, pairs, labels, output_file):
        """
        Save pairs and labels to file.
        
        Args:
            pairs (list): List of image pairs
            labels (list): List of labels
            output_file (str): Output file path
        """
        data = {
            'pairs': pairs,
            'labels': labels,
            'num_pairs': len(pairs),
            'num_positive': sum(labels),
            'num_negative': len(labels) - sum(labels)
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {len(pairs)} pairs to {output_file}")
    
    def load_pairs(self, input_file):
        """
        Load pairs and labels from file.
        
        Args:
            input_file (str): Input file path
            
        Returns:
            tuple: (pairs, labels)
        """
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded {data['num_pairs']} pairs from {input_file}")
        logger.info(f"Positive: {data['num_positive']}, Negative: {data['num_negative']}")
        
        return data['pairs'], data['labels']
    
    def get_pair_statistics(self, pairs, labels):
        """
        Get statistics about the pairs.
        
        Args:
            pairs (list): List of image pairs
            labels (list): List of labels
            
        Returns:
            dict: Statistics
        """
        total_pairs = len(pairs)
        positive_pairs = sum(labels)
        negative_pairs = total_pairs - positive_pairs
        
        return {
            'total_pairs': total_pairs,
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs,
            'positive_ratio': positive_pairs / total_pairs if total_pairs > 0 else 0,
            'negative_ratio': negative_pairs / total_pairs if total_pairs > 0 else 0
        }

def generate_lfw_pairs_for_training(data_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Generate and save LFW pairs for training and testing.
    
    Args:
        data_dir (str): Directory containing LFW data
        output_dir (str): Directory to save generated pairs
        train_ratio (float): Ratio of pairs for training
        seed (int): Random seed
    """
    # Initialize generator
    generator = FacePairGenerator(seed=seed)
    
    # Load LFW data
    data_file = os.path.join(data_dir, "lfw_pairs.pkl")
    with open(data_file, 'rb') as f:
        lfw_data = pickle.load(f)
    
    # Generate pairs
    pairs, labels = generator.generate_pairs_from_lfw(lfw_data)
    
    # Balance dataset
    pairs, labels = generator.balance_dataset(pairs, labels)
    
    # Split into train and test
    total_pairs = len(pairs)
    train_size = int(total_pairs * train_ratio)
    
    indices = list(range(total_pairs))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_pairs = [pairs[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_pairs = [pairs[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # Save pairs
    os.makedirs(output_dir, exist_ok=True)
    
    generator.save_pairs(train_pairs, train_labels, os.path.join(output_dir, "train_pairs.pkl"))
    generator.save_pairs(test_pairs, test_labels, os.path.join(output_dir, "test_pairs.pkl"))
    
    # Generate and save triplets for improved model
    train_triplets = generator.generate_triplets(train_pairs, train_labels)
    with open(os.path.join(output_dir, "train_triplets.pkl"), 'wb') as f:
        pickle.dump(train_triplets, f)
    
    logger.info("Pairs generation completed successfully!")

if __name__ == "__main__":
    # Example usage
    data_dir = "data/lfw"
    output_dir = "data/generated_pairs"
    
    generate_lfw_pairs_for_training(data_dir, output_dir)
