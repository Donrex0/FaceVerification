"""
Dataset Download Script for LFW (Labeled Faces in the Wild)
Automatically downloads and prepares the LFW dataset for face verification.
"""

import os
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from sklearn.model_selection import train_test_split
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LFWDatasetDownloader:
    """Downloads and prepares LFW dataset for face verification."""
    
    def __init__(self, data_dir="data/lfw", min_faces_per_person=10):
        """
        Initialize the downloader.
        
        Args:
            data_dir (str): Directory to save dataset
            min_faces_per_person (int): Minimum images per person to include
        """
        self.data_dir = data_dir
        self.min_faces_per_person = min_faces_per_person
        self.pairs_file = os.path.join(data_dir, "lfw_pairs.pkl")
        self.metadata_file = os.path.join(data_dir, "metadata.pkl")
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
    
    def download_lfw_pairs(self):
        """Download LFW pairs dataset."""
        logger.info("Downloading LFW pairs dataset...")
        
        try:
            # Download both positive (same person) and negative (different people) pairs
            lfw_pairs_train = fetch_lfw_pairs(subset='train', download_if_missing=True)
            lfw_pairs_test = fetch_lfw_pairs(subset='test', download_if_missing=True)
            
            logger.info(f"Training pairs: {len(lfw_pairs_train.pairs)}")
            logger.info(f"Test pairs: {len(lfw_pairs_test.pairs)}")
            
            # Combine datasets
            all_pairs = {
                'train': {
                    'pairs': lfw_pairs_train.pairs,
                    'targets': lfw_pairs_train.target,
                    'data': lfw_pairs_train.data
                },
                'test': {
                    'pairs': lfw_pairs_test.pairs,
                    'targets': lfw_pairs_test.target,
                    'data': lfw_pairs_test.data
                }
            }
            
            # Save dataset
            with open(self.pairs_file, 'wb') as f:
                pickle.dump(all_pairs, f)
            
            logger.info(f"Dataset saved to {self.pairs_file}")
            return all_pairs
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def download_lfw_people(self):
        """Download LFW people dataset for additional analysis."""
        logger.info("Downloading LFW people dataset...")
        
        try:
            from sklearn.datasets import fetch_lfw_people
            
            lfw_people = fetch_lfw_people(min_faces_per_person=self.min_faces_per_person, 
                                        download_if_missing=True)
            
            logger.info(f"Found {len(np.unique(lfw_people.target))} people")
            logger.info(f"Total images: {len(lfw_people.images)}")
            
            # Save metadata
            metadata = {
                'target_names': lfw_people.target_names,
                'images': lfw_people.images,
                'targets': lfw_people.target,
                'description': lfw_people.DESCR
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Metadata saved to {self.metadata_file}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error downloading people dataset: {e}")
            raise
    
    def get_dataset_info(self):
        """Get information about the downloaded dataset."""
        if not os.path.exists(self.pairs_file):
            return None
        
        try:
            with open(self.pairs_file, 'rb') as f:
                data = pickle.load(f)
            
            info = {
                'train_pairs': len(data['train']['pairs']),
                'test_pairs': len(data['test']['pairs']),
                'train_positive': sum(data['train']['targets']),
                'train_negative': len(data['train']['targets']) - sum(data['train']['targets']),
                'test_positive': sum(data['test']['targets']),
                'test_negative': len(data['test']['targets']) - sum(data['test']['targets'])
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error reading dataset info: {e}")
            return None
    
    def is_dataset_downloaded(self):
        """Check if dataset is already downloaded."""
        return os.path.exists(self.pairs_file)

def main():
    """Main function to download and prepare dataset."""
    # Initialize downloader
    downloader = LFWDatasetDownloader()
    
    # Check if dataset already exists
    if downloader.is_dataset_downloaded():
        logger.info("Dataset already exists. Getting info...")
        info = downloader.get_dataset_info()
        if info:
            logger.info("Dataset Information:")
            for key, value in info.items():
                logger.info(f"  {key}: {value}")
        return
    
    # Download datasets
    logger.info("Starting dataset download...")
    pairs_data = downloader.download_lfw_pairs()
    people_data = downloader.download_lfw_people()
    
    # Display final info
    info = downloader.get_dataset_info()
    logger.info("Download completed successfully!")
    logger.info("Final Dataset Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()
