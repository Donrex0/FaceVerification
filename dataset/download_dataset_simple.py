"""
Simplified Dataset Download Script for LFW
Creates dummy data structure for testing without heavy downloads.
"""

import os
import numpy as np
from PIL import Image
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DummyLFWDataset:
    """Creates a dummy LFW-like dataset for testing purposes."""
    
    def __init__(self, data_dir="data/lfw", num_people=50, images_per_person=10):
        """
        Initialize dummy dataset creator.
        
        Args:
            data_dir (str): Directory to save dataset
            num_people (int): Number of different people to simulate
            images_per_person (int): Number of images per person
        """
        self.data_dir = data_dir
        self.num_people = num_people
        self.images_per_person = images_per_person
        self.pairs_file = os.path.join(data_dir, "lfw_pairs.pkl")
        self.metadata_file = os.path.join(data_dir, "metadata.pkl")
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "people"), exist_ok=True)
    
    def create_dummy_images(self):
        """Create dummy face images for testing."""
        logger.info(f"Creating {self.num_people} people with {self.images_per_person} images each...")
        
        self.people_data = {}
        
        for person_id in range(self.num_people):
            person_name = f"person_{person_id:03d}"
            person_dir = os.path.join(self.data_dir, "people", person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            person_images = []
            
            for img_id in range(self.images_per_person):
                # Create a random RGB image (160x160)
                img_array = np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                img_path = os.path.join(person_dir, f"{person_name}_{img_id:04d}.jpg")
                img.save(img_path)
                person_images.append(img_path)
            
            self.people_data[person_name] = person_images
        
        logger.info(f"Created {len(self.people_data)} people with images")
    
    def create_pairs(self):
        """Create positive and negative pairs."""
        logger.info("Creating face pairs...")
        
        people_list = list(self.people_data.keys())
        positive_pairs = []
        negative_pairs = []
        
        # Create positive pairs (same person)
        for person_name in people_list:
            images = self.people_data[person_name]
            if len(images) >= 2:
                # Create multiple positive pairs per person
                for i in range(0, len(images)-1, 2):
                    if i+1 < len(images):
                        positive_pairs.append((images[i], images[i+1], 1))
        
        # Create negative pairs (different people)
        num_negative_pairs = len(positive_pairs)
        for _ in range(num_negative_pairs):
            # Select two different people
            person1, person2 = np.random.choice(people_list, 2, replace=False)
            
            # Select random images from each person
            img1 = np.random.choice(self.people_data[person1])
            img2 = np.random.choice(self.people_data[person2])
            
            negative_pairs.append((img1, img2, 0))
        
        # Combine and shuffle pairs
        all_pairs = positive_pairs + negative_pairs
        np.random.shuffle(all_pairs)
        
        # Split into train and test
        split_idx = int(0.8 * len(all_pairs))
        train_pairs = all_pairs[:split_idx]
        test_pairs = all_pairs[split_idx:]
        
        logger.info(f"Created {len(train_pairs)} training pairs and {len(test_pairs)} test pairs")
        
        return train_pairs, test_pairs
    
    def save_dataset(self):
        """Save the dataset metadata."""
        logger.info("Saving dataset metadata...")
        
        # Create pairs
        train_pairs, test_pairs = self.create_pairs()
        
        # Save pairs data
        pairs_data = {
            'train': train_pairs,
            'test': test_pairs,
            'num_people': self.num_people,
            'images_per_person': self.images_per_person
        }
        
        with open(self.pairs_file, 'wb') as f:
            pickle.dump(pairs_data, f)
        
        # Save metadata
        metadata = {
            'people': list(self.people_data.keys()),
            'people_data': self.people_data,
            'total_images': sum(len(images) for images in self.people_data.values()),
            'dataset_type': 'dummy_lfw'
        }
        
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Dataset saved to {self.data_dir}")
        logger.info(f"Pairs file: {self.pairs_file}")
        logger.info(f"Metadata file: {self.metadata_file}")
    
    def create_dataset(self):
        """Main method to create the complete dataset."""
        logger.info("Creating dummy LFW dataset...")
        
        self.create_dummy_images()
        self.save_dataset()
        
        logger.info("Dummy dataset creation completed!")

def main():
    """Main function to create the dataset."""
    downloader = DummyLFWDataset(
        data_dir="data/lfw",
        num_people=20,  # Reduced for faster testing
        images_per_person=8
    )
    
    downloader.create_dataset()
    
    # Verify the dataset was created
    if os.path.exists("data/lfw/lfw_pairs.pkl"):
        logger.info("✅ Dataset created successfully!")
        logger.info("You can now run the training scripts.")
    else:
        logger.error("❌ Dataset creation failed!")

if __name__ == "__main__":
    main()
