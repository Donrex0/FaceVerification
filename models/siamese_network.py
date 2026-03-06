"""
Siamese Network for Face Verification
Implements Siamese architecture with contrastive and triplet loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese networks.
    
    Loss = (1 - y) * 0.5 * distance^2 + y * 0.5 * max(0, margin - distance)^2
    
    where y = 0 for same class, y = 1 for different class
    """
    
    def __init__(self, margin=2.0):
        """
        Initialize contrastive loss.
        
        Args:
            margin (float): Margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Compute contrastive loss.
        
        Args:
            embedding1 (torch.Tensor): First embedding
            embedding2 (torch.Tensor): Second embedding
            label (torch.Tensor): Label (0 for same, 1 for different)
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        # Compute Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Contrastive loss
        loss_contrastive = torch.mean(
            (1 - label) * 0.5 * torch.pow(distance, 2) +
            label * 0.5 * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )
        
        return loss_contrastive

class TripletLoss(nn.Module):
    """
    Triplet loss function for improved Siamese networks.
    
    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    """
    
    def __init__(self, margin=0.2):
        """
        Initialize triplet loss.
        
        Args:
            margin (float): Margin between positive and negative pairs
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor (torch.Tensor): Anchor embedding
            positive (torch.Tensor): Positive embedding
            negative (torch.Tensor): Negative embedding
            
        Returns:
            torch.Tensor: Triplet loss
        """
        # Compute distances
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return torch.mean(losses)

class SiameseNetwork(nn.Module):
    """
    Siamese network for face verification.
    
    Takes two images and computes their similarity using a shared encoder.
    """
    
    def __init__(self, encoder, loss_type='contrastive'):
        """
        Initialize Siamese network.
        
        Args:
            encoder (nn.Module): Shared encoder network
            loss_type (str): 'contrastive' or 'triplet'
        """
        super(SiameseNetwork, self).__init__()
        
        self.encoder = encoder
        self.loss_type = loss_type
        
        # Initialize loss function
        if loss_type == 'contrastive':
            self.criterion = ContrastiveLoss(margin=2.0)
        elif loss_type == 'triplet':
            self.criterion = TripletLoss(margin=0.2)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        logger.info(f"Created Siamese network with {loss_type} loss")
    
    def forward(self, x1, x2=None, x3=None):
        """
        Forward pass through the Siamese network.
        
        Args:
            x1 (torch.Tensor): First input image (anchor)
            x2 (torch.Tensor): Second input image (positive for contrastive/triplet)
            x3 (torch.Tensor): Third input image (negative for triplet)
            
        Returns:
            dict: Dictionary containing embeddings and loss
        """
        # Encode inputs
        embedding1 = self.encoder(x1)
        
        if self.loss_type == 'contrastive' and x2 is not None:
            embedding2 = self.encoder(x2)
            return {
                'embedding1': embedding1,
                'embedding2': embedding2
            }
        elif self.loss_type == 'triplet' and x2 is not None and x3 is not None:
            embedding2 = self.encoder(x2)
            embedding3 = self.encoder(x3)
            return {
                'anchor': embedding1,
                'positive': embedding2,
                'negative': embedding3
            }
        else:
            return {'embedding1': embedding1}
    
    def compute_loss(self, outputs, labels=None):
        """
        Compute loss based on the loss type.
        
        Args:
            outputs (dict): Network outputs
            labels (torch.Tensor): Labels (for contrastive loss)
            
        Returns:
            torch.Tensor: Computed loss
        """
        if self.loss_type == 'contrastive':
            if 'embedding1' not in outputs or 'embedding2' not in outputs or labels is None:
                raise ValueError("Contrastive loss requires embedding1, embedding2, and labels")
            
            return self.criterion(outputs['embedding1'], outputs['embedding2'], labels)
        
        elif self.loss_type == 'triplet':
            if 'anchor' not in outputs or 'positive' not in outputs or 'negative' not in outputs:
                raise ValueError("Triplet loss requires anchor, positive, and negative embeddings")
            
            return self.criterion(outputs['anchor'], outputs['positive'], outputs['negative'])
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def predict_similarity(self, x1, x2, threshold=0.5):
        """
        Predict similarity between two images.
        
        Args:
            x1 (torch.Tensor): First image
            x2 (torch.Tensor): Second image
            threshold (float): Similarity threshold
            
        Returns:
            tuple: (similarity_score, prediction)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x1, x2)
            embedding1 = outputs['embedding1']
            embedding2 = outputs['embedding2']
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2)
            
            # Convert similarity score to distance
            distance = 1 - similarity
            
            # Predict based on distance threshold
            prediction = (distance < threshold).float()
            
            return similarity, prediction

class ImprovedSiameseNetwork(nn.Module):
    """
    Improved Siamese network with additional features.
    """
    
    def __init__(self, encoder, loss_type='triplet', margin=0.2):
        """
        Initialize improved Siamese network.
        
        Args:
            encoder (nn.Module): Shared encoder network
            loss_type (str): 'contrastive' or 'triplet'
            margin (float): Loss margin
        """
        super(ImprovedSiameseNetwork, self).__init__()
        
        self.encoder = encoder
        self.loss_type = loss_type
        
        # Initialize loss function with custom margin
        if loss_type == 'contrastive':
            self.criterion = ContrastiveLoss(margin=margin)
        elif loss_type == 'triplet':
            self.criterion = TripletLoss(margin=margin)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Additional projection head for better embeddings
        embedding_dim = encoder.get_embedding_dim() if hasattr(encoder, 'get_embedding_dim') else 128
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        logger.info(f"Created Improved Siamese network with {loss_type} loss (margin={margin})")
    
    def forward(self, x1, x2=None, x3=None):
        """
        Forward pass with projection head.
        """
        # Encode inputs
        embedding1 = self.encoder(x1)
        embedding1 = self.projection_head(embedding1)
        
        if self.loss_type == 'contrastive' and x2 is not None:
            embedding2 = self.encoder(x2)
            embedding2 = self.projection_head(embedding2)
            return {
                'embedding1': embedding1,
                'embedding2': embedding2
            }
        elif self.loss_type == 'triplet' and x2 is not None and x3 is not None:
            embedding2 = self.encoder(x2)
            embedding2 = self.projection_head(embedding2)
            embedding3 = self.encoder(x3)
            embedding3 = self.projection_head(embedding3)
            return {
                'anchor': embedding1,
                'positive': embedding2,
                'negative': embedding3
            }
        else:
            return {'embedding1': embedding1}
    
    def compute_loss(self, outputs, labels=None):
        """Compute loss."""
        if self.loss_type == 'contrastive':
            if 'embedding1' not in outputs or 'embedding2' not in outputs or labels is None:
                raise ValueError("Contrastive loss requires embedding1, embedding2, and labels")
            
            return self.criterion(outputs['embedding1'], outputs['embedding2'], labels)
        
        elif self.loss_type == 'triplet':
            if 'anchor' not in outputs or 'positive' not in outputs or 'negative' not in outputs:
                raise ValueError("Triplet loss requires anchor, positive, and negative embeddings")
            
            return self.criterion(outputs['anchor'], outputs['positive'], outputs['negative'])
        
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def predict_similarity(self, x1, x2, threshold=0.5):
        """Predict similarity with projection head."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x1, x2)
            embedding1 = outputs['embedding1']
            embedding2 = outputs['embedding2']
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(embedding1, embedding2)
            
            # Convert similarity score to distance
            distance = 1 - similarity
            
            # Predict based on distance threshold
            prediction = (distance < threshold).float()
            
            return similarity, prediction

def create_siamese_network(encoder, loss_type='contrastive', improved=False, margin=None):
    """
    Create a Siamese network.
    
    Args:
        encoder (nn.Module): Encoder network
        loss_type (str): Type of loss function
        improved (bool): Whether to use improved version
        margin (float): Custom margin (if None, uses default)
        
    Returns:
        nn.Module: Siamese network
    """
    if improved:
        if margin is None:
            margin = 0.2 if loss_type == 'triplet' else 2.0
        network = ImprovedSiameseNetwork(encoder, loss_type=loss_type, margin=margin)
    else:
        network = SiameseNetwork(encoder, loss_type=loss_type)
    
    return network

def test_siamese_network():
    """Test the Siamese network with dummy data."""
    from .baseline_cnn import create_baseline_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create encoder
    encoder = create_baseline_model(model_type='simple', embedding_dim=128)
    encoder.to(device)
    
    # Create Siamese networks
    contrastive_net = create_siamese_network(encoder, loss_type='contrastive')
    triplet_net = create_siamese_network(encoder, loss_type='triplet')
    
    contrastive_net.to(device)
    triplet_net.to(device)
    
    # Test data
    batch_size = 4
    dummy_x1 = torch.randn(batch_size, 3, 160, 160).to(device)
    dummy_x2 = torch.randn(batch_size, 3, 160, 160).to(device)
    dummy_x3 = torch.randn(batch_size, 3, 160, 160).to(device)
    dummy_labels = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    # Test contrastive network
    contrastive_outputs = contrastive_net(dummy_x1, dummy_x2)
    contrastive_loss = contrastive_net.compute_loss(contrastive_outputs, dummy_labels)
    
    print("Contrastive network outputs:")
    print(f"  Embedding1 shape: {contrastive_outputs['embedding1'].shape}")
    print(f"  Embedding2 shape: {contrastive_outputs['embedding2'].shape}")
    print(f"  Loss: {contrastive_loss.item():.4f}")
    
    # Test triplet network
    triplet_outputs = triplet_net(dummy_x1, dummy_x2, dummy_x3)
    triplet_loss = triplet_net.compute_loss(triplet_outputs)
    
    print("\nTriplet network outputs:")
    print(f"  Anchor shape: {triplet_outputs['anchor'].shape}")
    print(f"  Positive shape: {triplet_outputs['positive'].shape}")
    print(f"  Negative shape: {triplet_outputs['negative'].shape}")
    print(f"  Loss: {triplet_loss.item():.4f}")
    
    # Test similarity prediction
    similarity, prediction = contrastive_net.predict_similarity(dummy_x1[:2], dummy_x2[:2])
    print("\nSimilarity prediction:")
    print(f"  Similarity scores: {similarity}")
    print(f"  Predictions: {prediction}")

if __name__ == "__main__":
    test_siamese_network()
