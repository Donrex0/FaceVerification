"""
Baseline CNN Model for Face Verification
Simple convolutional neural network for generating face embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class BaselineCNN(nn.Module):
    """
    Baseline CNN architecture for face verification.
    
    Architecture:
    - Conv2D -> ReLU -> MaxPooling
    - Conv2D -> ReLU -> MaxPooling
    - Flatten -> Fully Connected -> 128-dim embedding
    """
    
    def __init__(self, embedding_dim=128, input_channels=3):
        """
        Initialize the baseline CNN.
        
        Args:
            embedding_dim (int): Dimension of the embedding vector
            input_channels (int): Number of input channels (3 for RGB)
        """
        super(BaselineCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 max pooling operations with 160x160 input:
        # 160 -> 80 -> 40 -> 20 -> 10
        # So the feature map size is 256 x 10 x 10 = 25600
        self.fc1 = nn.Linear(256 * 10 * 10, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Embedding vectors of shape (batch_size, embedding_dim)
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # Shape: (batch_size, 32, 80, 80)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # Shape: (batch_size, 64, 40, 40)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # Shape: (batch_size, 128, 20, 20)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)  # Shape: (batch_size, 256, 10, 10)
        
        # Flatten
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 256 * 10 * 10)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final embedding layer
        embedding = self.fc3(x)
        
        # L2 normalize embeddings for better distance computation
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def get_embedding_dim(self):
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def freeze_features(self):
        """Freeze feature extraction layers."""
        for name, param in self.named_parameters():
            if 'fc' not in name:  # Freeze all layers except fully connected
                param.requires_grad = False
        logger.info("Feature extraction layers frozen")
    
    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All layers unfrozen")

class SimpleBaselineCNN(nn.Module):
    """
    Simpler version of the baseline CNN for faster training.
    
    Architecture:
    - Conv2D -> ReLU -> MaxPooling
    - Conv2D -> ReLU -> MaxPooling
    - Flatten -> Fully Connected -> 128-dim embedding
    """
    
    def __init__(self, embedding_dim=128, input_channels=3):
        """
        Initialize the simple baseline CNN.
        
        Args:
            embedding_dim (int): Dimension of the embedding vector
            input_channels (int): Number of input channels
        """
        super(SimpleBaselineCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        # After 2 max pooling operations with 160x160 input:
        # 160 -> 80 -> 40
        # So the feature map size is 64 x 40 x 40 = 102400
        self.fc1 = nn.Linear(64 * 40 * 40, 256)
        self.fc2 = nn.Linear(256, embedding_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Embedding vectors
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # Shape: (batch_size, 32, 80, 80)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # Shape: (batch_size, 64, 40, 40)
        
        # Flatten
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 64 * 40 * 40)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Final embedding layer
        embedding = self.fc2(x)
        
        # L2 normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

def create_baseline_model(model_type='standard', embedding_dim=128, input_channels=3):
    """
    Create a baseline model.
    
    Args:
        model_type (str): 'standard' or 'simple'
        embedding_dim (int): Embedding dimension
        input_channels (int): Number of input channels
        
    Returns:
        nn.Module: Baseline model
    """
    if model_type == 'simple':
        model = SimpleBaselineCNN(embedding_dim=embedding_dim, input_channels=input_channels)
        logger.info("Created Simple Baseline CNN")
    else:
        model = BaselineCNN(embedding_dim=embedding_dim, input_channels=input_channels)
        logger.info("Created Standard Baseline CNN")
    
    return model

def test_model():
    """Test the model with dummy input."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_baseline_model(model_type='standard', embedding_dim=128)
    model.to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(4, 3, 160, 160).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norm: {torch.norm(output, dim=1).mean():.4f}")
    
    # Test simple model
    simple_model = create_baseline_model(model_type='simple', embedding_dim=128)
    simple_model.to(device)
    
    with torch.no_grad():
        simple_output = simple_model(dummy_input)
    
    print(f"Simple model output shape: {simple_output.shape}")
    print(f"Simple model output norm: {torch.norm(simple_output, dim=1).mean():.4f}")

if __name__ == "__main__":
    test_model()
