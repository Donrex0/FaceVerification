"""
Improved Face Verification Model with ResNet50 Backbone
Uses transfer learning with pretrained ImageNet weights for better performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)

class ResNet50Encoder(nn.Module):
    """
    ResNet50-based encoder for face verification.
    
    Architecture:
    - ResNet50 Backbone (pretrained on ImageNet)
    - Global Average Pooling
    - Dense Layer
    - Batch Normalization
    - Dropout
    - 128-dimension embedding
    """
    
    def __init__(self, embedding_dim=128, pretrained=True, freeze_backbone=False):
        """
        Initialize ResNet50 encoder.
        
        Args:
            embedding_dim (int): Dimension of the embedding vector
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze the ResNet backbone
        """
        super(ResNet50Encoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("ResNet50 backbone frozen")
        
        # Additional layers for face embedding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 512)  # ResNet50 output is 2048
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.embedding_layer = nn.Linear(256, embedding_dim)
        
        # Initialize new layers
        self._initialize_new_layers()
        
        logger.info(f"Created ResNet50 encoder with embedding_dim={embedding_dim}")
    
    def _initialize_new_layers(self):
        """Initialize newly added layers."""
        for m in [self.fc1, self.fc2, self.embedding_layer]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: Embedding vectors of shape (batch_size, embedding_dim)
        """
        # Pass through ResNet50 backbone
        features = self.backbone(x)  # Shape: (batch_size, 2048, 1, 1)
        
        # Global average pooling
        features = self.global_pool(features)  # Shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # Shape: (batch_size, 2048)
        
        # Fully connected layers
        x = self.fc1(features)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Final embedding
        embedding = self.embedding_layer(x)
        
        # L2 normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def get_embedding_dim(self):
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def unfreeze_backbone(self):
        """Unfreeze the ResNet backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        logger.info("ResNet50 backbone unfrozen")
    
    def freeze_backbone_layers(self):
        """Freeze the ResNet backbone layers."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.freeze_backbone = True
        logger.info("ResNet50 backbone frozen")

class ImprovedResNetEncoder(nn.Module):
    """
    Improved ResNet50 encoder with additional enhancements.
    """
    
    def __init__(self, embedding_dim=128, pretrained=True, freeze_backbone=False):
        """
        Initialize improved ResNet encoder.
        
        Args:
            embedding_dim (int): Dimension of the embedding vector
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze the ResNet backbone
        """
        super(ImprovedResNetEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Keep avg pool
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("ResNet50 backbone frozen")
        
        # Enhanced feature extraction
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Multiple projection heads for better representation
        self.projection_head1 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.projection_head2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.projection_head3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.embedding_layer = nn.Linear(256, embedding_dim)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        
        # Initialize new layers
        self._initialize_new_layers()
        
        logger.info(f"Created Improved ResNet50 encoder with embedding_dim={embedding_dim}")
    
    def _initialize_new_layers(self):
        """Initialize newly added layers."""
        modules = [self.projection_head1, self.projection_head2, 
                  self.projection_head3, self.embedding_layer, self.attention]
        
        for module in modules:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with attention mechanism.
        """
        # Pass through ResNet backbone
        features = self.backbone(x)  # Shape: (batch_size, 2048, H, W)
        
        # Global average pooling
        pooled_features = self.adaptive_pool(features)  # Shape: (batch_size, 2048, 1, 1)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # Shape: (batch_size, 2048)
        
        # Attention mechanism
        attention_weights = self.attention(pooled_features)  # Shape: (batch_size, 1)
        attended_features = pooled_features * attention_weights  # Shape: (batch_size, 2048)
        
        # Projection heads
        x = self.projection_head1(attended_features)
        x = self.projection_head2(x)
        x = self.projection_head3(x)
        
        # Final embedding
        embedding = self.embedding_layer(x)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def get_embedding_dim(self):
        """Get the embedding dimension."""
        return self.embedding_dim

class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-based encoder for better performance with fewer parameters.
    """
    
    def __init__(self, embedding_dim=128, model_name='efficientnet_b0', pretrained=True):
        """
        Initialize EfficientNet encoder.
        
        Args:
            embedding_dim (int): Dimension of the embedding vector
            model_name (str): EfficientNet model name
            pretrained (bool): Whether to use pretrained weights
        """
        super(EfficientNetEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported EfficientNet model: {model_name}")
        
        # Remove the classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        # Initialize new layers
        self._initialize_new_layers()
        
        logger.info(f"Created {model_name} encoder with embedding_dim={embedding_dim}")
    
    def _initialize_new_layers(self):
        """Initialize newly added layers."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        embedding = self.projection(features)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def get_embedding_dim(self):
        """Get the embedding dimension."""
        return self.embedding_dim

def create_improved_model(model_type='resnet50', embedding_dim=128, **kwargs):
    """
    Create an improved model for face verification.
    
    Args:
        model_type (str): Type of model ('resnet50', 'improved_resnet50', 'efficientnet')
        embedding_dim (int): Embedding dimension
        **kwargs: Additional model-specific arguments
        
    Returns:
        nn.Module: Improved model
    """
    if model_type == 'resnet50':
        model = ResNet50Encoder(embedding_dim=embedding_dim, **kwargs)
    elif model_type == 'improved_resnet50':
        model = ImprovedResNetEncoder(embedding_dim=embedding_dim, **kwargs)
    elif model_type.startswith('efficientnet'):
        model = EfficientNetEncoder(embedding_dim=embedding_dim, model_name=model_type, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def test_improved_models():
    """Test improved models with dummy input."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different models
    models_to_test = [
        ('resnet50', {}),
        ('improved_resnet50', {}),
        ('efficientnet_b0', {})
    ]
    
    dummy_input = torch.randn(4, 3, 160, 160).to(device)
    
    for model_name, kwargs in models_to_test:
        print(f"\nTesting {model_name}:")
        
        try:
            model = create_improved_model(model_type=model_name, embedding_dim=128, **kwargs)
            model.to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output norm: {torch.norm(output, dim=1).mean():.4f}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")

if __name__ == "__main__":
    test_improved_models()
