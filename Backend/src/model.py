"""
Model architecture for cancer stage and progression prediction.
Uses transfer learning with ResNet50/EfficientNet backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict, Optional


class CancerStagePredictor(nn.Module):
    """
    Multi-task model for cancer stage classification and progression risk estimation.
    
    Architecture:
    - Backbone: Pretrained CNN (ResNet50 or EfficientNet)
    - Stage head: Multi-class classifier for cancer stages (0-4)
    - Progression head: Risk level classifier (Low/Medium/High)
    """
    
    def __init__(
        self,
        num_stages: int = 5,
        num_risk_levels: int = 3,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5,
        feature_dim: int = 512
    ):
        """
        Initialize the model.
        
        Args:
            num_stages: Number of cancer stages (default: 5 for stages 0-4)
            num_risk_levels: Number of progression risk levels (default: 3)
            backbone: Backbone architecture ('resnet50' or 'efficientnet')
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            feature_dim: Dimension of feature vector before classification heads
        """
        super(CancerStagePredictor, self).__init__()
        
        self.num_stages = num_stages
        self.num_risk_levels = num_risk_levels
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Load backbone
        if backbone == 'resnet50':
            if pretrained:
                from torchvision.models import ResNet50_Weights
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            # Remove the final fully connected layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            backbone_features = 2048  # ResNet50 feature dimension
        elif backbone == 'efficientnet':
            try:
                from torchvision.models import efficientnet_b0
                try:
                    from torchvision.models import EfficientNet_B0_Weights
                    if pretrained:
                        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
                    else:
                        self.backbone = efficientnet_b0(weights=None)
                except ImportError:
                    # Fallback for older torchvision versions
                    if pretrained:
                        self.backbone = efficientnet_b0(pretrained=True)
                    else:
                        self.backbone = efficientnet_b0(pretrained=False)
                self.backbone = nn.Sequential(*list(self.backbone.features.children()))
                backbone_features = 1280  # EfficientNet-B0 feature dimension
            except:
                # Fallback to ResNet50 if EfficientNet not available
                print("EfficientNet not available, using ResNet50")
                if pretrained:
                    from torchvision.models import ResNet50_Weights
                    self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                else:
                    self.backbone = models.resnet50(weights=None)
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
                backbone_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature extraction layer
        # Note: AdaptiveAvgPool2d and Flatten are applied in forward pass
        # So this only contains the Linear projection layers
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Stage classification head
        self.stage_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_stages)
        )
        
        # Progression risk classification head
        self.progression_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_risk_levels)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for classification heads."""
        for m in [self.stage_classifier, self.progression_classifier]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, C, H, W)
        
        Returns:
            Dictionary containing:
                - 'stage_logits': Logits for stage classification
                - 'stage_probs': Probabilities for stage classification
                - 'progression_logits': Logits for progression risk
                - 'progression_probs': Probabilities for progression risk
                - 'features': Extracted feature vector
        """
        # Extract features using backbone
        backbone_features = self.backbone(x)
        
        # Get feature vector - backbone output is 4D (B, C, H, W)
        # Apply adaptive pooling and flatten
        if len(backbone_features.shape) == 4:
            features = F.adaptive_avg_pool2d(backbone_features, (1, 1))
            features = features.view(features.size(0), -1)
        else:
            features = backbone_features
        
        # Project to feature dimension using feature projection
        features = self.feature_projection(features)
        
        # Get stage predictions
        stage_logits = self.stage_classifier(features)
        stage_probs = F.softmax(stage_logits, dim=1)
        
        # Get progression risk predictions
        progression_logits = self.progression_classifier(features)
        progression_probs = F.softmax(progression_logits, dim=1)
        
        return {
            'stage_logits': stage_logits,
            'stage_probs': stage_probs,
            'progression_logits': progression_logits,
            'progression_probs': progression_probs,
            'features': features
        }
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features only (for Grad-CAM)."""
        backbone_features = self.backbone(x)
        if len(backbone_features.shape) == 4:
            features = F.adaptive_avg_pool2d(backbone_features, (1, 1))
            features = features.view(features.size(0), -1)
        else:
            features = backbone_features
        return self.feature_projection(features)
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features from backbone before classification heads (for Grad-CAM)."""
        return self.backbone(x)


def create_model(
    num_stages: int = 5,
    num_risk_levels: int = 3,
    backbone: str = 'resnet50',
    pretrained: bool = True,
    device: str = 'cuda'
) -> CancerStagePredictor:
    """
    Create and initialize model.
    
    Args:
        num_stages: Number of cancer stages
        num_risk_levels: Number of progression risk levels
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        device: Device to load model on
    
    Returns:
        Initialized model
    """
    model = CancerStagePredictor(
        num_stages=num_stages,
        num_risk_levels=num_risk_levels,
        backbone=backbone,
        pretrained=pretrained
    )
    
    model = model.to(device)
    return model
