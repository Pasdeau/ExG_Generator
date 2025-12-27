
import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) for EMG Gesture Recognition.
    Architecture:
    - Feature Extractor: CNN/TCN/ResNet (truncated)
    - Label Predictor: Classification Head
    - Domain Classifier: Domain Discrimination Head (Gradient Reversal)
    """
    def __init__(self, feature_extractor, input_dim, hidden_dim=128, n_classes=8, n_domains=2):
        super(DANN, self).__init__()
        
        self.feature_extractor = feature_extractor
        
        # Label Predictor
        self.label_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_domains)
        )
        
    def forward(self, x, alpha=1.0):
        """
        Forward pass.
        Args:
            x: Input tensor (B, C, L)
            alpha: GRL scaling factor (0 -> 1 during training)
        Returns:
            class_output, domain_output
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1) # Flatten (B, F)
        
        # Class prediction
        class_output = self.label_predictor(features)
        
        # Domain prediction (with GRL)
        reverse_feature = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output

def get_dann_model(backbone_name='tcn', n_channels=32, n_classes=8, n_domains=2, device='cpu'):
    """Factory for DANN model."""
    from .dl_models import get_model, SimpleCNN1D, TCN #, ResNet18_1D
    
    if backbone_name == 'tcn':
        # Create TCN and access .network + .min_pool
        # TCN output of .network is (B, 64, L)
        # after global pool (B, 64)
        base_model = TCN(n_channels=n_channels, n_classes=n_classes)
        feature_extractor = nn.Sequential(
            base_model.network,
            base_model.min_pool
        )
        feature_dim = 64 # From TCN definition
        
    elif backbone_name == 'cnn':
         base_model = SimpleCNN1D(n_channels=n_channels, n_classes=n_classes)
         feature_extractor = nn.Sequential(
             base_model.features,
             base_model.global_pool
         )
         # CNN hidden dim logic: conv_channels=[64, 128, 256]. Last is 256.
         feature_dim = 256
         
    else:
        raise ValueError(f"Backbone {backbone_name} not supported for DANN yet.")
        
    model = DANN(feature_extractor, feature_dim, hidden_dim=128, n_classes=n_classes, n_domains=n_domains)
    return model.to(device)
