"""
1D Convolutional Neural Network for EMG Gesture Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Conv1D -> BatchNorm -> ReLU -> MaxPool"""
    def __init__(self, in_ch, out_ch, kernel_size=5, pool_size=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class GestureClassifier1D(nn.Module):
    """
    Simple 1D-CNN for EMG Gesture Classification.
    
    Input: (B, n_channels*2, seq_len) e.g., (B, 32, 2048) - 16 EMG + 16 Velocity
    Output: (B, n_classes) logits
    """
    def __init__(self, n_channels: int = 32, n_classes: int = 16, seq_len: int = 2048):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder
        self.block1 = ConvBlock(n_channels, 32, kernel_size=7, pool_size=2)
        self.block2 = ConvBlock(32, 64, kernel_size=7, pool_size=2)
        self.block3 = ConvBlock(64, 128, kernel_size=5, pool_size=2)
        self.block4 = ConvBlock(128, 256, kernel_size=5, pool_size=2)
        
        # After 4 poolings: seq_len / 16
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, x):
        # x: (B, n_channels, seq_len)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global pooling
        x = self.gap(x) # (B, 256, 1)
        x = x.squeeze(-1) # (B, 256)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


if __name__ == "__main__":
    # Smoke Test
    model = GestureClassifier1D(n_channels=32, n_classes=16)
    x = torch.randn(4, 32, 2048) # 32 channels: 16 EMG + 16 Velocity
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

