import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (kept from V1.1)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResidualBlock1D(nn.Module):
    """Lightweight ResNet block for V2.0"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale temporal feature fusion for细粒度 gesture discrimination"""
    def __init__(self, d_model=512):
        super().__init__()
        # Capture different temporal scales
        self.conv_short = nn.Conv1d(d_model, d_model//4, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(d_model, d_model//4, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(d_model, d_model//4, kernel_size=15, padding=7)
        self.conv_global = nn.AdaptiveAvgPool1d(1)
        self.fc_global = nn.Linear(d_model, d_model//4)
        self.fusion = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x: (B, d_model, L)
        short = self.relu(self.conv_short(x))
        medium = self.relu(self.conv_medium(x))
        long = self.relu(self.conv_long(x))
        
        global_feat = self.conv_global(x).squeeze(-1)  # (B, d_model)
        global_feat = self.fc_global(global_feat).unsqueeze(-1).expand(-1, -1, x.size(2))  # (B, d/4, L)
        
        fused = torch.cat([short, medium, long, global_feat], dim=1)  # (B, d_model, L)
        out = self.fusion(fused)
        out = self.bn(out)
        return self.relu(out + x)  # Residual connection


class ChannelSpatialAttention(nn.Module):
    """Channel and Spatial attention for EMG signals"""
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = SEBlock(channels, reduction=16)
        # Spatial attention: emphasize important time points
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels),
            nn.BatchNorm1d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        x = self.channel_attn(x)
        # Spatial attention
        spatial_weight = self.spatial_conv(x)
        return x * spatial_weight


class HybridCNNTransformer(nn.Module):
    """
    V2.0 Architecture: CNN for spatial features + Transformer for temporal modeling
    """
    def __init__(self, n_channels=32, n_classes=16, seq_len=2048,
                 d_model=512, nhead=12, num_transformer_layers=6, 
                 dropout=0.25, multitask=False, n_subjects=43):
        super().__init__()
        
        self.multitask = multitask
        
        # Input attention for channel selection
        self.input_attn = ChannelSpatialAttention(n_channels)
        
        # Stem: reduce dimensionality
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Lightweight ResNet backbone (3 stages)
        self.stage1 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage2 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(256, d_model, num_blocks=2, stride=2)
        
        # Multi-scale feature fusion (NEW for V2.0)
        self.multi_scale_fusion = MultiScaleFeatureFusion(d_model)
        
        # Calculate sequence length after CNN
        # Initial: 2048 -> stem: /4 -> stage1: /2 -> stage2: /2 -> stage3: /2
        # = 2048 / 32 = 64
        self.seq_len_after_cnn = seq_len // 32
        
        # Transformer Encoder (Increased from 4->6 layers, 8->12 heads)
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len_after_cnn)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Classification heads
        self.dropout = nn.Dropout(dropout)
        self.fc_gesture = nn.Linear(d_model, n_classes)
        
        if multitask:
            self.fc_subject = nn.Linear(d_model, n_subjects)
        
        if multitask:
            self.fc_subject = nn.Linear(d_model, n_subjects)
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, C, L) = (B, 32, 2048)
        
        # Input-level attention
        x = self.input_attn(x)
        
        # CNN feature extraction
        x = self.stem(x)       # (B, 64, L/4)
        x = self.stage1(x)     # (B, 128, L/8)
        x = self.stage2(x)     # (B, 256, L/16)
        x = self.stage3(x)     # (B, 512, L/32)
        
        # Multi-scale feature fusion
        x = self.multi_scale_fusion(x)  # (B, 512, L/32)
        
        # Prepare for Transformer: (B, d_model, L') -> (L', B, d_model)
        x = x.permute(2, 0, 1)  # (L', B, d_model)
        
        # Positional encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (L', B, d_model)
        
        # Global pooling over sequence
        x = x.mean(dim=0)  # (B, d_model)
        
        # Classification
        x = self.dropout(x)
        gesture_logits = self.fc_gesture(x)
        
        if self.multitask:
            subject_logits = self.fc_subject(x)
            return {'gesture': gesture_logits, 'subject': subject_logits}
        else:
            return gesture_logits


def HybridTransformer_V2(n_channels=32, n_classes=16, **kwargs):
    """Factory function for V2.0 model"""
    return HybridCNNTransformer(
        n_channels=n_channels,
        n_classes=n_classes,
        d_model=512,
        nhead=8,
        num_transformer_layers=4,
        dropout=0.2,
        **kwargs
    )


if __name__ == "__main__":
    # Test
    model = HybridTransformer_V2(n_channels=32, n_classes=16, multitask=True, n_subjects=43)
    x = torch.randn(4, 32, 2048)  # (B, C, L)
    out = model(x)
    print(f"Gesture logits: {out['gesture'].shape}")
    print(f"Subject logits: {out['subject'].shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")


def HybridTransformer_V2_1(n_channels=32, n_classes=16, **kwargs):
    """
    V2.1 - Simplified and stabilized version
    Key changes from V2.0:
    - Reduced Transformer layers: 6 → 3
    - Reduced attention heads: 12 → 8
    - Increased dropout: 0.25 → 0.3
    - NO multi-task learning
    Target: 79-83% accuracy
    """
    return HybridCNNTransformer(
        n_channels=n_channels,
        n_classes=n_classes,
        d_model=512,
        nhead=8,
        num_transformer_layers=3,
        dropout=0.3,
        multitask=False,  # Key: disable multi-task
        **kwargs
    )
