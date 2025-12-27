import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv1D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=15, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class NoiseDetector1D(nn.Module):
    """
    Simple 1D U-Net for Binary Segmentation of Time Series.
    Input: (B, 2, L) - Dual channel: Amplitude + Velocity
    Output: (B, 1, L) - Probability of noise
    """
    def __init__(self, n_channels=2, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool1d(2), DoubleConv(64, 128))

        # Decoder
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64) # 64+64 -> 64 (after concat)

        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32) # 32+32 -> 32

        self.up3 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(32, 16) # 16+16 -> 16
        
        # Output
        self.outc = nn.Conv1d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        # Handle potential size mismatch due to odd pooling
        diff = x3.size()[2] - x.size()[2]
        if diff > 0:
             x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        diff = x2.size()[2] - x.size()[2]
        if diff > 0:
             x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        diff = x1.size()[2] - x.size()[2]
        if diff > 0:
             x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)

        logits = self.outc(x)
        return torch.sigmoid(logits)

if __name__ == "__main__":
    # Smoke Test
    model = NoiseDetector1D()
    x = torch.randn(2, 2, 8000) # 2 channels: Amplitude + Velocity
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

