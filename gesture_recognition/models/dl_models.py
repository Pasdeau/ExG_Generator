
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class SimpleCNN1D(nn.Module):
    """
    A simple yet effective 1D CNN baseline.
    Input: (B, C, L)
    """
    def __init__(self, n_channels=32, n_classes=8, conv_channels=[64, 128, 256], kernel_size=3, dropout=0.5):
        super(SimpleCNN1D, self).__init__()
        
        layers = []
        in_c = n_channels
        
        for out_c in conv_channels:
            layers.append(nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ))
            in_c = out_c
            
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_c, n_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Chomp1d(nn.Module):
    """
    Removes the last elements of a sequence to ensure causality.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    TCN Temporal Block with Dilated Causal Convolution and Residual Connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Branch 1
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Branch 2
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN).
    """
    def __init__(self, n_channels=32, n_classes=8, num_channels=[64, 64, 64, 64], kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal padding = (kernel_size - 1) * dilation
            padding = (kernel_size - 1) * dilation_size
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)
        self.min_pool = nn.AdaptiveAvgPool1d(1) # Or use last step
        self.fc = nn.Linear(num_channels[-1], n_classes)

    def forward(self, x):
        # x: (B, C, L)
        y = self.network(x) # (B, 64, L)
        # TCN usually takes the last output for classification, but Global Avg Pool is robust
        # Let's try Global Avg Pool first
        y = self.min_pool(y).squeeze(-1)
        return self.fc(y)

# Import and re-export ResNet1D
try:
    from .resnet import ResNet18_1D
except ImportError:
    # Fallback if relative import fails during testing
    try:
        from gesture_recognition.models.resnet import ResNet18_1D
    except ImportError:
        ResNet18_1D = None

def get_model(model_name, n_channels, n_classes, device='cpu'):
    """Factory function to get model."""
    if model_name.lower() == 'cnn':
        model = SimpleCNN1D(n_channels=n_channels, n_classes=n_classes)
    elif model_name.lower() == 'tcn':
        model = TCN(n_channels=n_channels, n_classes=n_classes)
    elif model_name.lower() == 'resnet':
        if ResNet18_1D is None:
            raise ImportError("ResNet module not found")
        model = ResNet18_1D(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model.to(device)

if __name__ == "__main__":
    # Test models
    input_tensor = torch.randn(8, 32, 409) # Batch, Channels, Length
    
    print("Testing CNN...")
    cnn = SimpleCNN1D(n_channels=32, n_classes=8)
    out_cnn = cnn(input_tensor)
    print(f"CNN Output: {out_cnn.shape}")
    
    print("\nTesting TCN...")
    tcn = TCN(n_channels=32, n_classes=8)
    out_tcn = tcn(input_tensor)
    print(f"TCN Output: {out_tcn.shape}")
    
    print("\nDeep Learning Models Verified!")
