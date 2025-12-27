import torch
import numpy as np


class MixUp:
    """MixUp data augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_x, batch_y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size).to(batch_x.device)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index, :]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam


class MagnitudeWarping:
    """Random magnitude warping to simulate muscle fatigue"""
    def __init__(self, sigma=0.2, p=0.5):
        self.sigma = sigma
        self.p = p
    
    def __call__(self, x):
        if np.random.random() < self.p:
            # x: (C, L)
            C, L = x.shape
            # Generate smooth warping curve
            knot = 4  # number of knots for spline
            orig_steps = np.arange(L)
            random_warps = np.random.normal(loc=1.0, scale=self.sigma, size=(C, knot+2))
            warp_steps = np.linspace(0, L-1, num=knot+2)
            
            # Interpolate for each channel
            for c in range(C):
                warper = np.interp(orig_steps, warp_steps, random_warps[c])
                x[c] *= torch.from_numpy(warper).float()
        
        return x


class SpecAugmentEMG:
    """Frequency masking for EMG signals"""
    def __init__(self, freq_mask_param=10, p=0.5):
        self.freq_mask_param = freq_mask_param
        self.p = p
    
    def __call__(self, x):
        if np.random.random() < self.p:
            # x: (C, L)
            # Apply FFT
            x_fft = torch.fft.rfft(x, dim=-1)
            
            # Mask random frequency band
            freq_len = x_fft.shape[-1]
            mask_start = np.random.randint(0, max(1, freq_len - self.freq_mask_param))
            mask_end = min(mask_start + self.freq_mask_param, freq_len)
            
            x_fft[:, mask_start:mask_end] = 0
            
            # Inverse FFT
            x = torch.fft.irfft(x_fft, n=x.shape[-1], dim=-1)
        
        return x


class ComposeV2:
    """Compose transforms for V2.0"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
