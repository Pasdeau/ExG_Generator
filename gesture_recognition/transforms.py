import torch
import numpy as np
import random
from scipy.interpolate import interp1d

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class GaussianNoise:
    """Add Gaussian noise to the signal."""
    def __init__(self, snr_db_range=(10, 30), p=0.5):
        self.snr_db_range = snr_db_range
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            # x shape: (Channels, Time)
            # Calculate signal power per channel
            sig_power = torch.mean(x**2, dim=1, keepdim=True)
            
            # Generate random SNR
            snr_db = random.uniform(*self.snr_db_range)
            snr = 10 ** (snr_db / 10)
            
            # Calculate noise power
            noise_power = sig_power / snr
            
            # Generate noise
            noise = torch.randn_like(x) * torch.sqrt(noise_power)
            return x + noise
        return x

class RandomTimeWarp:
    """Randomly stretch or compress the signal in time."""
    def __init__(self, warp_factor=0.2, p=0.5):
        self.warp_factor = warp_factor # e.g. 0.2 means +/- 20% speed change
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            # x shape: (Channels, Time)
            n_channels, n_samples = x.shape
            
            # Generate random speed factor
            speed = random.uniform(1 - self.warp_factor, 1 + self.warp_factor)
            
            # Original time points
            orig_t = np.arange(n_samples)
            
            # New time points (stretched/compressed)
            new_t = np.linspace(0, n_samples - 1, int(n_samples * speed))
            
            # Interpolate
            x_np = x.numpy()
            warped_x = np.zeros((n_channels, len(new_t)), dtype=np.float32)
            
            for c in range(n_channels):
                f = interp1d(orig_t, x_np[c], kind='linear', fill_value='extrapolate')
                warped_x[c] = f(new_t)
            
            # Resample back to original length (crop or pad)
            final_x = np.zeros((n_channels, n_samples), dtype=np.float32)
            
            if len(new_t) > n_samples:
                # Crop middle
                start = (len(new_t) - n_samples) // 2
                final_x = warped_x[:, start:start+n_samples]
            else:
                # Pad middle
                start = (n_samples - len(new_t)) // 2
                final_x[:, start:start+len(new_t)] = warped_x
                
            return torch.from_numpy(final_x)
            
        return x

class ElectrodeShift:
    """Simulate electrode displacement by mixing adjacent channels."""
    def __init__(self, max_shift=0.3, p=0.5):
        self.max_shift = max_shift # Max mixing factor
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            # x shape: (Channels, Time). Assuming channels are ordered ring 1 (0-7), ring 2 (8-15)
            # We handle the first 16 channels (EMG), ignore velocity (16-31) for mixing logic, 
            # OR mix velocity same way? Better to mix EMG then recompute velocity?
            # For simplicity, let's assume input is just raw EMG or we mix both parts independently if stacked.
            
            # Let's assume input is (32, Time) where 0-15 is EMG, 16-31 is Velocity.
            # We will mix 0-15.
            
            out = x.clone()
            n_emg = 16
            
            # Random shift direction (+1 or -1) and magnitude
            shift_mag = random.uniform(0, self.max_shift)
            direction = 1 if random.random() > 0.5 else -1
            
            for ring_start in [0, 8]: # Two rings of 8 electrodes
                for i in range(8):
                    curr_idx = ring_start + i
                    next_idx = ring_start + (i + direction) % 8
                    
                    # Mix: New = (1-alpha)*Current + alpha*Next
                    out[curr_idx] = (1 - shift_mag) * x[curr_idx] + shift_mag * x[next_idx]
                    
                    # Also mix corresponding velocity channel??
                    # Velocity is derived from EMG, so ideally we recompute.
                    # But if we just mix velocity linearly it's approx correct (derivative is linear operator)
                    if x.shape[0] >= 32:
                        curr_v = curr_idx + 16
                        next_v = next_idx + 16
                        out[curr_v] = (1 - shift_mag) * x[curr_v] + shift_mag * x[next_v]
            
            return out
        return x
