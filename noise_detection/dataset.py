import torch
from torch.utils.data import IterableDataset
import numpy as np
import sys
import os

# Ensure we can import the simulator from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from eng_surface_sim import simulate, SimConfig, ArtifactGenerator
except ImportError:
    # If running from root, direct import might work
    from eng_surface_sim import simulate, SimConfig, ArtifactGenerator

class ENGSimulationDataset(IterableDataset):
    """
    Infinite stream of simulated ENG signals + Noise Labels.
    """
    def __init__(self, fs=8000, duration_s=1.0, normalize=True, min_noise_ratio=0.02):
        self.fs = fs
        self.duration_s = duration_s
        self.normalize = normalize
        self.min_noise_ratio = min_noise_ratio  # Minimum fraction of samples that should be noise
        
        # We create a config template
        self.cfg = SimConfig(fs=fs, duration_s=duration_s)
        # Randomize parameters slightly per sample? 
        # For now, let's keep default params but allow random noise generation.

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = 123
        if worker_info is not None:
            seed = worker_info.id + worker_info.seed

        while True:
            # Update seed for randomness
            seed += 1
            self.cfg.seed = seed
            
            # Run simulation
            # We want to ensure specific artifacts appear sometimes.
            # The current simulator uses probability.
            
            # NOTE: simulate now returns:
            # t, raw, proc, meta, events, artifact_labels, clean, noise_only
            try:
                t, raw, proc, meta, events, artifact_labels, clean, noise_only = simulate(self.cfg)
            except ValueError: 
                 # Handle older API if file wasn't reloaded properly (unlikely in fresh run)
                 t, raw, proc, meta, events, artifact_labels = simulate(self.cfg)
            
            # Create Binary Mask (1 = Noise, 0 = Clean)
            mask = np.zeros_like(raw, dtype=np.float32)
            for l in artifact_labels:
                if l['type_id'] > 0: # 0 is Clean
                    # Convert time to indices
                    start_idx = int(l['start_s'] * self.fs)
                    end_idx = int(l['end_s'] * self.fs)
                    # Clip
                    start_idx = max(0, start_idx)
                    end_idx = min(len(mask), end_idx)
                    mask[start_idx:end_idx] = 1.0
            
            # Check minimum noise ratio - regenerate if too clean
            noise_ratio = mask.sum() / len(mask)
            if noise_ratio < self.min_noise_ratio:
                # Too clean, skip and regenerate with next seed
                continue
            
            # Normalization (Z-score)
            sig = raw.astype(np.float32)
            if self.normalize and len(sig) > 0:
                mean = np.mean(sig)
                std = np.std(sig)
                if std > 1e-6:
                    sig = (sig - mean) / std
                else:
                    sig = sig - mean
            
            # Compute velocity (first derivative)
            velocity = np.gradient(sig)
            if self.normalize and np.std(velocity) > 1e-6:
                velocity = (velocity - np.mean(velocity)) / np.std(velocity)
            
            # Stack into 2 channels: [amplitude, velocity]
            sig_2ch = np.stack([sig, velocity], axis=0).astype(np.float32)  # (2, L)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            
            yield torch.from_numpy(sig_2ch), mask_tensor
