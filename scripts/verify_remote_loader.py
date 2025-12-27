
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset

def verify_loader():
    print("Verifying GRABMyoWindowDataset (Remote Mode)")
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    
    # Init with settings matching LOSO job
    ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[1],
        subjects=[1], # Just one subject for speed
        gestures=list(range(10, 12)), # 2 gestures
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=False, # TEST TARGET
        causal=True,
        augment_rotation=True
    )
    
    print(f"Dataset length: {len(ds)}")
    if len(ds) == 0:
        print("Dataset empty! Check paths.")
        return

    # Check first 10 samples
    for i in range(10):
        x, y = ds[i]
        
        # Check Shape
        if x.shape != (32, 409):
            print(f"[FAIL] Sample {i} shape mismatch: {x.shape}")
            continue
            
        # Check NaNs
        if torch.isnan(x).any():
            print(f"[FAIL] Sample {i} contains NaNs!")
            continue
            
        # Check Zeros (pure silence)
        if x.abs().sum() < 1e-6:
            print(f"[FAIL] Sample {i} is all zeros!")
            continue
            
        # Check Label
        if y < 0 or y > 7:
            print(f"[FAIL] Sample {i} label invalid: {y}")
            continue
            
        print(f"[PASS] Sample {i}: Mean={x.mean():.4f}, Std={x.std():.4f}, Label={y}")

    print("Verification Complete.")

if __name__ == "__main__":
    verify_loader()
