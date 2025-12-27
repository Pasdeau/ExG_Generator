
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset
from gesture_recognition.models.dl_models import get_model

def evaluate_ensemble(models, loader, device):
    for m in models:
        m.eval()
        
    all_preds = []
    all_labels = []
    
    print(f"Evaluating Ensemble with {len(models)} models...")
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            labels = labels.long().to(device)
            
            # Ensemble Voting with TTA (Test-Time Augmentation)
            # Shifts: [-1, 0, 1] corresponding to -45, 0, +45 degrees
            shifts = [-1, 0, 1]
            ensemble_outputs = torch.zeros(inputs.size(0), 8).to(device)
            
            for shift in shifts:
                # Create rotated input COPY
                x_aug = inputs.clone()
                
                if shift != 0:
                    # Apply roll to all 4 component groups (Ring1/2 Signal, Ring1/2 Velocity)
                    # Groups: 0-7, 8-15, 16-23, 24-31
                    # axis 1 is Channel dimension
                    x_aug[:, 0:8]   = torch.roll(x_aug[:, 0:8], shifts=shift, dims=1)
                    x_aug[:, 8:16]  = torch.roll(x_aug[:, 8:16], shifts=shift, dims=1)
                    x_aug[:, 16:24] = torch.roll(x_aug[:, 16:24], shifts=shift, dims=1)
                    x_aug[:, 24:32] = torch.roll(x_aug[:, 24:32], shifts=shift, dims=1)
                
                # Pass through all models
                for m in models:
                    outputs = m(x_aug)
                    probs = torch.softmax(outputs, dim=1)
                    ensemble_outputs += probs
            
            # Average probabilities (5 models * 3 TTA views = 15 voters)
            ensemble_outputs /= (len(models) * len(shifts))
            
            _, preds = torch.max(ensemble_outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(all_labels, all_preds)
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    ckpt_dir = Path.home() / "exg_train/checkpoints"
    
    # 1. Load Models
    seed_list = [int(s) for s in args.seeds.split(',')]
    models = []
    
    print("Loading models...")
    for seed in seed_list:
        model = get_model('tcn', n_channels=32, n_classes=8, device=device.type)
        ckpt_path = ckpt_dir / f"ensemble_seed{seed}_fold1_best.pth"
        
        if not ckpt_path.exists(): 
            # Try remote path structure if running locally? No, this runs remote.
            # Check if file exists
            print(f"Error: Checkpoint not found: {ckpt_path}")
            continue
            
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        models.append(model)
        print(f"Loaded Model Seed {seed}")
        
    if not models:
        print("No models loaded!")
        return

    # 2. Load Test Data (Session 3, All Subjects)
    # Consistent with train_ensemble.py Fold 1 Test
    print("Loading Test Dataset (Session 3, All 43 Subjects)...")
    subjects = list(range(1, 44))
    test_ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[3], # Session 3 (Cross-Day)
        subjects=subjects,
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True, 
        causal=True,
        augment_rotation=False 
    )
    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Evaluate
    t0 = time.time()
    acc = evaluate_ensemble(models, test_loader, device)
    dt = time.time() - t0
    
    print(f"\n{'='*30}")
    print(f"Ensemble Result (Seeds {args.seeds})")
    print(f"Test Accuracy (Cross-Day): {acc*100:.2f}%")
    print(f"Time Elapsed: {dt:.2f}s")
    print(f"{'='*30}")

if __name__ == "__main__":
    main()
