
import sys
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset
from gesture_recognition.models.dl_models import get_model

def get_features(model, loader, device):
    """Extract features from the TCN backbone (before the final FC layer)."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            
            # Forward pass up to GAP
            # TCN structure: x -> network -> min_pool -> squeeze -> fc
            y = model.network(inputs)
            features = model.min_pool(y).squeeze(-1)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            
    return torch.cat(all_features), torch.cat(all_labels)

def predict_nearest_centroid(prototypes, features):
    """
    Classify features based on Cosine Similarity to prototypes.
    """
    # Normalize features and prototypes
    # features: (N, D)
    # prototypes: (K, D)
    
    f_norm = torch.nn.functional.normalize(features, p=2, dim=1)
    p_norm = torch.nn.functional.normalize(prototypes, p=2, dim=1)
    
    # Cosine similarity = dot product of normalized vectors
    # (N, D) @ (D, K) -> (N, K)
    sims = torch.mm(f_norm, p_norm.t())
    
    _, preds = torch.max(sims, dim=1)
    return preds

def main():
    print("Testing Prototype Matching (Nearest Centroid) for Cross-Day Adaptation")
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    ckpt_path = str(Path.home() / "exg_train/checkpoints/ensemble_seed45_fold1_best.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Full Session 3 Data
    print("Loading Session 3 Data...")
    ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[3], 
        subjects=list(range(1, 44)),
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True,
        causal=True
    )
    
    total_len = len(ds)
    calib_size = int(0.1 * total_len) # 10% data for calibration
    test_size = total_len - calib_size
    
    calib_ds = Subset(ds, range(0, calib_size))
    test_ds = Subset(ds, range(calib_size, total_len))
    
    print(f"Split: {calib_size} Calibration samples vs {test_size} Test samples")
    
    # Loaders
    calib_loader = DataLoader(calib_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Load Model
    model = get_model('tcn', 32, 8, device=device.type)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    # 1. Evaluate Zero-shot Baseline (Original FC)
    print("\n[Baseline] Extracting Features & Evaluating Zero-shot...")
    test_feats, test_labels = get_features(model, test_loader, device)
    
    # Verify baseline with original FC
    with torch.no_grad():
        logits = model.fc(test_feats.to(device))
        _, base_preds = torch.max(logits, 1)
        base_acc = (base_preds.cpu() == test_labels).float().mean().item()
    print(f"Baseline Accuracy (Original Head): {base_acc:.4f}")
    
    # 2. Compute Prototypes from Calibration Data
    print("\n[Calibration] Computing Prototypes from 10% data...")
    t0 = time.time()
    calib_feats, calib_labels = get_features(model, calib_loader, device)
    
    num_classes = 8
    feat_dim = calib_feats.shape[1]
    prototypes = torch.zeros(num_classes, feat_dim).to(device)
    
    for c in range(num_classes):
        # Get features for class c
        idx = (calib_labels == c)
        if idx.sum() == 0:
            print(f"Warning: No samples for class {c} in calibration set!")
            continue
        class_feats = calib_feats[idx].to(device)
        prototypes[c] = class_feats.mean(dim=0)
        
    print(f"Prototypes computed in {time.time()-t0:.2f}s")
    
    # 3. Evaluate Nearest Centroid on Test Data
    print("\n[After Calibration] Evaluating Nearest Centroid Classifier...")
    proto_preds = predict_nearest_centroid(prototypes, test_feats.to(device))
    proto_acc = (proto_preds.cpu() == test_labels).float().mean().item()
    
    print(f"Prototype Matching Accuracy: {proto_acc:.4f}")
    print(f"Improvement: {proto_acc - base_acc:+.4f}")
    
    if proto_acc > 0.90:
        print("🚀 SUCCESS: Prototype Matching restores high accuracy!")

if __name__ == "__main__":
    main()
