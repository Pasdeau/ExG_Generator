
import sys
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from gesture_recognition.data_loader_v2 import GRABMyoWindowDataset
from gesture_recognition.models.dl_models import get_model

def extract_features(models, loader, device):
    """Extract concatenated features from all models in the ensemble."""
    for m in models: m.eval()
    
    all_feats = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            
            # Get features from each model
            batch_feats = []
            for m in models:
                # TCN feature extraction: network -> min_pool -> squeeze
                y = m.network(inputs)
                feats = m.min_pool(y).squeeze(-1) # (B, F)
                batch_feats.append(feats)
                
            # Concatenate features from all models
            # Shape: (B, F * num_models)
            concat_feats = torch.cat(batch_feats, dim=1)
            
            all_feats.append(concat_feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.vstack(all_feats), np.array(all_labels)

def evaluate_subject_calibration(subject_id, models, device, data_root):
    # Load ONLY this subject's Session 3
    ds = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[3],
        subjects=[subject_id],
        gestures=list(range(10, 18)),
        window_len_ms=200, hop_ms=50,
        apply_car=True, apply_bandpass=True, normalize=True,
        preload=True, causal=True
    )
    
    total = len(ds)
    if total < 100:
        return None, None
        
    # Split: First 15% for Calibration (~1-2 mins), Rest for Test
    calib_size = int(0.15 * total)
    
    # We must ensure class balance or at least presence of all classes in Calib?
    # Since dataset is ordered by Gesture, taking first 15% linearly gives ONLY Gesture 1!
    # CRITICAL FIX: We need stratifed split or shuffle split?
    # BUT: In real life, user provides data sequentially. "Do gesture 1... Done. Do gesture 2... Done."
    # So we should assume we get N samples per gesture.
    
    # GRABMyo structure: random access by index does NOT guarantee order, 
    # but _index_windows usually appends sequentially.
    # To simulate "Calibration Phase", we should assume the user performs each gesture.
    # Let's use Random Split to simulate "Collecting data for all gestures".
    
    indices = np.arange(total)
    # Stratified shuffle split simulation
    from sklearn.model_selection import train_test_split
    # We extract ALL labels first to split
    # This is slow if we iterate. But we can peek at ds.windows
    all_labels = [ds.windows[i]['gesture'] for i in range(total)]
    
    train_idx, test_idx = train_test_split(indices, train_size=0.15, stratify=all_labels, random_state=42)
    
    calib_loader = DataLoader(Subset(ds, train_idx), batch_size=256, shuffle=True)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=256, shuffle=False)
    
    # 1. Extract Features
    X_calib, y_calib = extract_features(models, calib_loader, device)
    X_test, y_test = extract_features(models, test_loader, device)
    
    # 2. Train Linear Classifier (Personalized Head)
    # C=1.0 is default. Stronger regularization (lower C) might help small data.
    clf = LogisticRegression(C=1.0, solver='liblinear', max_iter=200)
    clf.fit(X_calib, y_calib)
    
    # 3. Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Contrast with Zero-shot (using original heads)?
    # That would require voting logic again. Skipping for speed.
    
    return acc, len(y_test)

def main():
    print("Evaluating Ensemble + Personalized Calibration (Linear Probe)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = str(Path.home() / "exg_train/physionet.org/files/grabmyo/1.0.2")
    ckpt_dir = Path.home() / "exg_train/checkpoints"
    
    # Load Ensemble Models
    seeds = [42, 43, 44, 45, 46]
    models = []
    print(f"Loading {len(seeds)} models...")
    for seed in seeds:
        model = get_model('tcn', 32, 8, device=device.type)
        path = ckpt_dir / f"ensemble_seed{seed}_fold1_best.pth"
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        models.append(model)
        
    accuracies = []
    
    # Iterate over all 43 subjects
    print(f"{'Subject':<10} | {'Acc':<10}")
    print("-" * 25)
    
    for subj in range(1, 44):
        acc, n_samples = evaluate_subject_calibration(subj, models, device, data_root)
        if acc is not None:
            print(f"{subj:<10} | {acc:.4f}")
            accuracies.append(acc)
            
    mean_acc = np.mean(accuracies)
    print("=" * 25)
    print(f"Mean Calibration Accuracy: {mean_acc:.4f}")
    
    if mean_acc > 0.95:
         print("🚀 SUCCESS: Personalized Calibration hits >95%!")
    elif mean_acc > 0.90:
         print("✅ SUCCESS: Personalized Calibration hits >90%!")

if __name__ == "__main__":
    main()
