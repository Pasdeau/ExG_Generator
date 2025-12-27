
import sys
import argparse
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

def load_ensemble(device, ckpt_dir):
    seeds = [42, 43, 44, 45, 46]
    models = []
    print(f"Loading {len(seeds)} Ensemble Models...", end=" ")
    for seed in seeds:
        model = get_model('tcn', 32, 8, device=device.type)
        path = ckpt_dir / f"ensemble_seed{seed}_fold1_best.pth"
        if not path.exists():
            print(f"\n[Error] Checkpoint not found: {path}")
            sys.exit(1)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    print("Done.")
    return models

def extract_features(models, loader, device):
    all_feats = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.float().to(device)
            
            # Feature Extraction: Concat features from all 5 models
            batch_feats = []
            for m in models:
                y = m.network(inputs)
                feats = m.min_pool(y).squeeze(-1) # (B, 128)
                batch_feats.append(feats)
                
            concat_feats = torch.cat(batch_feats, dim=1) # (B, 640)
            all_feats.append(concat_feats.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.vstack(all_feats), np.array(all_labels)

def verify_subject(subject_id, data_root, device, models):
    print(f"\n[{'='*10} Verifying Subject {subject_id} {'='*10}]")
    
    # Load Session 3 (Cross-Day/New User Scenario)
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
    if total == 0:
        print("No data found.")
        return

    # Simulate 1-Minute Calibration (15% of data)
    calib_size = int(0.15 * total)
    test_size = total - calib_size
    
    # Split
    indices = np.arange(total)
    # Simple time-based split simulating "First minute of usage"
    # Note: In a real demo, we'd ensure class balance. Here we assume the dataset is somewhat shuffled or composed of trials.
    # Actually GRABMyo is sequential. Taking first 15% might miss gestures.
    # So we use a Stratified Split to simulate "User performs EACH gesture".
    from sklearn.model_selection import train_test_split
    all_labels = [ds.windows[i]['gesture'] for i in range(total)]
    train_idx, test_idx = train_test_split(indices, train_size=0.15, stratify=all_labels, random_state=42)
    
    calib_loader = DataLoader(Subset(ds, train_idx), batch_size=256, shuffle=True)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=256, shuffle=False)
    
    print(f"Data: {total} samples. Calibration: {len(train_idx)} (~45s). Test: {len(test_idx)}.")
    
    # 1. Feature Extraction
    print("Extracting features...", end=" ")
    X_calib, y_calib = extract_features(models, calib_loader, device)
    X_test, y_test = extract_features(models, test_loader, device)
    print("Done.")
    
    # 2. Zero-shot Baseline (Voting)
    # skipped for speed, focusing on calibrated result
    
    # 3. Calibration (Linear Probe)
    print("Training Personal Classifier (Linear Probe)...", end=" ")
    clf = LogisticRegression(C=1.0, solver='liblinear', max_iter=200)
    clf.fit(X_calib, y_calib)
    print("Done.")
    
    # 4. Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n>>> Final Accuracy for Subject {subject_id}: {acc*100:.2f}% <<<")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Final Product Verification")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to verify (1-43)")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    root_dir = Path.home() / "exg_train"
    data_root = root_dir / "physionet.org/files/grabmyo/1.0.2"
    ckpt_dir = root_dir / "checkpoints"
    
    models = load_ensemble(device, ckpt_dir)
    verify_subject(args.subject, data_root, device, models)

if __name__ == "__main__":
    main()
