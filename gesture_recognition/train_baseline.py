"""
Baseline Training Script for EMG Gesture Classification.

Uses traditional Hudgins time-domain features + classical ML classifiers:
- LDA (Linear Discriminant Analysis)
- SVM (Support Vector Machine with RBF kernel)
- Random Forest
- XGBoost

Evaluation modes:
- Intra-day: 5-fold CV on single session
- Cross-day: 3-fold CV (train on 2 sessions, test on 1)
"""

import numpy as np
import argparse
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote servers
import matplotlib.pyplot as plt
import seaborn as sns
import sys

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed, skipping XGBoost classifier")
    sys.stdout.flush()

import sys
sys.path.append(str(Path(__file__).parent))

from data_loader_v2 import GRABMyoWindowDataset, create_cross_day_folds
import features
import preprocessing


def extract_features_from_loader(dataset):
    """
    Extract Hudgins features from all windows in dataset.
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
    """
    print(f"Extracting features from {len(dataset)} windows...")
    sys.stdout.flush()
    
    X_list = []
    y_list = []
    
    for idx in range(len(dataset)):
        # Get window info
        info = dataset.windows[idx]
        
        # Load and preprocess signal
        import wfdb
        record = wfdb.rdrecord(info["path"])
        sig = record.p_signal
        
        # Preprocess
        sig = preprocessing.preprocess_trial(sig, fs=dataset.fs, apply_notch=False)
        
        # Extract window
        start = info["window_idx"] * dataset.hop
        end = start + dataset.window_len
        window = sig[start:end, :]
        
        # Normalize
        if dataset.normalize:
            window, _ = preprocessing.normalize_signal(window)
        
        # Extract Hudgins features
        feats = features.extract_hudgins_features(window, ar_order=4)
        
        # Label (gesture 10-17 -> 0-7)
        label = info["gesture"] - 10
        
        X_list.append(feats)
        y_list.append(label)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} windows...")
            sys.stdout.flush()
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Feature extraction complete: X.shape={X.shape}, y.shape={y.shape}")
    sys.stdout.flush()
    return X, y


def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def train_intra_day(data_root, session=1, n_folds=5, n_subjects=5):
    """
    Intra-day evaluation: 5-fold CV on single session.
    """
    print(f"\n{'=' * 60}")
    print(f"INTRA-DAY EVALUATION: Session {session}, {n_folds}-fold CV")
    print(f"{'=' * 60}\n")
    
    # Load dataset
    dataset = GRABMyoWindowDataset(
        data_root=data_root,
        sessions=[session],
        subjects=list(range(1, n_subjects + 1)),  # Limit subjects for quick test
        gestures=list(range(10, 18)),  # All 8 gestures
        window_len_ms=200.0,
        hop_ms=50.0,
        apply_car=True,
        apply_bandpass=True,
        normalize=True
    )
    
    # Extract features
    X, y = extract_features_from_loader(dataset)
    
    # K-Fold CV
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if HAS_XGBOOST:
        classifiers['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    results = {name: [] for name in classifiers.keys()}
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)
            print(f"  {name}: {acc * 100:.2f}%")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("INTRA-DAY RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for name, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{name:10s}: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    
    return results


def train_cross_day(data_root, n_subjects=5):
    """
    Cross-day evaluation: 3-fold (train on 2 sessions, test on 1).
    """
    print(f"\n{'=' * 60}")
    print(f"CROSS-DAY EVALUATION: 3-fold (train 2 sessions, test 1)")
    print(f"{'=' * 60}\n")
    
    folds = create_cross_day_folds(data_root)
    
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale'),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if HAS_XGBOOST:
        classifiers['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    
    results = {name: [] for name in classifiers.keys()}
    
    for fold_idx, (train_sessions, test_session) in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/3: Train on {train_sessions}, Test on {test_session} ---")
        
        # Load train dataset
        train_dataset = GRABMyoWindowDataset(
            data_root=data_root,
            sessions=train_sessions,
            subjects=list(range(1, n_subjects + 1)),
            gestures=list(range(10, 18)),
            window_len_ms=200.0,
            hop_ms=50.0,
            apply_car=True,
            apply_bandpass=True,
            normalize=True
        )
        
        # Load test dataset
        test_dataset = GRABMyoWindowDataset(
            data_root=data_root,
            sessions=[test_session],
            subjects=list(range(1, n_subjects + 1)),
            gestures=list(range(10, 18)),
            window_len_ms=200.0,
            hop_ms=50.0,
            apply_car=True,
            apply_bandpass=True,
            normalize=True
        )
        
        # Extract features
        X_train, y_train = extract_features_from_loader(train_dataset)
        X_test, y_test = extract_features_from_loader(test_dataset)
        
        # Train and evaluate
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)
            print(f"  {name}: {acc * 100:.2f}%")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("CROSS-DAY RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for name, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"{name:10s}: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    
    return results


def main(args):
    print("\n" + "=" * 60)
    print("EMG GESTURE RECOGNITION - BASELINE EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.data_root}")
    print(f"Evaluation mode: {args.mode}")
    print(f"Number of subjects: {args.n_subjects}")
    print("=" * 60)
    
    if args.mode == 'intra-day':
        results = train_intra_day(
            args.data_root, 
            session=args.session,
            n_folds=args.n_folds,
            n_subjects=args.n_subjects
        )
    elif args.mode == 'cross-day':
        results = train_cross_day(
            args.data_root,
            n_subjects=args.n_subjects
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print("\n✅ Baseline evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline EMG Gesture Classification")
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to GRABMyo dataset root")
    parser.add_argument("--mode", type=str, default="intra-day",
                       choices=['intra-day', 'cross-day'],
                       help="Evaluation mode")
    parser.add_argument("--session", type=int, default=1,
                       help="Session index for intra-day evaluation (1-3)")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds for intra-day CV")
    parser.add_argument("--n_subjects", type=int, default=5,
                       help="Number of subjects to use (for quick testing)")
    
    args = parser.parse_args()
    main(args)
