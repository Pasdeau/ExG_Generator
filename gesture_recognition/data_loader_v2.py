"""
Enhanced Data Loader for GRABMyo (PhysioNet) EMG Gesture Dataset.
https://physionet.org/content/grabmyo/1.0.2/

V2.0 Features:
- Common Average Re-reference (CAR)
- Bandpass filtering (20-450 Hz)
- Sliding windows (200ms / 150ms overlap)
- Cross-day evaluation support
- Per-session normalization
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import wfdb
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import KFold
import preprocessing


# GRABMyo channel layout (based on header inspection)
# F1-F16: Forearm channels 0-15 (two rings of 8)
# W1-W12: Wrist channels 16-27
# U1-U4: Unused channels 28-31
FOREARM_RING1 = list(range(0, 8))   # F1-F8
FOREARM_RING2 = list(range(8, 16))  # F9-F16
FOREARM_CHANNELS = FOREARM_RING1 + FOREARM_RING2


class GRABMyoWindowDataset(Dataset):
    """
    PyTorch Dataset for GRABMyo with sliding windows.
    
    Args:
        data_root: Path to GRABMyo directory (containing Session1, Session2, Session3).
        sessions: List of session indices to include, e.g., [1, 2] for Session1 and Session2.
        subjects: List of subject indices or None for all (1-43).
        gestures: List of gesture indices (10-17 in files, corresponding to gesture 1-16 + rest).
        channels: List of channel indices to use. Default is forearm channels (0-15).
        window_len_ms: Window length in milliseconds (default 200ms).
        hop_ms: Hop size in milliseconds (default 50ms, i.e., 150ms overlap).
        fs: Sampling rate in Hz.
        apply_car: Whether to apply Common Average Re-reference.
        apply_bandpass: Whether to apply bandpass filter.
        apply_notch: Whether to apply notch filter.
        normalize: Whether to Z-score normalize each window.
        transform: Optional transform to apply to windows.
        session_stats: Optional pre-computed normalization stats for cross-day testing.
    """
    def __init__(
        self,
        data_root: str,
        sessions: List[int] = [1],
        subjects: Optional[List[int]] = None,
        gestures: Optional[List[int]] = None,
        channels: List[int] = FOREARM_CHANNELS,
        window_len_ms: float = 200.0,
        hop_ms: float = 50.0,
        fs: int = 2048,
        apply_car: bool = True,
        apply_bandpass: bool = True,
        apply_notch: bool = False,
        normalize: bool = True,
        transform = None,
        session_stats: Optional[Dict] = None,
        preload: bool = False
    ):
        self.data_root = Path(data_root)
        self.sessions = sessions
        self.subjects = subjects if subjects else list(range(1, 44))
        # Default gestures: 10-17 in filenames (corresponding to 8 gestures including rest)
        self.gestures = gestures if gestures else list(range(10, 18))
        self.channels = channels
        self.fs = fs
        self.window_len = int(window_len_ms / 1000 * fs)
        self.hop = int(hop_ms / 1000 * fs)
        self.apply_car = apply_car
        self.apply_bandpass = apply_bandpass
        self.apply_notch = apply_notch
        self.normalize = normalize
        self.transform = transform
        self.session_stats = session_stats if session_stats else {}
        self.preload = preload
        self.trial_cache = {}
        
        # Build window index
        self.windows = []
        self._index_windows()
        
        if self.preload:
            print(f"[GRABMyoWindowDataset] Preloading {len(self.trial_cache)} trials into RAM...")
    
    def _index_windows(self):
        """Build a list of (trial_path, window_idx, gesture_label, session) tuples."""
        for session in self.sessions:
            session_dir = self.data_root / f"Session{session}"
            if not session_dir.exists():
                print(f"[WARN] Session{session} not found at {session_dir}")
                continue
            
            for subj in self.subjects:
                subj_dir = session_dir / f"session{session}_participant{subj}"
                if not subj_dir.exists():
                    continue
                
                for gest in self.gestures:
                    # Each gesture has 7 trials
                    for trial in range(1, 8):
                        fname = f"session{session}_participant{subj}_gesture{gest}_trial{trial}"
                        fpath = subj_dir / fname
                        
                        # Check if .dat exists
                        dat_path = subj_dir / (fname + ".dat")
                        if dat_path.exists():
                            # Cache data if preloading
                            str_path = str(fpath)
                            if self.preload and str_path not in self.trial_cache:
                                try:
                                    record = wfdb.rdrecord(str_path)
                                    self.trial_cache[str_path] = record.p_signal.astype(np.float32)
                                except Exception as e:
                                    print(f"[WARN] Failed to read {str_path}: {e}")
                                    continue

                            # Compute number of windows for this 5-sec trial
                            trial_len = int(5.0 * self.fs)  # 5 seconds @ 2048 Hz = 10240 samples
                            n_windows = (trial_len - self.window_len) // self.hop + 1
                            
                            for win_idx in range(n_windows):
                                self.windows.append({
                                    "path": str_path,
                                    "window_idx": win_idx,
                                    "session": session,
                                    "subject": subj,
                                    "gesture": gest,
                                    "trial": trial
                                })
        
        print(f"[GRABMyoWindowDataset] Indexed {len(self.windows)} windows from {len(self.sessions)} session(s).")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        info = self.windows[idx]
        
        # Read signal
        if self.preload:
            sig = self.trial_cache[info["path"]]
        else:
            record = wfdb.rdrecord(info["path"])
            sig = record.p_signal  # Shape: (10240, 32)
        
        # Preprocessing pipeline
        if self.apply_car or self.apply_bandpass or self.apply_notch:
            sig = preprocessing.preprocess_trial(
                sig, 
                fs=self.fs, 
                apply_notch=self.apply_notch
            )
        else:
            # Just select forearm channels
            sig = sig[:, self.channels]
        
        # Extract window
        start = info["window_idx"] * self.hop
        end = start + self.window_len
        window = sig[start:end, :]  # Shape: (window_len, n_channels)
        
        # Normalize
        if self.normalize:
            session_key = f"session{info['session']}"
            stats = self.session_stats.get(session_key, None)
            window, _ = preprocessing.normalize_signal(window, method='zscore', stats=stats)
        
        # Compute velocity (first derivative)
        velocity = preprocessing.compute_velocity(window)
        if self.normalize:
            velocity, _ = preprocessing.normalize_signal(velocity, method='zscore')
        
        # Transpose to (channels, time) and stack [signal, velocity]
        window = window.T.astype(np.float32)  # (n_channels, window_len)
        velocity = velocity.T.astype(np.float32)
        sig_dual = np.concatenate([window, velocity], axis=0)  # (2*n_channels, window_len)
        
        # Label: gesture index (0-indexed for CrossEntropyLoss)
        # GRABMyo files use gesture 10-17, we map to 0-7
        label = info["gesture"] - 10
        
        sig_tensor = torch.from_numpy(sig_dual)
        
        if self.transform:
            sig_tensor = self.transform(sig_tensor)
        
        return sig_tensor, label


def create_cross_day_folds(data_root: str, 
                           subjects: Optional[List[int]] = None,
                           gestures: Optional[List[int]] = None) -> List[Tuple[List[int], int]]:
    """
    Create 3-fold cross-day splits.
    
    Returns:
        List of (train_sessions, test_session) tuples:
        - Fold 0: train=[1,2], test=3
        - Fold 1: train=[1,3], test=2
        - Fold 2: train=[2,3], test=1
    """
    folds = [
        ([1, 2], 3),
        ([1, 3], 2),
        ([2, 3], 1)
    ]
    return folds


def compute_session_stats(data_root: str,
                          sessions: List[int],
                          subjects: Optional[List[int]] = None,
                          gestures: Optional[List[int]] = None,
                          n_samples: int = 1000) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute per-session normalization statistics by sampling windows.
    
    Args:
        data_root: Path to GRABMyo dataset
        sessions: List of sessions to compute stats for
        subjects: Subject list
        gestures: Gesture list
        n_samples: Number of windows to sample per session
        
    Returns:
        Dict of {f"session{i}": (mean, std)} for each session
    """
    stats_dict = {}
    
    for session in sessions:
        # Create temporary dataset for this session only
        temp_ds = GRABMyoWindowDataset(
            data_root=data_root,
            sessions=[session],
            subjects=subjects,
            gestures=gestures,
            apply_car=True,
            apply_bandpass=True,
            apply_notch=False,
            normalize=False  # Don't normalize yet
        )
        
        if len(temp_ds) == 0:
            print(f"[WARN] No data for session {session}")
            continue
        
        # Sample windows
        sample_indices = np.random.choice(len(temp_ds), min(n_samples, len(temp_ds)), replace=False)
        
        all_windows = []
        for idx in sample_indices:
            # Get raw window (without normalization)
            record_info = temp_ds.windows[idx]
            record = wfdb.rdrecord(record_info["path"])
            sig = record.p_signal
            
            # Preprocess
            sig = preprocessing.preprocess_trial(sig, fs=temp_ds.fs, apply_notch=False)
            
            # Extract window
            start = record_info["window_idx"] * temp_ds.hop
            end = start + temp_ds.window_len
            window = sig[start:end, :]
            
            all_windows.append(window)
        
        # Compute stats across all windows
        all_windows = np.array(all_windows)  # (n_samples, window_len, n_channels)
        all_windows = all_windows.reshape(-1, all_windows.shape[-1])  # (n_samples*window_len, n_channels)
        
        mean = all_windows.mean(axis=0, keepdims=True)
        std = all_windows.std(axis=0, keepdims=True) + 1e-6
        
        stats_dict[f"session{session}"] = (mean, std)
        print(f"[Session {session}] Computed stats from {len(sample_indices)} windows")
    
    return stats_dict


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader_v2.py <path_to_grabmyo_root>")
        print("Example: python data_loader_v2.py ~/exg_train/physionet.org/files/grabmyo/1.0.2")
        sys.exit(1)
    
    data_root = sys.argv[1]
    
    print("=" * 60)
    print("Testing GRABMyoWindowDataset (V2.0)")
    print("=" * 60)
    
    # Test 1: Single session, single subject
    print("\n[Test 1] Single session, single subject")
    ds = GRABMyoWindowDataset(
        data_root, 
        sessions=[1], 
        subjects=[1],
        gestures=[10, 11],  # Just 2 gestures for quick test
        window_len_ms=200.0,
        hop_ms=50.0
    )
    
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Sample 0: Signal shape = {x.shape}, Label = {y}")
        print(f"Expected shape: (32, 409) for 16 channels + 16 velocity, 409 samples @ 200ms")
    else:
        print("No samples found. Check data path.")
    
    # Test 2: Cross-day folds
    print("\n[Test 2] Cross-day evaluation folds")
    folds = create_cross_day_folds(data_root)
    for i, (train_sess, test_sess) in enumerate(folds):
        print(f"Fold {i}: Train on sessions {train_sess}, Test on session {test_sess}")
    
    # Test 3: Session stats computation
    print("\n[Test 3] Computing per-session statistics")
    stats = compute_session_stats(
        data_root, 
        sessions=[1], 
        subjects=[1, 2],  # Just 2 subjects for quick test
        gestures=[10, 11],
        n_samples=50
    )
    for sess_key, (mean, std) in stats.items():
        print(f"{sess_key}: mean shape={mean.shape}, std shape={std.shape}")
    
    print("\n✅ All tests completed!")
