"""
Data Loader for GRABMyo (PhysioNet) EMG Gesture Dataset.
https://physionet.org/content/grabmyo/1.0.2/
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import wfdb
from typing import Tuple, List, Optional

# GRABMyo has 16 gestures (index 1-16) + Rest (index 17 in some naming)
# Gesture channels: Forearm = 0-7, 8-15 (two rings of 8)
FOREARM_CHANNELS = list(range(16)) # Columns 0-15 (rings 1 & 2)

class GRABMyoDataset(Dataset):
    """
    PyTorch Dataset for GRABMyo.
    
    Args:
        data_root: Path to the downloaded GRABMyo directory (containing Session1, Session2, Session3).
        sessions: List of session indices to include, e.g., [1, 2] for Session1 and Session2.
        subjects: List of subject indices or None for all (1-43).
        gestures: List of gesture indices or None for all (1-16).
        channels: List of channel indices to use. Default is forearm channels (0-15).
        segment_len: Fixed length of each sample (in samples). Default 2048 (1 sec at 2048 Hz).
        normalize: Whether to Z-score normalize each sample.
    """
    def __init__(
        self,
        data_root: str,
        sessions: List[int] = [1],
        subjects: Optional[List[int]] = None,
        gestures: Optional[List[int]] = None,
        channels: List[int] = FOREARM_CHANNELS,
        segment_len: int = 2048,
        normalize: bool = True
    ):
        self.data_root = Path(data_root)
        self.sessions = sessions
        self.subjects = subjects if subjects else list(range(1, 44))
        self.gestures = gestures if gestures else list(range(1, 17))
        self.channels = channels
        self.segment_len = segment_len
        self.normalize = normalize
        
        # Build file index
        self.samples = []
        self._index_files()
    
    def _index_files(self):
        """Build a list of (file_path_stem, gesture_label) tuples."""
        for session in self.sessions:
            session_dir = self.data_root / f"Session{session}"
            if not session_dir.exists():
                print(f"[WARN] Session{session} not found at {session_dir}")
                continue
            
            for subj in self.subjects:
                subj_dir = session_dir / f"session{session}_subject{subj}"
                if not subj_dir.exists():
                    continue
                
                for gest in self.gestures:
                    # Each gesture has 7 trials
                    for trial in range(1, 8):
                        # File naming: session{i}_subject{j}_gesture{k}_trial{t}.dat
                        fname = f"session{session}_subject{subj}_gesture{gest}_trial{trial}"
                        fpath = subj_dir / fname
                        
                        # Check if .dat exists
                        if (subj_dir / (fname + ".dat")).exists():
                            self.samples.append({
                                "path": str(fpath),
                                "session": session,
                                "subject": subj,
                                "gesture": gest,
                                "trial": trial
                            })
        
        print(f"[GRABMyoDataset] Indexed {len(self.samples)} samples.")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        info = self.samples[idx]
        
        # Read WFDB record
        record = wfdb.rdrecord(info["path"])
        sig = record.p_signal # Shape: (10240, 32) - 5 sec @ 2048 Hz, 32 channels
        
        # Select forearm channels
        sig = sig[:, self.channels] # (10240, n_channels)
        
        # Segment: take the middle segment_len samples (skip first/last edge)
        n_samples = sig.shape[0]
        if n_samples >= self.segment_len:
            start = (n_samples - self.segment_len) // 2
            sig = sig[start:start + self.segment_len, :]
        else:
            # Pad if too short (shouldn't happen with 5s recordings)
            pad_len = self.segment_len - n_samples
            sig = np.pad(sig, ((0, pad_len), (0, 0)), mode='constant')
        
        # Transpose to (Channels, Time)
        sig = sig.T.astype(np.float32) # (n_channels, segment_len)
        
        # Normalize per channel
        if self.normalize:
            mean = sig.mean(axis=1, keepdims=True)
            std = sig.std(axis=1, keepdims=True) + 1e-6
            sig = (sig - mean) / std
        
        # Compute velocity (first derivative) per channel
        velocity = np.gradient(sig, axis=1).astype(np.float32)
        if self.normalize:
            v_mean = velocity.mean(axis=1, keepdims=True)
            v_std = velocity.std(axis=1, keepdims=True) + 1e-6
            velocity = (velocity - v_mean) / v_std
        
        # Stack: [original_channels, velocity_channels] -> (2*n_channels, segment_len)
        sig_dual = np.concatenate([sig, velocity], axis=0)
        
        # Label: 0-indexed for CrossEntropyLoss
        label = info["gesture"] - 1 # 0-15
        
        return torch.from_numpy(sig_dual), label


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_grabmyo_root>")
        sys.exit(1)
    
    data_root = sys.argv[1]
    ds = GRABMyoDataset(data_root, sessions=[1], subjects=[1, 2])
    
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Sample 0: Signal shape = {x.shape}, Label = {y}")
    else:
        print("No samples found. Check data path.")
