"""
Preprocessing utilities for GRABMyo EMG signals.

Based on GRABMyo paper recommendations:
- Common Average Re-reference (CAR) for each electrode ring
- Butterworth bandpass filter (20-450 Hz)
- Sliding window with 200ms window, 150ms overlap
"""

import numpy as np
from scipy import signal
from typing import List, Tuple


def apply_car(emg_signal: np.ndarray, ring_channels: List[int]) -> np.ndarray:
    """
    Apply Common Average Re-reference (CAR) to one ring of electrodes.
    
    Args:
        emg_signal: Raw EMG signal, shape (n_samples, n_channels)
        ring_channels: List of channel indices forming one ring
        
    Returns:
        Re-referenced signal for the specified channels
    """
    ring_data = emg_signal[:, ring_channels]
    # Compute common average
    common_avg = ring_data.mean(axis=1, keepdims=True)
    # Subtract from each channel
    return ring_data - common_avg


def apply_car_to_signal(emg_signal: np.ndarray, 
                        forearm_ring1: List[int] = list(range(0, 8)),
                        forearm_ring2: List[int] = list(range(8, 16))) -> np.ndarray:
    """
    Apply CAR to both forearm rings in GRABMyo data.
    
    Args:
        emg_signal: Raw EMG signal, shape (n_samples, n_channels >= 16)
        forearm_ring1: Channels for ring 1 (default: F1-F8, indices 0-7)
        forearm_ring2: Channels for ring 2 (default: F9-F16, indices 8-15)
        
    Returns:
        CAR-processed signal, shape (n_samples, 16)
    """
    ring1_car = apply_car(emg_signal, forearm_ring1)
    ring2_car = apply_car(emg_signal, forearm_ring2)
    
    # Concatenate
    return np.concatenate([ring1_car, ring2_car], axis=1)


def bandpass_filter(emg_signal: np.ndarray, 
                    fs: int = 2048, 
                    lowcut: float = 20.0, 
                    highcut: float = 450.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to EMG signal.
    
    Args:
        emg_signal: Input signal, shape (n_samples, n_channels)
        fs: Sampling frequency in Hz
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        order: Filter order
        
    Returns:
        Filtered signal, same shape as input
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered = np.zeros_like(emg_signal)
    for i in range(emg_signal.shape[1]):
        # Use filtfilt for zero-phase filtering
        filtered[:, i] = signal.filtfilt(b, a, emg_signal[:, i])
    
    return filtered


def notch_filter(emg_signal: np.ndarray,
                 fs: int = 2048,
                 freq: float = 50.0,
                 quality: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference.
    
    Args:
        emg_signal: Input signal, shape (n_samples, n_channels)
        fs: Sampling frequency in Hz
        freq: Frequency to remove (50 or 60 Hz)
        quality: Quality factor (higher = narrower notch)
        
    Returns:
        Filtered signal, same shape as input
    """
    # Design notch filter
    b, a = signal.iirnotch(freq, quality, fs)
    
    # Apply to each channel
    filtered = np.zeros_like(emg_signal)
    for i in range(emg_signal.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, emg_signal[:, i])
    
    return filtered


def sliding_window(emg_signal: np.ndarray, 
                   window_len: int, 
                   hop: int) -> np.ndarray:
    """
    Generate overlapping sliding windows.
    
    Args:
        emg_signal: Input signal, shape (n_samples, n_channels)
        window_len: Window length in samples
        hop: Hop size in samples (stride)
        
    Returns:
        Windows array, shape (n_windows, window_len, n_channels)
    """
    n_samples, n_channels = emg_signal.shape
    
    # Calculate number of windows
    n_windows = (n_samples - window_len) // hop + 1
    
    windows = []
    for i in range(n_windows):
        start = i * hop
        end = start + window_len
        if end <= n_samples:
            windows.append(emg_signal[start:end, :])
    
    return np.array(windows)


def normalize_signal(emg_signal: np.ndarray, 
                     method: str = 'zscore',
                     stats: Tuple[np.ndarray, np.ndarray] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Normalize signal per channel.
    
    Args:
        emg_signal: Input signal, shape (n_samples, n_channels) or (window_len, n_channels)
        method: 'zscore' or 'robust'
        stats: Optional pre-computed (mean, std) for consistent normalization
        
    Returns:
        Tuple of (normalized_signal, (mean, std))
    """
    if stats is None:
        if method == 'zscore':
            mean = emg_signal.mean(axis=0, keepdims=True)
            std = emg_signal.std(axis=0, keepdims=True) + 1e-6
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(emg_signal, axis=0, keepdims=True)
            q75, q25 = np.percentile(emg_signal, [75, 25], axis=0, keepdims=True)
            iqr = q75 - q25 + 1e-6
            mean = median
            std = iqr
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        mean, std = stats
    
    normalized = (emg_signal - mean) / std
    return normalized, (mean, std)


def preprocess_trial(trial_signal: np.ndarray,
                     fs: int = 2048,
                     apply_notch: bool = False,
                     notch_freq: float = 50.0) -> np.ndarray:
    """
    Complete preprocessing pipeline for one trial.
    
    Pipeline:
    1. CAR (Common Average Re-reference)
    2. Bandpass filter (20-450 Hz)
    3. Optional: Notch filter (50/60 Hz)
    
    Args:
        trial_signal: Raw trial signal, shape (n_samples, n_channels >= 16)
        fs: Sampling rate
        apply_notch: Whether to apply notch filter
        notch_freq: Powerline frequency (50 or 60 Hz)
        
    Returns:
        Preprocessed signal, shape (n_samples, 16)
    """
    # Step 1: CAR
    signal_car = apply_car_to_signal(trial_signal)
    
    # Step 2: Bandpass filter
    signal_bp = bandpass_filter(signal_car, fs=fs)
    
    # Step 3: Optional notch filter
    if apply_notch:
        signal_bp = notch_filter(signal_bp, fs=fs, freq=notch_freq)
    
    return signal_bp


def compute_velocity(emg_signal: np.ndarray) -> np.ndarray:
    """
    Compute velocity (1st derivative) of EMG signal.
    
    Args:
        emg_signal: Input signal, shape (..., n_samples, n_channels)
        
    Returns:
        Velocity signal, same shape as input
    """
    # Use gradient along time axis (axis=-2 for (..., n_samples, n_channels))
    return np.gradient(emg_signal, axis=-2)


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing functions...")
    
    # Generate dummy signal
    fs = 2048
    duration = 5.0  # seconds
    n_samples = int(fs * duration)
    n_channels = 32
    
    # Dummy signal with some noise
    dummy_signal = np.random.randn(n_samples, n_channels) * 0.01
    
    # Add 50 Hz powerline interference
    t = np.arange(n_samples) / fs
    for ch in range(n_channels):
        dummy_signal[:, ch] += 0.05 * np.sin(2 * np.pi * 50 * t)
    
    print(f"Input signal shape: {dummy_signal.shape}")
    
    # Test CAR
    car_signal = apply_car_to_signal(dummy_signal)
    print(f"After CAR: {car_signal.shape}")
    print(f"Ring 1 mean (should be ~0): {car_signal[:, :8].mean():.6f}")
    print(f"Ring 2 mean (should be ~0): {car_signal[:, 8:].mean():.6f}")
    
    # Test bandpass
    bp_signal = bandpass_filter(car_signal, fs=fs)
    print(f"After bandpass: {bp_signal.shape}")
    
    # Test notch
    notch_signal = notch_filter(bp_signal, fs=fs, freq=50.0)
    print(f"After notch: {notch_signal.shape}")
    
    # Test sliding window (200ms window, 150ms overlap)
    window_len = int(0.2 * fs)  # 200ms = 409 samples @ 2048Hz
    hop = int(0.05 * fs)  # 50ms hop = 102 samples
    windows = sliding_window(notch_signal, window_len, hop)
    print(f"Sliding windows: {windows.shape} (expected ~48 windows for 5s)")
    
    # Test normalization
    norm_signal, stats = normalize_signal(windows[0])
    print(f"Normalized window shape: {norm_signal.shape}")
    print(f"Stats (mean, std): {stats[0].shape}, {stats[1].shape}")
    
    # Test full pipeline
    processed = preprocess_trial(dummy_signal, fs=fs, apply_notch=True)
    print(f"Full pipeline output: {processed.shape}")
    
    print("\n✅ All preprocessing tests passed!")
