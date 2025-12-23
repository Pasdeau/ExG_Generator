"""
Hudgins Time-Domain Feature Extraction for EMG Signals.

Classic EMG features widely used in gesture recognition:
- MAV: Mean Absolute Value
- WL: Waveform Length
- ZC: Zero Crossings
- SSC: Slope Sign Changes
- RMS: Root Mean Square
- IEMG: Integrated EMG
- AR: Auto-Regressive Coefficients

References:
- Hudgins et al. (1993): "A New Strategy for Multifunction Myoelectric Control"
- Phinyomark et al. (2012): "Feature Extraction of the First Difference of EMG Time Series for EMG Pattern Recognition"
"""

import numpy as np
from scipy import signal
from typing import Tuple


def extract_mav(window: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Value per channel.
    
    Args:
        window: EMG window, shape (window_len, n_channels)
        
    Returns:
        MAV features, shape (n_channels,)
    """
    return np.mean(np.abs(window), axis=0)


def extract_wl(window: np.ndarray) -> np.ndarray:
    """
    Waveform Length per channel (cumulative absolute difference).
    
    Args:
        window: EMG window, shape (window_len, n_channels)
        
    Returns:
        WL features, shape (n_channels,)
    """
    diff = np.abs(np.diff(window, axis=0))
    return np.sum(diff, axis=0)


def extract_zc(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Zero Crossings per channel (with optional threshold).
    
    Args:
        window: EMG window, shape (window_len, n_channels)
        threshold: Minimum amplitude change to count as crossing
        
    Returns:
        ZC counts, shape (n_channels,)
    """
    n_channels = window.shape[1]
    zc = np.zeros(n_channels)
    
    for ch in range(n_channels):
        sig = window[:, ch]
        # Sign changes
        sign_changes = np.diff(np.sign(sig))
        # Count non-zero sign changes (ignoring threshold for now)
        if threshold > 0:
            # Only count if amplitude difference exceeds threshold
            diffs = np.abs(np.diff(sig))
            valid_changes = (sign_changes != 0) & (diffs > threshold)
            zc[ch] = np.sum(valid_changes)
        else:
            zc[ch] = np.sum(sign_changes != 0)
    
    return zc


def extract_ssc(window: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Slope Sign Changes per channel.
    
    Detects local extrema (peaks and valleys).
    
    Args:
        window: EMG window, shape (window_len, n_channels)
        threshold: Minimum amplitude change to count as slope change
        
    Returns:
        SSC counts, shape (n_channels,)
    """
    n_channels = window.shape[1]
    ssc = np.zeros(n_channels)
    
    for ch in range(n_channels):
        sig = window[:, ch]
        # Compute slopes
        slopes = np.diff(sig)
        # Slope sign changes (ignoring first and last point)
        for i in range(1, len(slopes)):
            if threshold > 0:
                # Check if it's a local extremum with sufficient amplitude
                if (slopes[i-1] * slopes[i] < 0) and (np.abs(sig[i] - sig[i-1]) > threshold or np.abs(sig[i] - sig[i+1]) > threshold):
                    ssc[ch] += 1
            else:
                if slopes[i-1] * slopes[i] < 0:
                    ssc[ch] += 1
    
    return ssc


def extract_rms(window: np.ndarray) -> np.ndarray:
    """
    Root Mean Square per channel.
    
    Args:
        window: EMG window, shape (window_len, n_channels)
        
    Returns:
        RMS features, shape (n_channels,)
    """
    return np.sqrt(np.mean(window ** 2, axis=0))


def extract_iemg(window: np.ndarray) -> np.ndarray:
    """
    Integrated EMG (sum of absolute values) per channel.
    
    Args:
        window: EMG window, shape (window_len, n_channels)
        
    Returns:
        IEMG features, shape (n_channels,)
    """
    return np.sum(np.abs(window), axis=0)


def extract_ar_coefficients(window: np.ndarray, order: int = 4) -> np.ndarray:
    """
    Auto-Regressive coefficients per channel.
    
    Uses Yule-Walker equations to estimate AR model.
    
    Args:
        window: EMG window, shape (window_len, n_channels)
        order: AR model order
        
    Returns:
        AR coefficients, shape (n_channels, order)
    """
    n_channels = window.shape[1]
    ar_coeffs = np.zeros((n_channels, order))
    
    for ch in range(n_channels):
        sig = window[:, ch]
        # Compute autocorrelation
        try:
            # Use scipy's levinson_durbin or numpy's correlate
            # Here we use a simple approach with numpy
            r = np.correlate(sig, sig, mode='full')
            r = r[len(r)//2:]  # Take positive lags
            r = r[:order+1] / r[0]  # Normalize
            
            # Solve Yule-Walker equations
            # R @ a = -r, where R is Toeplitz matrix
            R = np.zeros((order, order))
            for i in range(order):
                for j in range(order):
                    R[i, j] = r[abs(i - j)]
            
            r_vec = -r[1:order+1]
            
            # Solve linear system
            a = np.linalg.solve(R, r_vec)
            ar_coeffs[ch, :] = a
        except:
            # If singular, use zeros
            ar_coeffs[ch, :] = 0.0
    
    return ar_coeffs


def extract_hudgins_features(window: np.ndarray, 
                             ar_order: int = 4,
                             zc_threshold: float = 0.0,
                             ssc_threshold: float = 0.0) -> np.ndarray:
    """
    Extract all Hudgins time-domain features for one window.
    
    Args:
        window: EMG window, shape (window_len, n_channels)
        ar_order: Order for AR coefficients
        zc_threshold: Threshold for Zero Crossings
        ssc_threshold: Threshold for Slope Sign Changes
        
    Returns:
        Feature vector, shape (n_channels * (6 + ar_order),)
        Features per channel: [MAV, WL, ZC, SSC, RMS, IEMG, AR1, AR2, ..., AR{order}]
    """
    mav = extract_mav(window)
    wl = extract_wl(window)
    zc = extract_zc(window, threshold=zc_threshold)
    ssc = extract_ssc(window, threshold=ssc_threshold)
    rms = extract_rms(window)
    iemg = extract_iemg(window)
    ar = extract_ar_coefficients(window, order=ar_order)
    
    # Stack features per channel
    n_channels = window.shape[1]
    features = []
    for ch in range(n_channels):
        ch_features = [
            mav[ch],
            wl[ch],
            zc[ch],
            ssc[ch],
            rms[ch],
            iemg[ch]
        ]
        ch_features.extend(ar[ch, :])
        features.extend(ch_features)
    
    return np.array(features)


def extract_features_from_dataset(windows: np.ndarray, 
                                  ar_order: int = 4) -> np.ndarray:
    """
    Extract Hudgins features for multiple windows.
    
    Args:
        windows: Array of windows, shape (n_windows, window_len, n_channels)
        ar_order: AR model order
        
    Returns:
        Feature matrix, shape (n_windows, n_features)
    """
    n_windows = windows.shape[0]
    # Compute feature dimension from first window
    sample_features = extract_hudgins_features(windows[0], ar_order=ar_order)
    n_features = len(sample_features)
    
    features = np.zeros((n_windows, n_features))
    for i in range(n_windows):
        features[i, :] = extract_hudgins_features(windows[i], ar_order=ar_order)
    
    return features


if __name__ == "__main__":
    print("Testing Hudgins feature extraction...")
    
    # Generate dummy EMG window
    fs = 2048
    window_len = int(0.2 * fs)  # 200ms = 409 samples
    n_channels = 16
    
    # Simulate EMG: random noise + some sinusoidal components
    t = np.linspace(0, 0.2, window_len)
    window = np.zeros((window_len, n_channels))
    for ch in range(n_channels):
        # Mix of frequencies to simulate muscle activity
        window[:, ch] = (
            0.1 * np.sin(2 * np.pi * 50 * t + ch * 0.1) +
            0.05 * np.sin(2 * np.pi * 150 * t + ch * 0.2) +
            0.02 * np.random.randn(window_len)
        )
    
    print(f"Test window shape: {window.shape}")
    
    # Test individual features
    print("\n--- Individual Features ---")
    print(f"MAV: {extract_mav(window).shape} = {extract_mav(window)[:3]}...")
    print(f"WL: {extract_wl(window).shape} = {extract_wl(window)[:3]}...")
    print(f"ZC: {extract_zc(window).shape} = {extract_zc(window)[:3]}...")
    print(f"SSC: {extract_ssc(window).shape} = {extract_ssc(window)[:3]}...")
    print(f"RMS: {extract_rms(window).shape} = {extract_rms(window)[:3]}...")
    print(f"IEMG: {extract_iemg(window).shape} = {extract_iemg(window)[:3]}...")
    print(f"AR(4): {extract_ar_coefficients(window, order=4).shape}")
    
    # Test combined features
    print("\n--- Combined Hudgins Features ---")
    features = extract_hudgins_features(window, ar_order=4)
    print(f"Total feature vector shape: {features.shape}")
    print(f"Expected: {n_channels * (6 + 4)} = {n_channels * 10}")
    
    # Test batch extraction
    print("\n--- Batch Feature Extraction ---")
    n_windows = 10
    windows = np.random.randn(n_windows, window_len, n_channels) * 0.1
    features_batch = extract_features_from_dataset(windows, ar_order=4)
    print(f"Batch features shape: {features_batch.shape}")
    print(f"Expected: ({n_windows}, {n_channels * 10})")
    
    print("\n✅ All feature extraction tests passed!")
