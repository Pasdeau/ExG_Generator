#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Surface-ENG (electroneurogram) simulation prototype @ 8 kSPS.

Key idea:
- Underlying "neural source" = multi-unit spike trains convolved with extracellular-like biphasic waveforms.
- Two surface electrodes observe filtered/attenuated versions of the same source (slight delay + scaling),
  then bipolar differential recording is formed (E1 - E2).
- Add realistic contaminations: EMG bursts (band-limited), baseline drift, 50 Hz mains + harmonics,
  1/f noise, occasional motion transients, optional ECG artifact.
- Output raw and processed signals (bandpass + notch).

This is for algorithm development / simulation only (not clinical).
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np


# -----------------------------
# IO helpers
# -----------------------------
def save_csv(path: Path, t: np.ndarray, y: np.ndarray, colname: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", colname])
        for ti, yi in zip(t.tolist(), y.tolist()):
            w.writerow([f"{ti:.9f}", f"{yi:.6f}"])


def save_meta(path: Path, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# Signal building blocks
# -----------------------------
def biphasic_unit_waveform(fs: int, width_ms: float, amp_uV: float) -> np.ndarray:
    """Difference-of-Gaussians biphasic spike-like waveform."""
    width_s = max(0.05e-3, width_ms / 1000.0)
    t = np.arange(-4 * width_s, 4 * width_s, 1.0 / fs)
    sigma1 = width_s / 3.0
    sigma2 = width_s / 2.0
    g1 = np.exp(-0.5 * (t / sigma1) ** 2)
    g2 = np.exp(-0.5 * (t / sigma2) ** 2)
    w = g1 - 0.85 * g2
    w = w / (np.max(np.abs(w)) + 1e-12)
    return (amp_uV * w).astype(np.float64)


def add_wave_centered(y: np.ndarray, idx: int, w: np.ndarray, scale: float = 1.0) -> None:
    """Add waveform w centered at idx into y."""
    n = len(y)
    m = len(w)
    half = m // 2
    i0 = idx - half
    i1 = i0 + m
    if i1 <= 0 or i0 >= n:
        return
    src0 = max(0, -i0)
    src1 = m - max(0, i1 - n)
    dst0 = max(0, i0)
    dst1 = min(n, i1)
    y[dst0:dst1] += scale * w[src0:src1]


def bandpass_fft(y: np.ndarray, fs: int, f_lo: float, f_hi: float) -> np.ndarray:
    """FFT bandpass (simple, dependency-free)."""
    n = len(y)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (f >= max(0.0, f_lo)) & (f <= min(fs / 2, f_hi))
    return np.fft.irfft(Y * mask, n=n)


def notch_iir_coeff(fs: int, f0: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple biquad notch filter coefficients.
    b(z) = 1 - 2cos(w0)z^-1 + z^-2
    a(z) = 1 - 2r cos(w0)z^-1 + r^2 z^-2
    where r controls bandwidth; q maps to r approximately.
    """
    w0 = 2.0 * math.pi * (f0 / fs)
    c = math.cos(w0)

    # map Q to pole radius r (heuristic). higher Q => narrower notch => r closer to 1.
    q = max(0.5, q)
    bw = (f0 / q)  # approx bandwidth in Hz
    r = math.exp(-math.pi * bw / fs)
    b = np.array([1.0, -2.0 * c, 1.0], dtype=np.float64)
    a = np.array([1.0, -2.0 * r * c, r * r], dtype=np.float64)
    return b, a


def lfilter_iir(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Direct-form I IIR filter (a[0] must be 1)."""
    y = np.zeros_like(x, dtype=np.float64)
    for n in range(len(x)):
        y[n] = b[0] * x[n]
        if n - 1 >= 0:
            y[n] += b[1] * x[n - 1] - a[1] * y[n - 1]
        if n - 2 >= 0:
            y[n] += b[2] * x[n - 2] - a[2] * y[n - 2]
    return y


def filtfilt_iir(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero-phase by forward-backward filtering."""
    y = lfilter_iir(b, a, x)
    y = lfilter_iir(b, a, y[::-1])[::-1]
    return y


def pink_noise_1_over_f(n: int, fs: int, rng: np.random.Generator, std_uV: float) -> np.ndarray:
    """Approximate 1/f noise using frequency shaping."""
    if std_uV <= 0:
        return np.zeros(n, dtype=np.float64)
    w = rng.normal(0.0, 1.0, size=n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    shape = np.ones_like(f)
    shape[1:] = 1.0 / np.sqrt(f[1:])
    y = np.fft.irfft(W * shape, n=n)
    y = y / (np.std(y) + 1e-12)
    return (std_uV * y).astype(np.float64)


def band_limited_noise(n: int, fs: int, rng: np.random.Generator, f_lo: float, f_hi: float, std_uV: float) -> np.ndarray:
    """Generate band-limited noise by FFT masking."""
    if std_uV <= 0:
        return np.zeros(n, dtype=np.float64)
    x = rng.normal(0.0, 1.0, size=n)
    x = bandpass_fft(x, fs, f_lo, f_hi)
    x = x / (np.std(x) + 1e-12)
    return (std_uV * x).astype(np.float64)


def smooth_envelope_gate(t: np.ndarray, rng: np.random.Generator, n_bursts: int, dur_s: Tuple[float, float]) -> np.ndarray:
    """Create smooth burst gate (0..1) by summing raised-cosine windows."""
    g = np.zeros_like(t, dtype=np.float64)
    T = t[-1] if len(t) else 0.0
    for _ in range(max(0, n_bursts)):
        d = rng.uniform(dur_s[0], dur_s[1])
        c = rng.uniform(0.1, max(0.1, T - 0.1))
        t0 = c
        t1 = min(T, c + d)
        idx = (t >= t0) & (t <= t1)
        if not np.any(idx):
            continue
        tt = (t[idx] - t0) / max(1e-9, (t1 - t0))
        # raised cosine
        g[idx] += 0.5 - 0.5 * np.cos(2 * np.pi * tt)
    return np.clip(g, 0.0, 1.0)


import scipy.signal
from scipy.stats import gamma

# ... (Previous imports)

# -----------------------------
# Advanced Artifact Generation (Markov Chain + Colored Noise)
# -----------------------------
class ArtifactGenerator:
    """
    Generates structured artifacts based on a Markov Chain model.
    States: 0=Clean, 1=DeviceDisp, 2=ForearmMotion, 3=HandMotion, 4=PoorContact.
    """
    def __init__(self, fs: int, duration_s: float, seed: int):
        self.fs = fs
        self.duration_s = duration_s
        self.rng = np.random.default_rng(seed)
        self.n_samples = int(round(fs * duration_s))
        
        # Default Transition Probabilities (Clean -> Artifact) per second approx
        # These will be scaled by sample rate in usage or checked per segment
        self.prob_clean_to_artifact = [0.0, 0.05, 0.05, 0.05, 0.05] # [Dummy, to_1, to_2, to_3, to_4]
        
        # Artifact Durations (mean seconds) - Exponential distribution
        self.avg_durations = [2.0, 0, 0, 0, 1.5] # [Clean, A1, A2, A3, A4]
        
        # Amplitude Parameters (Gamma Dist: shape k, scale theta) - tuned for uV scale
        # artifacts are usually much larger than ENG (e.g. 50-200 uV)
        self.amp_params = [
            (0, 0),         # Clean (not used)
            (2.0, 40.0),    # 1: Device Displacement (Large, Low Freq)
            (1.5, 25.0),    # 2: Forearm (Muscle-like)
            (1.5, 25.0),    # 3: Hand (Muscle-like)
            (1.0, 60.0)     # 4: Poor Contact (Huge, fast changes)
        ]

        # Filter Designs for Coloration (FIR)
        # We define rough bandpass/lowpass characteristics for each
        self.filters = {}
        self._design_filters()

    def _design_filters(self):
        # Nyquist
        nyq = 0.5 * self.fs
        
        # 1. Device Displacement: Very low freq (< 5 Hz)
        # Using simple Butterworth for smoothness
        b1, a1 = scipy.signal.butter(2, 5.0 / nyq, btype='low')
        self.filters[1] = (b1, a1)

        # 2. Forearm Motion: EMG-like bandwidth (20-300 Hz)
        b2, a2 = scipy.signal.butter(2, [10.0 / nyq, 300.0 / nyq], btype='band')
        self.filters[2] = (b2, a2)

        # 3. Hand Motion: Similar to Forearm but maybe slightly different spectral shape
        b3, a3 = scipy.signal.butter(2, [15.0 / nyq, 250.0 / nyq], btype='band')
        self.filters[3] = (b3, a3)

        # 4. Poor Contact: 50Hz noise bursts + erratic baseline (wideband)
        # We'll just use a wider bandpass
        b4, a4 = scipy.signal.butter(2, [1.0 / nyq, 500.0 / nyq], btype='band')
        self.filters[4] = (b4, a4)

    def generate(self) -> Tuple[np.ndarray, list]:
        """Returns (artifact_signal, label_list)"""
        signal_out = np.zeros(self.n_samples, dtype=np.float64)
        labels = [] # List of (start_s, end_s, label_code, label_name)
        
        current_state = 0 # Start clean
        idx = 0
        
        state_names = {0: "Clean", 1: "DeviceDisp", 2: "ForearmMotion", 3: "HandMotion", 4: "PoorContact"}

        while idx < self.n_samples:
            # Determine duration of this state
            mean_dur = self.avg_durations[current_state]
            # Exponential duration
            dur_s = self.rng.exponential(mean_dur)
            dur_s = max(0.1, dur_s) # Min duration
            dur_samples = int(dur_s * self.fs)
            
            # Clip to end
            end_idx = min(idx + dur_samples, self.n_samples)
            actual_samples = end_idx - idx
            
            if current_state > 0:
                # Generate Artifact
                # 1. White noise
                noise = self.rng.standard_normal(actual_samples)
                
                # 2. Color it
                b, a = self.filters[current_state]
                colored = scipy.signal.lfilter(b, a, noise)
                
                # 3. Scale it (Gamma dist amplitude)
                k, theta = self.amp_params[current_state]
                amp_factor = self.rng.gamma(k, theta)
                
                # Normalize filter output roughly so amp_factor means something (uV)
                if np.std(colored) > 1e-9:
                    colored /= np.std(colored)
                
                segment = colored * amp_factor
                
                # Add to output
                signal_out[idx:end_idx] += segment
                
                # Log Label
                labels.append({
                    "start_s": idx / self.fs,
                    "end_s": end_idx / self.fs,
                    "type_id": int(current_state),
                    "type_name": state_names[current_state]
                })
                
                # Transition: Always back to Clean after artifact
                next_state = 0
            
            else:
                # Clean State (Zeros in output, just wait)
                # Transition: Randomly to one of the artifacts based on weights
                # probs = [Prob(0->1), Prob(0->2)...]
                # We normalize them to sum to 1 for choice, or just pick.
                # Let's say we pick *which* artifact to go to next, or stay clean?
                # Actually Markov model usually:
                # If Clean, chance to go to 1, 2, 3, 4. 
                # Let's just pick uniformly among artifacts for now, or use weights.
                
                r = self.rng.random()
                # Simplified transition logic:
                # After a clean segment, we ALWAYS go to an artifact? 
                # Or we decide "which artifact is next".
                # To make it sparse, "Clean" duration should be long.
                
                next_state = self.rng.choice([1, 2, 3, 4])
            
            idx = end_idx
            current_state = next_state
            
        return signal_out, labels


# -----------------------------
# Config
# -----------------------------
@dataclass
class NeuralModel:
    n_units: int = 35
    rate_hz_mean: float = 6.0
    rate_hz_sd: float = 2.0
    unit_amp_uV_mean: float = 6.0
    unit_amp_uV_sd: float = 2.5
    # TUNED: 1.0 ms is more realistic for Sensory Nerve Action Potentials (SNAPs) (vs 2.0ms for EMG)
    unit_width_ms_mean: float = 1.0
    unit_width_ms_sd: float = 0.2
    burst_prob_per_s: float = 0.12
    burst_dur_s: Tuple[float, float] = (0.08, 0.25)
    burst_rate_mult: float = 3.5


@dataclass
class ElectrodeModel:
    # approximate differences between two surface electrodes for bipolar recording
    e2_delay_ms: float = 0.35
    e2_scale: float = 0.92
    # additional lowpass-like smoothing due to tissue / geometry (very mild)
    spatial_smooth_ms: float = 0.6


@dataclass
class ContaminationModel:
    # drift / mains / noise are in uV
    drift_amp_uV: float = 25.0
    drift_hz: float = 0.15

    mains_hz: float = 50.0
    mains_amp_uV: float = 6.0
    mains_harmonics: int = 2  # add 100,150,...

    white_std_uV: float = 4.5
    pink_std_uV: float = 3.0

    # EMG-like bursts: band-limited noise with envelope
    emg_enabled: bool = True
    emg_band: Tuple[float, float] = (30.0, 450.0)
    emg_bursts: int = 4
    emg_burst_dur_s: Tuple[float, float] = (0.15, 0.6)
    emg_std_uV: float = 10.0

    # motion transient: low-frequency bump events
    motion_enabled: bool = True
    motion_events: int = 2
    motion_amp_uV: float = 60.0
    motion_dur_s: Tuple[float, float] = (0.08, 0.22)

    # optional ECG artifact (very simplified)
    ecg_enabled: bool = False
    heart_rate_bpm: float = 70.0
    ecg_amp_uV: float = 30.0


@dataclass
class ProcessingModel:
    bandpass_lo_hz: float = 20.0
    # TUNED: Increased to 3000 Hz to capture faster SNAP spikes
    bandpass_hi_hz: float = 3000.0
    notch_hz: float = 50.0
    notch_q: float = 30.0
    notch_harmonics: int = 2  # notch 50,100,150...
    output_processed: bool = True


@dataclass
class SimConfig:
    fs: int = 8000
    duration_s: float = 10.0
    seed: int = 123
    neural: NeuralModel = field(default_factory=NeuralModel)
    electrode: ElectrodeModel = field(default_factory=ElectrodeModel)
    contam: ContaminationModel = field(default_factory=ContaminationModel)
    proc: ProcessingModel = field(default_factory=ProcessingModel)


# -----------------------------
# Core simulation
# -----------------------------
def simulate(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict, list, list, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    n = int(round(cfg.fs * cfg.duration_s))
    t = np.arange(n, dtype=np.float64) / cfg.fs

    # ---- Neural source: multi-unit spikes convolved with biphasic waveforms
    src = np.zeros(n, dtype=np.float64)
    events = []

    # burst gate for "normal occasional bursts"
    expected_bursts = cfg.neural.burst_prob_per_s * cfg.duration_s
    n_bursts = rng.poisson(expected_bursts)
    burst_gate = smooth_envelope_gate(t, rng, n_bursts, cfg.neural.burst_dur_s)
    # rate multiplier in bursts
    rate_mult = 1.0 + burst_gate * (cfg.neural.burst_rate_mult - 1.0)

    for u in range(cfg.neural.n_units):
        rate = max(0.1, rng.normal(cfg.neural.rate_hz_mean, cfg.neural.rate_hz_sd))
        rate_t = rate * rate_mult
        # Poisson spikes via Bernoulli per sample
        p = np.clip(rate_t / cfg.fs, 0.0, 0.2)
        spikes = rng.random(n) < p
        idxs = np.flatnonzero(spikes)

        amp = max(0.5, rng.normal(cfg.neural.unit_amp_uV_mean, cfg.neural.unit_amp_uV_sd))
        width = max(0.8, rng.normal(cfg.neural.unit_width_ms_mean, cfg.neural.unit_width_ms_sd))
        w = biphasic_unit_waveform(cfg.fs, width, amp)

        for idx in idxs:
            add_wave_centered(src, int(idx), w, scale=1.0)
            events.append(("spike", float(idx) / cfg.fs))

    # mild tissue lowpass-ish smoothing of neural source (spatial effect proxy)
    if cfg.electrode.spatial_smooth_ms > 0:
        # simple gaussian smoothing kernel
        sigma_samp = (cfg.electrode.spatial_smooth_ms / 1000.0) * cfg.fs / 2.5
        sigma_samp = max(0.2, sigma_samp)
        L = int(6 * sigma_samp) + 1
        x = np.arange(L) - L // 2
        k = np.exp(-0.5 * (x / sigma_samp) ** 2)
        k = k / (np.sum(k) + 1e-12)
        src = np.convolve(src, k, mode="same")

    # ---- Two electrodes observe slightly different versions -> bipolar differential
    e1 = src.copy()

    delay_samp = int(round((cfg.electrode.e2_delay_ms / 1000.0) * cfg.fs))
    e2 = np.zeros_like(e1)
    if delay_samp > 0:
        e2[delay_samp:] = e1[:-delay_samp]
    else:
        e2[:] = e1
    e2 *= cfg.electrode.e2_scale

    # Bipolar recording (what you likely store as ENG)
    eng_neural = e1 - e2

    # ---- Add contaminations (some common-mode, some differential)
    drift = cfg.contam.drift_amp_uV * np.sin(2 * np.pi * cfg.contam.drift_hz * t + 0.3)

    mains = cfg.contam.mains_amp_uV * np.sin(2 * np.pi * cfg.contam.mains_hz * t + 0.1)
    for h in range(2, 2 + max(0, cfg.contam.mains_harmonics)):
        mains += (cfg.contam.mains_amp_uV / (h * 1.8)) * np.sin(2 * np.pi * (h * cfg.contam.mains_hz) * t + 0.1 * h)

    white = rng.normal(0.0, cfg.contam.white_std_uV, size=n)
    pink = pink_noise_1_over_f(n, cfg.fs, rng, cfg.contam.pink_std_uV)

    # ---- Advanced Artifact Generation (Replaces old EMG/Motion logic if enabled)
    # We will simply ADD this to the contamination.
    # Logic: if cfg.contam.use_advanced_artifacts is True (we'll assume True for this task updates)
    art_gen = ArtifactGenerator(cfg.fs, cfg.duration_s, cfg.seed + 1)
    artifact_sig, artifact_labels = art_gen.generate()

    # NOTE: Old simple EMG/Motion is KEPT as "background" baseline noise if desired, 
    # but the user asked to "apply the logic".
    # We will reduce the old random motion/emg to avoid double-counting if we want pure control,
    # OR we treat the new Generator as "Major Artifacts" and the old one as "Background".
    # Let's treat the new one as the primary source of "Events".
    
    # common-mode components reduced by bipolar; model imperfect rejection:
    cm_reject = 0.25  # 0 means perfect cancel; 1 means no cancel
    
    # Total Raw Signal
    # We now separate components for visualization if needed
    clean_baseline = eng_neural + white + pink # "Physiological + Electronic noise" (Clean-ish)
    raw = clean_baseline + cm_reject * (drift + mains) + artifact_sig

    # ---- Processing (typical surface pipeline): bandpass + notch
    proc = raw.copy()
    if cfg.proc.output_processed:
        proc = bandpass_fft(proc, cfg.fs, cfg.proc.bandpass_lo_hz, cfg.proc.bandpass_hi_hz)
        # notch 50 Hz and harmonics
        for h in range(1, 1 + max(0, cfg.proc.notch_harmonics)):
            f0 = cfg.proc.notch_hz * h
            if f0 < cfg.fs / 2:
                b, a = notch_iir_coeff(cfg.fs, f0, cfg.proc.notch_q)
                proc = filtfilt_iir(b, a, proc)

    meta = {
        "fs": cfg.fs,
        "duration_s": cfg.duration_s,
        "seed": cfg.seed,
        "models": {
            "neural": asdict(cfg.neural),
            "electrode": asdict(cfg.electrode),
            "contam": asdict(cfg.contam),
            "processing": asdict(cfg.proc),
        },
        "notes": "Synthetic surface-ENG for algorithm prototyping. Units are microvolts (uV).",
        "artifact_labels": artifact_labels # Include in meta too
    }

    # Return expanded set for debugging/comparison
    return t, raw, proc, meta, events, artifact_labels, clean_baseline, artifact_sig


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser("eng_surface_sim", description="Surface-ENG simulation prototype @ 8 kSPS.")
    ap.add_argument("--fs", type=int, default=8000)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-prefix", type=str, default="out/eng_normal")

    # quick knobs
    ap.add_argument("--no-emg", action="store_true")
    ap.add_argument("--no-motion", action="store_true")
    ap.add_argument("--ecg", action="store_true")

    ap.add_argument("--bandpass-lo", type=float, default=20.0)
    ap.add_argument("--bandpass-hi", type=float, default=3000.0)
    ap.add_argument("--notch-q", type=float, default=30.0)
    ap.add_argument("--notch-harmonics", type=int, default=2)

    ap.add_argument("--save-npy", action="store_true", help="Also save .npy arrays for fast loading.")
    ap.add_argument("--save-events", action="store_true", help="Also save spike event times (ground truth).")
    ap.add_argument("--plot", action="store_true", help="Generate summary plots.")

    args = ap.parse_args()

    cfg = SimConfig(fs=args.fs, duration_s=args.duration, seed=args.seed)
    cfg.contam.emg_enabled = not args.no_emg
    cfg.contam.motion_enabled = not args.no_motion
    cfg.contam.ecg_enabled = bool(args.ecg)

    cfg.proc.bandpass_lo_hz = args.bandpass_lo
    cfg.proc.bandpass_hi_hz = args.bandpass_hi
    cfg.proc.notch_q = args.notch_q
    cfg.proc.notch_harmonics = max(0, args.notch_harmonics)

    t, raw, proc, meta, events, artifact_labels, clean, noise_only = simulate(cfg)

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    save_csv(prefix.with_suffix(".raw.csv"), t, raw, "eng_raw_uV")
    save_csv(prefix.with_suffix(".proc.csv"), t, proc, "eng_proc_uV")
    save_meta(prefix.with_suffix(".meta.json"), meta)

    if args.save_npy:
        np.save(prefix.with_suffix(".t.npy"), t)
        np.save(prefix.with_suffix(".raw.npy"), raw)
        np.save(prefix.with_suffix(".proc.npy"), proc)


    if args.save_events:
        # events.csv: (type, time_s)
        with prefix.with_suffix(".events.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["event_type", "time_s"])
            for et, ts in events:
                w.writerow([et, f"{ts:.9f}"])

    # Save Artifact Labels (New)
    with prefix.with_suffix(".labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "type_id", "type_name"])
        for row in artifact_labels:
            w.writerow([f"{row['start_s']:.9f}", f"{row['end_s']:.9f}", row['type_id'], row['type_name']])

    print(f"[OK] wrote:\n  {prefix.with_suffix('.raw.csv')}\n  {prefix.with_suffix('.proc.csv')}\n  {prefix.with_suffix('.meta.json')}\n  {prefix.with_suffix('.labels.csv')}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plot_results_comparison(t, clean, noise_only, raw, proc, artifact_labels, prefix.with_suffix(".png"))
            print(f"  {prefix.with_suffix('.png')}")
        except ImportError:
            print("[WARN] matplotlib not found, skipping plot.")


def plot_results_comparison(t: np.ndarray, clean: np.ndarray, noise: np.ndarray, raw: np.ndarray, proc: np.ndarray, labels: list, out_path: Path) -> None:
    """Generate a comparison plot: Clean vs Noise vs Combined vs Processed."""
    import matplotlib.pyplot as plt
    
    # Setup 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    colors = {1: 'orange', 2: 'green', 3: 'purple', 4: 'red'}

    # 1. Clean Signal (Baseline)
    axes[0].plot(t, clean, color='k', linewidth=0.6, label="Clean (Neural + Thermal)")
    axes[0].set_ylabel("uV")
    axes[0].set_title("1. Clean Baseline (Neural + Background Noise)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # 2. Pure Artifacts (The "Noise" added)
    axes[1].plot(t, noise, color='r', linewidth=0.6, label="Injected Artifacts")
    for l in labels:
        if l['type_id'] > 0:
            axes[1].axvspan(l['start_s'], l['end_s'], color=colors.get(l['type_id'], 'gray'), alpha=0.1, label=l['type_name'])
    
    # Deduplicate legend
    handles, lbls = axes[1].get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    axes[1].legend(by_label.values(), by_label.keys(), loc="upper right", fontsize='small')
    
    axes[1].set_ylabel("uV")
    axes[1].set_title("2. Injected Artifacts Only")
    axes[1].grid(True, alpha=0.3)

    # 3. Combined Raw
    axes[2].plot(t, raw, color='k', linewidth=0.6, label="Total Raw (1 + 2)")
    # Shade regions for reference
    for l in labels:
        if l['type_id'] > 0:
            axes[2].axvspan(l['start_s'], l['end_s'], color=colors.get(l['type_id'], 'gray'), alpha=0.1)
    
    axes[2].set_ylabel("uV")
    axes[2].set_title("3. Resulting Raw Signal (Sensor Input)")
    axes[2].grid(True, alpha=0.3)

    # 4. Processed
    axes[3].plot(t, proc, color='#007acc', linewidth=0.8, label="Bandpass + Notch")
    axes[3].set_ylabel("uV")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("4. Processed Output")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
