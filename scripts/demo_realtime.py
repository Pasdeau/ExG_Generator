
import sys
import os
import time
import glob
import numpy as np
import wfdb
import argparse
import torch
from typing import List

# Ensure we can import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gesture_recognition.realtime.inference import RealTimeGestureRecognizer

# GRABMyo Gesture mapping (1-17 original -> 0-15 mapped -> or select specific subset?)
# In our training, we used gestures 10-17 mapped to 0-7.
# Let's verify this mapping.
# Gesture 10: Lateral Prehension
# Gesture 11: Thumb Adduction
# ...
GESTURE_NAMES = {
    0: "Lateral Prehension",
    1: "Thumb Adduction",
    2: "Thumb Extension",
    3: "Thumb Flexion",
    4: "Index Extension",
    5: "Index Flexion",
    6: "Little Finger Extension",
    7: "Little Finger Flexion"
}

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_trial_data(trial_path: str) -> np.ndarray:
    """Load a WFDB trial file."""
    # trial_path should not have extension
    record = wfdb.rdrecord(trial_path)
    return record.p_signal.astype(np.float32)

def find_trials(session_dir: str, subject_id: int, gestures: List[int]) -> List[str]:
    """Find all trials for specific gestures in a session directory."""
    # Path pattern: sessionX_participantY/sessionX_participantY_gestureZ_trialK
    # We downloaded structure: data/sample_grabmyo/Session1/session1_participant1/...
    
    trials = []
    # Recursively find .hea files
    for root, dirs, files in os.walk(session_dir):
        for file in files:
            if file.endswith('.hea'):
                # Check subject
                if f"participant{subject_id}" not in file:
                    continue
                
                # Check gesture
                # filename format: session1_participant1_gesture10_trial1.hea
                parts = file.split('_')
                for p in parts:
                    if p.startswith('gesture'):
                        g_id = int(p.replace('gesture', ''))
                        if g_id in gestures:
                             trials.append(os.path.join(root, file.replace('.hea', '')))
    return sorted(trials)

def main():
    parser = argparse.ArgumentParser(description="Real-time EMG Gesture Recognition Demo")
    parser.add_argument('--session_dir', type=str, default='data/sample_grabmyo/Session1', help='Directory containing trial data')
    parser.add_argument('--model_path', type=str, default='checkpoints/tcn_fold2_best.pth', help='Path to model checkpoint')
    parser.add_argument('--subject', type=int, default=1, help='Subject ID to simulate')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed factor (1.0 = real-time)')
    args = parser.parse_args()

    # 1. Initialize Engine
    print(f"{Colors.HEADER}Initializing Real-time Engine...{Colors.ENDC}")
    try:
        engine = RealTimeGestureRecognizer(
            model_path=args.model_path,
            model_name='tcn',
            device='mps' if torch.backends.mps.is_available() else 'cpu'
        )
    except Exception as e:
        print(f"{Colors.FAIL}Failed to load engine: {e}{Colors.ENDC}")
        return

    # 2. Find Test Data (Gestures 10-17)
    target_gestures = list(range(10, 18))
    trial_paths = find_trials(args.session_dir, args.subject, target_gestures)
    
    if not trial_paths:
        print(f"{Colors.FAIL}No trials found for Subject {args.subject} in {args.session_dir}{Colors.ENDC}")
        return
        
    print(f"{Colors.GREEN}Found {len(trial_paths)} trials. Starting simulation...{Colors.ENDC}")
    time.sleep(1)

    # 3. Simulation Loop
    try:
        for trial_idx, trial_path in enumerate(trial_paths):
            # Extract gesture GT from filename
            filename = os.path.basename(trial_path)
            # format: session1_participant1_gesture10_trial1
            gt_gesture_original = int(filename.split('_')[2].replace('gesture', ''))
            gt_label = gt_gesture_original - 10 # Map to 0-7
            gt_name = GESTURE_NAMES.get(gt_label, "Unknown")
            
            print(f"\n{Colors.BLUE}=== Trial {trial_idx+1}/{len(trial_paths)}: {filename} ==={Colors.ENDC}")
            print(f"{Colors.BLUE}Ground Truth: {gt_name} ({gt_label}){Colors.ENDC}")
            
            # Load signal
            signal = load_trial_data(trial_path) # (N, 32)
            
            # Stream samples
            fs = 2048
            block_size = 102 # Feed in 50ms chunks (Hop size)
            
            # Skip very beginning to settle filters?
            # signal = signal[fs//2:] 
            
            # Stats
            correct_preds = 0
            total_preds = 0
            
            for i in range(0, len(signal), block_size):
                chunk = signal[i:i+block_size]
                if len(chunk) < block_size:
                    break
                    
                t0 = time.time()
                
                # Feed chunk to engine
                # engine.add_chunk returns a list of predictions (usually 0 or 1 per chunk if chunk=hop)
                preds = engine.add_chunk(chunk)
                
                # Simulation delay to match real-time
                process_time = time.time() - t0
                wait_time = (len(chunk) / fs) / args.speed - process_time
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Display Result
                if preds:
                    for pred in preds:
                        pred_cls, conf, lat = pred
                        pred_name = GESTURE_NAMES.get(pred_cls, "Unknown")
                        
                        total_preds += 1
                        if pred_cls == gt_label:
                            correct_preds += 1
                        
                        # Visualization
                        status_color = Colors.GREEN if pred_cls == gt_label else Colors.FAIL
                        bar_len = int(conf * 20)
                        bar = "█" * bar_len + "░" * (20 - bar_len)
                        
                        sys.stdout.write(f"\rTime: {i/fs:.1f}s | Pred: {status_color}{pred_name:<25}{Colors.ENDC} | {bar} {conf:.1%} | Latency: {lat:.1f}ms   ")
                        sys.stdout.flush()
            
            acc = correct_preds / total_preds if total_preds > 0 else 0
            print(f"\n{Colors.BLUE}Trial Summary: Accuracy = {acc:.1%} ({correct_preds}/{total_preds}){Colors.ENDC}")

            
            print("") # Newline after trial
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Simulation stopped by user.{Colors.ENDC}")

if __name__ == "__main__":
    main()
