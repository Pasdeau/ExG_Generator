#!/usr/bin/env python3
"""
Stress Test for Real-Time ENG Noise Detection
模拟 ADS1298 持续数据流，验证模型在长时间运行下的稳定性
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from noise_detection.realtime import RealTimeNoiseDetector
from eng_surface_sim import simulate, SimConfig

def test_continuous_streaming(duration_minutes=5, fs=8000):
    """
    模拟持续数据流并监控性能
    
    Args:
        duration_minutes: 测试持续时间 (分钟)
        fs: 采样率 (Hz)
    """
    print("="*60)
    print("🔬 实时模型压力测试")
    print("="*60)
    print(f"采样率: {fs} Hz")
    print(f"测试时长: {duration_minutes} 分钟")
    print(f"预计样本数: {int(duration_minutes * 60 * fs):,}")
    print("="*60)
    
    # 初始化模型
    model_path = "noise_detection/checkpoints/eng_best_model.pth"
    detector = RealTimeNoiseDetector(model_path, fs=fs, window_dur=2.0, step_dur=0.1)
    
    # 统计指标
    total_samples = 0
    inference_count = 0
    inference_times = []
    noise_probabilities = []
    
    # 模拟数据生成器
    def data_generator():
        """持续生成合成 ENG 信号"""
        seed = 100
        while True:
            cfg = SimConfig(fs=fs, duration_s=5.0, seed=seed)
            t, raw, proc, meta, events, artifact_labels, clean, noise_only = simulate(cfg)
            for sample in raw:
                yield sample
            seed += 1
    
    gen = data_generator()
    
    print("\n开始测试...")
    start_time = time.time()
    target_duration = duration_minutes * 60
    
    last_report_time = start_time
    report_interval = 10  # 每10秒报告一次
    
    try:
        while (time.time() - start_time) < target_duration:
            # 获取下一个样本
            sample = next(gen)
            
            # 计时推理
            infer_start = time.time()
            prob = detector.process(sample)
            infer_elapsed = time.time() - infer_start
            
            total_samples += 1
            
            # 记录推理（只记录实际执行推理的情况）
            if detector.samples_since_infer == 1:  # 刚执行过推理
                inference_count += 1
                inference_times.append(infer_elapsed * 1000)  # ms
                noise_probabilities.append(prob)
            
            # 定期报告
            if time.time() - last_report_time > report_interval:
                elapsed = time.time() - start_time
                samples_per_sec = total_samples / elapsed
                
                print(f"\n[{elapsed:.1f}s] 状态报告:")
                print(f"  └─ 总样本数: {total_samples:,}")
                print(f"  └─ 推理次数: {inference_count}")
                print(f"  └─ 实时速率: {samples_per_sec:.0f} samples/s (目标: {fs})")
                
                if inference_times:
                    avg_infer = np.mean(inference_times[-100:])
                    max_infer = np.max(inference_times[-100:])
                    print(f"  └─ 推理延迟: {avg_infer:.2f}ms (avg), {max_infer:.2f}ms (max)")
                
                if noise_probabilities:
                    avg_prob = np.mean(noise_probabilities[-100:])
                    print(f"  └─ 噪声概率: {avg_prob:.3f} (近期平均)")
                
                last_report_time = time.time()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断测试")
    
    # 最终统计
    elapsed_total = time.time() - start_time
    print("\n" + "="*60)
    print("📊 测试完成 - 最终统计")
    print("="*60)
    print(f"运行时长: {elapsed_total:.2f} 秒")
    print(f"总样本数: {total_samples:,}")
    print(f"推理次数: {inference_count}")
    print(f"平均吞吐: {total_samples / elapsed_total:.0f} samples/s")
    print(f"理论目标: {fs} samples/s")
    
    if inference_times:
        print(f"\n推理性能:")
        print(f"  └─ 平均延迟: {np.mean(inference_times):.2f} ms")
        print(f"  └─ 最大延迟: {np.max(inference_times):.2f} ms")
        print(f"  └─ 最小延迟: {np.min(inference_times):.2f} ms")
        print(f"  └─ 标准差: {np.std(inference_times):.2f} ms")
    
    # 性能评估
    print(f"\n✅ 性能评估:")
    actual_rate = total_samples / elapsed_total
    if actual_rate >= fs * 0.95:
        print(f"  [通过] 能够实时处理 (达到 {actual_rate/fs*100:.1f}% 目标速率)")
    else:
        print(f"  [警告] 可能跟不上实时速率 (仅达到 {actual_rate/fs*100:.1f}%)")
    
    max_allowed_latency = 1000 / fs  # 单样本间隔时间 (ms)
    if inference_times and np.mean(inference_times) < max_allowed_latency * 100:
        print(f"  [通过] 推理延迟合理 ({np.mean(inference_times):.2f}ms)")
    
    print("="*60)
    
    return {
        'total_samples': total_samples,
        'inference_count': inference_count,
        'avg_latency_ms': np.mean(inference_times) if inference_times else 0,
        'throughput_sps': actual_rate
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time model stress test")
    parser.add_argument('--duration', type=float, default=1.0, 
                        help='Test duration in minutes (default: 1)')
    parser.add_argument('--fs', type=int, default=8000,
                        help='Sampling rate in Hz (default: 8000)')
    args = parser.parse_args()
    
    results = test_continuous_streaming(duration_minutes=args.duration, fs=args.fs)
