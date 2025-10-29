#!/usr/bin/env python3
"""
测试脚本：验证WaveletTokenizer的完整tokenize/detokenize流程
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsflow.tokenizer.freq_tokenizer import WaveletTokenizer

def generate_test_data():
    """生成测试用的时间序列数据"""
    # 生成包含多种频率成分的合成信号
    t = np.linspace(0, 4*np.pi, 256)
    
    # 组合信号：低频趋势 + 中频周期 + 高频噪声
    signal = (
        2 * np.sin(0.5 * t) +           # 低频成分
        1.5 * np.sin(3 * t) +          # 中频成分
        0.8 * np.sin(10 * t) +         # 高频成分
        0.3 * np.random.randn(len(t))  # 噪声
    )
    
    return signal.tolist()

def test_wavelet_tokenizer():
    """测试WaveletTokenizer的完整流程"""
    print("=== WaveletTokenizer 测试开始 ===")
    
    # 1. 初始化tokenizer
    config = {
        'wavelet': 'db4',
        'level': 3,
        'time_tokens': 1022,  # 1024 - 2 (PAD, EOS)
        'special_tokens': {'PAD': 0, 'EOS': 1, 'TIME_START': 2},
        'threshold_method': 'soft'
    }
    tokenizer = WaveletTokenizer(config)
    
    print(f"初始化tokenizer: wavelet={tokenizer.wavelet}, level={tokenizer.level}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 2. 生成测试数据
    test_signals = []
    for i in range(5):
        signal = generate_test_data()
        test_signals.append(signal)
    
    print(f"生成 {len(test_signals)} 个测试信号，每个长度: {len(test_signals[0])}")
    
    # 3. 学习量化参数
    print("\n--- 学习量化参数 ---")
    tokenizer.learn_quantization_params(test_signals)
    
    print(f"量化参数:")
    print(f"  min_val: {tokenizer.min_val:.6f}")
    print(f"  max_val: {tokenizer.max_val:.6f}")
    print(f"  bin_size: {tokenizer.bin_size:.6f}")
    print(f"  实际bins数量: {int((tokenizer.max_val - tokenizer.min_val) / tokenizer.bin_size) + 1}")
    
    # 4. 测试tokenize/detokenize流程
    print("\n--- 测试tokenize/detokenize流程 ---")
    
    # 选择第一个信号进行详细测试
    original_signal = test_signals[0]
    print(f"原始信号统计: mean={np.mean(original_signal):.4f}, std={np.std(original_signal):.4f}")
    
    # Tokenize
    tokens, mean, std, coeff_lengths = tokenizer.tokenize(original_signal)
    print(f"Tokenize结果:")
    print(f"  tokens数量: {len(tokens)}")
    print(f"  mean: {mean:.6f}, std: {std:.6f}")
    print(f"  coeff_lengths: {coeff_lengths}")
    print(f"  token范围: [{min(tokens)}, {max(tokens)}]")
    
    # 检查token是否在有效范围内
    time_start_token = tokenizer.special_tokens['TIME_START']
    valid_tokens = [t for t in tokens if time_start_token <= t < time_start_token + tokenizer.time_tokens]
    print(f"  有效tokens比例: {len(valid_tokens)}/{len(tokens)} ({100*len(valid_tokens)/len(tokens):.1f}%)")
    
    # Detokenize
    reconstructed_signal = tokenizer.detokenize(tokens, mean, std, coeff_lengths)
    print(f"Detokenize结果:")
    print(f"  重构信号长度: {len(reconstructed_signal)}")
    print(f"  重构信号统计: mean={np.mean(reconstructed_signal):.4f}, std={np.std(reconstructed_signal):.4f}")
    
    # 5. 计算重构误差
    print("\n--- 重构质量评估 ---")
    
    # 确保长度一致（可能由于小波变换的边界效应导致长度略有不同）
    min_len = min(len(original_signal), len(reconstructed_signal))
    orig_truncated = np.array(original_signal[:min_len])
    recon_truncated = np.array(reconstructed_signal[:min_len])
    
    # 计算各种误差指标
    mse = np.mean((orig_truncated - recon_truncated) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig_truncated - recon_truncated))
    
    # 计算相关系数
    correlation = np.corrcoef(orig_truncated, recon_truncated)[0, 1]
    
    print(f"重构误差指标:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  相关系数: {correlation:.6f}")
    
    # 6. 可视化结果
    print("\n--- 生成可视化图表 ---")
    
    plt.figure(figsize=(15, 10))
    
    # 子图1：原始信号 vs 重构信号
    plt.subplot(2, 2, 1)
    plt.plot(orig_truncated, label='原始信号', alpha=0.8)
    plt.plot(recon_truncated, label='重构信号', alpha=0.8)
    plt.title('原始信号 vs 重构信号')
    plt.legend()
    plt.grid(True)
    
    # 子图2：重构误差
    plt.subplot(2, 2, 2)
    error = orig_truncated - recon_truncated
    plt.plot(error, color='red', alpha=0.7)
    plt.title(f'重构误差 (RMSE: {rmse:.4f})')
    plt.ylabel('误差')
    plt.grid(True)
    
    # 子图3：token分布
    plt.subplot(2, 2, 3)
    plt.hist(tokens, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Token分布')
    plt.xlabel('Token值')
    plt.ylabel('频次')
    plt.grid(True)
    
    # 子图4：频谱比较
    plt.subplot(2, 2, 4)
    freqs_orig = np.fft.fftfreq(len(orig_truncated))
    fft_orig = np.abs(np.fft.fft(orig_truncated))
    fft_recon = np.abs(np.fft.fft(recon_truncated))
    
    plt.plot(freqs_orig[:len(freqs_orig)//2], fft_orig[:len(fft_orig)//2], 
             label='原始频谱', alpha=0.8)
    plt.plot(freqs_orig[:len(freqs_orig)//2], fft_recon[:len(fft_recon)//2], 
             label='重构频谱', alpha=0.8)
    plt.title('频域比较')
    plt.xlabel('频率')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../freq_tokenizer_test_results.png', dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: ../freq_tokenizer_test_results.png")
    
    # 7. 测试多个信号的一致性
    print("\n--- 测试多信号一致性 ---")
    
    reconstruction_errors = []
    for i, signal in enumerate(test_signals):
        tokens, mean, std, coeff_lengths = tokenizer.tokenize(signal)
        reconstructed = tokenizer.detokenize(tokens, mean, std, coeff_lengths)
        
        min_len = min(len(signal), len(reconstructed))
        mse = np.mean((np.array(signal[:min_len]) - np.array(reconstructed[:min_len])) ** 2)
        reconstruction_errors.append(mse)
        
        print(f"信号 {i+1}: MSE = {mse:.6f}, tokens数量 = {len(tokens)}")
    
    print(f"平均重构MSE: {np.mean(reconstruction_errors):.6f}")
    print(f"MSE标准差: {np.std(reconstruction_errors):.6f}")
    
    print("\n=== WaveletTokenizer 测试完成 ===")
    
    # 返回测试结果
    return {
        'tokenizer': tokenizer,
        'reconstruction_rmse': rmse,
        'correlation': correlation,
        'avg_mse': np.mean(reconstruction_errors),
        'vocab_utilization': len(valid_tokens) / len(tokens)
    }

if __name__ == "__main__":
    try:
        results = test_wavelet_tokenizer()
        
        print(f"\n=== 测试总结 ===")
        print(f"重构RMSE: {results['reconstruction_rmse']:.6f}")
        print(f"相关系数: {results['correlation']:.6f}")
        print(f"平均MSE: {results['avg_mse']:.6f}")
        print(f"词汇表利用率: {results['vocab_utilization']:.2%}")
        
        # 判断测试是否通过
        if (results['correlation'] > 0.9 and 
            results['reconstruction_rmse'] < 1.0 and 
            results['vocab_utilization'] > 0.8):
            print("\n✅ 测试通过！WaveletTokenizer实现正确。")
        else:
            print("\n⚠️  测试未完全通过，可能需要进一步优化。")
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()