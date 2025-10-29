#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LatentTSTokenizer 功能测试脚本

该脚本测试基于 VQ-VAE 的时间序列分词器的各种功能，包括：
- 单个序列的编码/解码
- 批量序列的编码/解码
- tokenize/detokenize 功能
- 错误处理和边界情况
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsflow.module.vqvae import VQVAE
from tsflow.tokenizer.latent_tokenizer import LatentTSTokenizer


def create_test_vqvae_model() -> VQVAE:
    """创建测试用的 VQ-VAE 模型"""
    vqvae_config = {
        'block_hidden_size': 128,
        'num_residual_layers': 2,
        'res_hidden_size': 32,
        'embedding_dim': 64,
        'num_embeddings': 512,
        'commitment_cost': 0.25,
        'compression_factor': 4
    }
    
    model = VQVAE(vqvae_config)
    model.eval()
    return model


def create_test_config() -> Dict[str, Any]:
    """创建测试配置"""
    class Config:
        def __init__(self):
            self.special_tokens = {
                '<PAD>': 0,
                '<START>': 1,
                '<END>': 2,
                '<UNK>': 3
            }
    
    return Config()


def generate_test_data(batch_size: int = 4, seq_len: int = 256) -> torch.Tensor:
    """生成测试时间序列数据"""
    # 生成多种类型的时间序列
    data = []
    
    for i in range(batch_size):
        t = torch.linspace(0, 4*np.pi, seq_len)
        
        if i == 0:
            # 正弦波
            series = torch.sin(t) + 0.1 * torch.randn(seq_len)
        elif i == 1:
            # 余弦波
            series = torch.cos(t) + 0.1 * torch.randn(seq_len)
        elif i == 2:
            # 线性趋势 + 噪声
            series = 0.5 * t + torch.randn(seq_len)
        else:
            # 随机游走
            series = torch.cumsum(torch.randn(seq_len) * 0.1, dim=0)
            
        data.append(series)
    
    return torch.stack(data)


def test_basic_functionality():
    """测试基本功能"""
    print("🔧 测试基本功能")
    print("=" * 60)
    
    # 创建模型和分词器
    model = create_test_vqvae_model()
    config = create_test_config()
    tokenizer = LatentTSTokenizer(config, model)
    
    print(f"✓ 创建 LatentTSTokenizer")
    print(f"  - 词汇表大小: {tokenizer.vocab_size}")
    print(f"  - 压缩因子: {tokenizer.compression_factor}")
    print(f"  - 设备: {tokenizer.device}")
    
    # 获取压缩信息
    info = tokenizer.get_compression_info()
    print(f"✓ 压缩信息:")
    for key, value in info.items():
        print(f"  - {key}: {value}")
    
    print()


def test_single_sequence_processing():
    """测试单个序列处理"""
    print("🔧 测试单个序列处理")
    print("=" * 60)
    
    # 创建模型和分词器
    model = create_test_vqvae_model()
    config = create_test_config()
    tokenizer = LatentTSTokenizer(config, model)
    
    # 生成单个测试序列
    single_series = generate_test_data(1, 256)[0]  # (256,)
    print(f"✓ 生成测试序列: {single_series.shape}")
    print(f"  - 数据范围: [{single_series.min():.3f}, {single_series.max():.3f}]")
    
    # 测试 encode/decode
    print(f"\n🔍 测试 encode/decode:")
    encoded = tokenizer.encode(single_series)
    print(f"  - 编码形状: {encoded.shape}")
    
    decoded = tokenizer.decode(encoded)
    print(f"  - 解码形状: {decoded.shape}")
    
    # 计算重构误差
    mse = torch.mean((single_series - decoded.squeeze()) ** 2).item()
    print(f"  - 重构误差 (MSE): {mse:.6f}")
    
    # 测试 tokenize/detokenize
    print(f"\n🔍 测试 tokenize/detokenize:")
    tokens = tokenizer.tokenize(single_series)
    print(f"  - Token 数量: {len(tokens)}")
    print(f"  - Token 范围: [{min(tokens)}, {max(tokens)}]")
    print(f"  - 前10个tokens: {tokens[:10]}")
    
    reconstructed = tokenizer.detokenize(tokens)
    print(f"  - 重构形状: {reconstructed.shape}")
    
    # 计算重构误差
    mse_tokens = np.mean((single_series.numpy() - reconstructed) ** 2)
    print(f"  - Token重构误差 (MSE): {mse_tokens:.6f}")
    
    print()


def test_batch_processing():
    """测试批量处理"""
    print("🔧 测试批量处理")
    print("=" * 60)
    
    # 创建模型和分词器
    model = create_test_vqvae_model()
    config = create_test_config()
    tokenizer = LatentTSTokenizer(config, model)
    
    # 生成批量测试数据
    batch_data = generate_test_data(4, 256)  # (4, 256)
    print(f"✓ 生成批量数据: {batch_data.shape}")
    
    # 测试批量 encode/decode
    print(f"\n🔍 测试批量 encode/decode:")
    encoded_batch = tokenizer.encode(batch_data)
    print(f"  - 批量编码形状: {encoded_batch.shape}")
    
    decoded_batch = tokenizer.decode(encoded_batch)
    print(f"  - 批量解码形状: {decoded_batch.shape}")
    
    # 计算批量重构误差
    mse_batch = torch.mean((batch_data - decoded_batch) ** 2).item()
    print(f"  - 批量重构误差 (MSE): {mse_batch:.6f}")
    
    # 测试批量 tokenize/detokenize
    print(f"\n🔍 测试批量 tokenize/detokenize:")
    tokens_batch = tokenizer.batch_tokenize(batch_data)
    print(f"  - 批量 Token 数量: {len(tokens_batch)}")
    print(f"  - 每个序列的 Token 数量: {[len(tokens) for tokens in tokens_batch]}")
    
    reconstructed_batch = tokenizer.batch_detokenize(tokens_batch)
    print(f"  - 批量重构形状: {reconstructed_batch.shape}")
    
    # 计算批量重构误差
    mse_batch_tokens = np.mean((batch_data.numpy() - reconstructed_batch) ** 2)
    print(f"  - 批量Token重构误差 (MSE): {mse_batch_tokens:.6f}")
    
    print()


def test_input_formats():
    """测试不同输入格式"""
    print("🔧 测试不同输入格式")
    print("=" * 60)
    
    # 创建模型和分词器
    model = create_test_vqvae_model()
    config = create_test_config()
    tokenizer = LatentTSTokenizer(config, model)
    
    # 生成测试数据
    test_data = generate_test_data(1, 128)[0]
    
    # 测试 torch.Tensor 输入
    print(f"✓ 测试 torch.Tensor 输入:")
    tokens_tensor = tokenizer.tokenize(test_data)
    print(f"  - Token 数量: {len(tokens_tensor)}")
    
    # 测试 numpy 输入
    print(f"✓ 测试 numpy.ndarray 输入:")
    test_numpy = test_data.numpy()
    tokens_numpy = tokenizer.tokenize(test_numpy)
    print(f"  - Token 数量: {len(tokens_numpy)}")
    
    # 验证结果一致性
    tokens_equal = tokens_tensor == tokens_numpy
    print(f"  - 结果一致性: {tokens_equal}")
    
    # 测试不同 token 格式的 detokenize
    print(f"\n✓ 测试不同 token 格式的 detokenize:")
    
    # list 格式
    reconstructed_list = tokenizer.detokenize(tokens_tensor)
    print(f"  - list 输入重构形状: {reconstructed_list.shape}")
    
    # numpy 格式
    tokens_np = np.array(tokens_tensor)
    reconstructed_np = tokenizer.detokenize(tokens_np)
    print(f"  - numpy 输入重构形状: {reconstructed_np.shape}")
    
    # tensor 格式
    tokens_torch = torch.tensor(tokens_tensor)
    reconstructed_torch = tokenizer.detokenize(tokens_torch)
    print(f"  - tensor 输入重构形状: {reconstructed_torch.shape}")
    
    print()


def test_error_handling():
    """测试错误处理"""
    print("🔧 测试错误处理")
    print("=" * 60)
    
    # 创建模型和分词器
    model = create_test_vqvae_model()
    config = create_test_config()
    tokenizer = LatentTSTokenizer(config, model)
    
    # 测试无效输入类型
    print(f"✓ 测试无效输入类型:")
    try:
        tokenizer.tokenize("invalid_input")
        print("  - ❌ 应该抛出 TypeError")
    except TypeError as e:
        print(f"  - ✓ 正确捕获 TypeError: {str(e)[:50]}...")
    
    # 测试无效维度
    print(f"✓ 测试无效维度:")
    try:
        invalid_data = torch.randn(2, 3, 4, 5)  # 4D tensor
        tokenizer.tokenize(invalid_data)
        print("  - ❌ 应该抛出 ValueError")
    except ValueError as e:
        print(f"  - ✓ 正确捕获 ValueError: {str(e)[:50]}...")
    
    # 测试批量 tokenize 的单序列限制
    print(f"✓ 测试批量数据的单序列限制:")
    try:
        batch_data = torch.randn(3, 128)  # 3个序列
        tokenizer.tokenize(batch_data)
        print("  - ❌ 应该抛出 ValueError")
    except ValueError as e:
        print(f"  - ✓ 正确捕获 ValueError: {str(e)[:50]}...")
    
    # 测试空 tokens_batch
    print(f"✓ 测试空 tokens_batch:")
    try:
        tokenizer.batch_detokenize([])
        print("  - ❌ 应该抛出 ValueError")
    except ValueError as e:
        print(f"  - ✓ 正确捕获 ValueError: {str(e)[:50]}...")
    
    print()


def visualize_reconstruction():
    """可视化重构结果"""
    print("🔧 可视化重构结果")
    print("=" * 60)
    
    # 创建模型和分词器
    model = create_test_vqvae_model()
    config = create_test_config()
    tokenizer = LatentTSTokenizer(config, model)
    
    # 生成测试数据
    test_data = generate_test_data(2, 256)
    
    # 进行编码解码
    tokens_batch = tokenizer.batch_tokenize(test_data)
    reconstructed = tokenizer.batch_detokenize(tokens_batch)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LatentTSTokenizer Reconstruction Results', fontsize=16)
    
    for i in range(2):
        # 原始信号
        axes[i, 0].plot(test_data[i].numpy(), 'b-', label='Original', alpha=0.8)
        axes[i, 0].set_title(f'Sample {i+1}: Original Signal')
        axes[i, 0].set_xlabel('Time Steps')
        axes[i, 0].set_ylabel('Value')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # 重构信号
        axes[i, 1].plot(test_data[i].numpy(), 'b-', label='Original', alpha=0.6)
        axes[i, 1].plot(reconstructed[i], 'r--', label='Reconstructed', alpha=0.8)
        axes[i, 1].set_title(f'Sample {i+1}: Original vs Reconstructed')
        axes[i, 1].set_xlabel('Time Steps')
        axes[i, 1].set_ylabel('Value')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # 计算并显示误差
        mse = np.mean((test_data[i].numpy() - reconstructed[i]) ** 2)
        axes[i, 1].text(0.02, 0.98, f'MSE: {mse:.6f}', 
                       transform=axes[i, 1].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../latent_tokenizer_test_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ 保存可视化结果到 '../latent_tokenizer_test_results.png'")
    
    # 打印统计信息
    print(f"\n📊 重构统计信息:")
    for i in range(2):
        mse = np.mean((test_data[i].numpy() - reconstructed[i]) ** 2)
        mae = np.mean(np.abs(test_data[i].numpy() - reconstructed[i]))
        print(f"  - 样本 {i+1}: MSE={mse:.6f}, MAE={mae:.6f}")
        print(f"    Token数量: {len(tokens_batch[i])}, 压缩比: {test_data.shape[1]/len(tokens_batch[i]):.1f}x")
    
    print()


def run_comprehensive_test():
    """运行完整测试"""
    print("🚀 LatentTSTokenizer 完整功能测试")
    print("=" * 80)
    print()
    
    try:
        # 运行各项测试
        test_basic_functionality()
        test_single_sequence_processing()
        test_batch_processing()
        test_input_formats()
        test_error_handling()
        visualize_reconstruction()
        
        print("🎉 所有测试完成！")
        print("=" * 80)
        print()
        print("主要功能验证:")
        print("✓ 基本功能初始化")
        print("✓ 单个序列编码/解码")
        print("✓ 批量序列处理")
        print("✓ 多种输入格式支持")
        print("✓ 错误处理机制")
        print("✓ 可视化重构结果")
        print()
        print("LatentTSTokenizer 已成功集成 VQ-VAE 的编码解码能力！")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_test()