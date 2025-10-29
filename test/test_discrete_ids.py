#!/usr/bin/env python3
"""
演示如何使用 VQ-VAE 将连续输入转换为离散 ID
"""

import torch
import numpy as np
from tsflow.module.vqvae import VQVAE


def create_vqvae_model():
    """创建并返回一个 VQ-VAE 模型"""
    config = {
        'block_hidden_size': 128,
        'num_residual_layers': 2,
        'res_hidden_size': 32,
        'embedding_dim': 64,
        'num_embeddings': 512,  # 码本大小，决定了可能的离散 ID 数量
        'commitment_cost': 0.25,
        'compression_factor': 4
    }
    return VQVAE(config)


def input_to_discrete_ids(model, input_data):
    """
    将连续输入转换为离散 ID
    
    Args:
        model: VQ-VAE 模型
        input_data: 输入数据，形状为 (batch_size, sequence_length)
    
    Returns:
        discrete_ids: 离散 ID，形状为 (batch_size, compressed_length)
        quantized: 量化后的连续表示
        perplexity: 困惑度（衡量码本使用情况）
    """
    model.eval()
    with torch.no_grad():
        # 方法1: 使用完整的前向传播
        recon, vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = model(input_data)
        
        # encoding_indices 就是离散 ID
        discrete_ids = encoding_indices
        
        return discrete_ids, quantized, perplexity


def discrete_ids_to_output(model, discrete_ids):
    """
    将离散 ID 转换回连续输出
    
    Args:
        model: VQ-VAE 模型
        discrete_ids: 离散 ID，形状为 (batch_size, compressed_length)
    
    Returns:
        reconstructed: 重构的连续输出
    """
    model.eval()
    with torch.no_grad():
        # 从离散 ID 获取量化向量
        batch_size, seq_len = discrete_ids.shape
        
        # 将 ID 转换为 one-hot 编码
        one_hot = torch.zeros(batch_size, seq_len, model.vq._num_embeddings)
        one_hot.scatter_(2, discrete_ids.unsqueeze(-1), 1)
        
        # 获取对应的嵌入向量
        quantized = torch.matmul(one_hot, model.vq._embedding.weight)
        quantized = quantized.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
        
        # 解码
        reconstructed = model.decode(quantized)
        
        return reconstructed


def demonstrate_tokenization():
    """演示完整的 tokenization 过程"""
    print("=" * 60)
    print("VQ-VAE 离散化演示")
    print("=" * 60)
    
    # 1. 创建模型
    model = create_vqvae_model()
    print(f"✓ 创建 VQ-VAE 模型")
    print(f"  - 码本大小: {model.vq._num_embeddings}")
    print(f"  - 嵌入维度: {model.vq._embedding_dim}")
    print(f"  - 压缩率: {model.compression_factor}x")
    
    # 2. 生成测试数据
    batch_size, seq_length = 4, 256
    # 生成一些示例时间序列
    t = torch.linspace(0, 4*np.pi, seq_length)
    input_data = torch.sin(t.unsqueeze(0) * torch.randn(batch_size, 1) * 2) + 0.1 * torch.randn(batch_size, seq_length)
    
    print(f"\n✓ 生成测试数据")
    print(f"  - 输入形状: {input_data.shape}")
    print(f"  - 输入范围: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    # 3. 转换为离散 ID
    discrete_ids, quantized, perplexity = input_to_discrete_ids(model, input_data)
    
    print(f"\n✓ 转换为离散 ID")
    print(f"  - 离散 ID 形状: {discrete_ids.shape}")
    print(f"  - ID 范围: [{discrete_ids.min()}, {discrete_ids.max()}]")
    print(f"  - 困惑度: {perplexity:.2f}")
    print(f"  - 压缩比: {input_data.shape[-1]} -> {discrete_ids.shape[-1]} ({input_data.shape[-1]/discrete_ids.shape[-1]:.1f}x)")
    
    # 4. 显示具体的 ID 序列
    print(f"\n✓ 第一个样本的离散 ID 序列:")
    ids_sample = discrete_ids[0].numpy()
    print(f"  - 前10个ID: {ids_sample[:10]}")
    print(f"  - 后10个ID: {ids_sample[-10:]}")
    print(f"  - 唯一ID数量: {len(np.unique(ids_sample))}")
    
    # 5. 从离散 ID 重构
    reconstructed = discrete_ids_to_output(model, discrete_ids)
    
    print(f"\n✓ 从离散 ID 重构")
    print(f"  - 重构形状: {reconstructed.shape}")
    print(f"  - 重构范围: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    # 6. 计算重构误差
    mse_error = torch.mean((input_data - reconstructed) ** 2)
    print(f"  - 重构误差 (MSE): {mse_error:.6f}")
    
    return discrete_ids, input_data, reconstructed


def save_and_load_ids():
    """演示如何保存和加载离散 ID"""
    print("\n" + "=" * 60)
    print("离散 ID 保存和加载演示")
    print("=" * 60)
    
    model = create_vqvae_model()
    
    # 生成数据并获取 ID
    input_data = torch.randn(2, 128)
    discrete_ids, _, _ = input_to_discrete_ids(model, input_data)
    
    # 保存 ID（可以保存为各种格式）
    print("✓ 保存离散 ID:")
    
    # 方法1: 保存为 numpy 数组
    np.save('../discrete_ids.npy', discrete_ids.numpy())
    print("  - 保存为 ../discrete_ids.npy 文件")
    
    # 方法2: 保存为文本文件
    with open('../discrete_ids.txt', 'w') as f:
        for i, seq in enumerate(discrete_ids):
            f.write(f"序列{i}: {' '.join(map(str, seq.numpy()))}\n")
    print("  - 保存为 ../discrete_ids.txt 文件")
    
    # 方法3: 保存为 PyTorch tensor
    torch.save(discrete_ids, '../discrete_ids.pt')
    print("  - 保存为 ../discrete_ids.pt 文件")
    
    # 加载并验证
    print("\n✓ 加载离散 ID:")
    loaded_ids = torch.load('../discrete_ids.pt')
    print(f"  - 加载成功，形状: {loaded_ids.shape}")
    print(f"  - 数据一致性: {torch.equal(discrete_ids, loaded_ids)}")
    
    return discrete_ids


def main():
    """主函数"""
    print("🚀 VQ-VAE 离散化完整演示")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 1. 基本的离散化演示
        discrete_ids, input_data, reconstructed = demonstrate_tokenization()
        
        # 2. 保存和加载演示
        save_and_load_ids()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print("\n主要步骤总结:")
        print("1. 连续输入 -> VQ-VAE编码器 -> 连续特征")
        print("2. 连续特征 -> 向量量化 -> 离散ID")
        print("3. 离散ID -> 嵌入查找 -> 量化特征")
        print("4. 量化特征 -> VQ-VAE解码器 -> 重构输出")
        print("\n离散ID可以用于:")
        print("- 序列建模（如语言模型）")
        print("- 数据压缩和存储")
        print("- 离散表示学习")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()