#!/usr/bin/env python3
"""
VQ-VAE 测试脚本
测试 VQ-VAE 模型的各个功能：编码、解码、量化、训练等
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tsflow.module.vqvae import VQVAE


def create_test_config():
    """创建测试用的 VQ-VAE 配置"""
    return {
        'block_hidden_size': 128,      # 隐藏层大小
        'num_residual_layers': 2,      # 残差层数量
        'res_hidden_size': 32,         # 残差层隐藏大小
        'embedding_dim': 64,           # 嵌入维度
        'num_embeddings': 512,         # 码本大小
        'commitment_cost': 0.25,       # 承诺损失权重
        'compression_factor': 4        # 压缩率 (4, 8, 12, 16)
    }


def generate_test_data(batch_size=8, seq_length=256):
    """生成测试用的时间序列数据"""
    # 生成多种类型的时间序列
    t = torch.linspace(0, 4*np.pi, seq_length)
    
    # 正弦波 + 噪声
    sine_waves = torch.sin(t.unsqueeze(0) * torch.randn(batch_size//2, 1) * 2) + 0.1 * torch.randn(batch_size//2, seq_length)
    
    # 余弦波 + 趋势
    cos_waves = torch.cos(t.unsqueeze(0) * torch.randn(batch_size//2, 1) * 1.5) + t.unsqueeze(0) * 0.1 + 0.1 * torch.randn(batch_size//2, seq_length)
    
    # 合并数据
    data = torch.cat([sine_waves, cos_waves], dim=0)
    
    # 标准化
    data = (data - data.mean(dim=1, keepdim=True)) / (data.std(dim=1, keepdim=True) + 1e-8)
    
    return data


def test_model_initialization():
    """测试模型初始化"""
    print("=" * 50)
    print("测试 1: 模型初始化")
    print("=" * 50)
    
    config = create_test_config()
    model = VQVAE(config)
    
    print(f"✓ 模型初始化成功")
    print(f"  - 压缩率: {model.compression_factor}")
    print(f"  - 嵌入维度: {config['embedding_dim']}")
    print(f"  - 码本大小: {config['num_embeddings']}")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数数: {total_params:,}")
    print(f"  - 可训练参数数: {trainable_params:,}")
    
    return model


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 50)
    print("测试 2: 前向传播")
    print("=" * 50)
    
    # 生成测试数据
    batch_size, seq_length = 4, 256
    test_data = generate_test_data(batch_size, seq_length)
    
    print(f"输入数据形状: {test_data.shape}")
    
    # 前向传播
    with torch.no_grad():
        recon, vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = model(test_data)
    
    print(f"✓ 前向传播成功")
    print(f"  - 重构数据形状: {recon.shape}")
    print(f"  - 量化损失: {vq_loss.item():.4f}")
    print(f"  - 困惑度: {perplexity.item():.4f}")
    print(f"  - 编码索引形状: {encoding_indices.shape}")
    print(f"  - 量化后形状: {quantized.shape}")
    
    # 计算重构误差
    recon_error = F.mse_loss(recon, test_data)
    print(f"  - 重构误差 (MSE): {recon_error.item():.4f}")
    
    return test_data, recon, encoding_indices


def test_encode_decode(model):
    """测试编码和解码功能"""
    print("\n" + "=" * 50)
    print("测试 3: 编码/解码功能")
    print("=" * 50)
    
    test_data = generate_test_data(2, 128)
    print(f"输入数据形状: {test_data.shape}")
    
    # 测试编码
    with torch.no_grad():
        encoded = model.encode(test_data)
        print(f"✓ 编码成功，输出形状: {encoded.shape}")
        
        # 测试解码
        decoded = model.decode(encoded)
        print(f"✓ 解码成功，输出形状: {decoded.shape}")
        
        # 验证形状一致性
        assert test_data.shape == decoded.shape, f"形状不匹配: {test_data.shape} vs {decoded.shape}"
        print(f"✓ 输入输出形状一致")


def test_different_compression_factors():
    """测试不同压缩率"""
    print("\n" + "=" * 50)
    print("测试 4: 不同压缩率")
    print("=" * 50)
    
    compression_factors = [4, 8, 12, 16]
    test_data = generate_test_data(2, 256)
    
    for cf in compression_factors:
        print(f"\n测试压缩率 {cf}x:")
        config = create_test_config()
        config['compression_factor'] = cf
        
        try:
            model = VQVAE(config)
            with torch.no_grad():
                encoded = model.encode(test_data)
                decoded = model.decode(encoded)
                
            expected_length = test_data.shape[-1] // cf
            actual_length = encoded.shape[-1]
            
            print(f"  ✓ 压缩率 {cf}x 测试成功")
            print(f"    - 原始长度: {test_data.shape[-1]}")
            print(f"    - 编码长度: {actual_length} (期望: {expected_length})")
            print(f"    - 解码形状: {decoded.shape}")
            
        except Exception as e:
            print(f"  ✗ 压缩率 {cf}x 测试失败: {e}")


def test_training_step(model):
    """Test training steps with more iterations to see improvement"""
    print("\n" + "=" * 50)
    print("Test 5: Training Steps")
    print("=" * 50)
    
    # Create optimizer
    optimizer = model.configure_optimizers(lr=1e-3)
    
    # Generate training data
    train_data = generate_test_data(8, 256)
    
    print(f"Training data shape: {train_data.shape}")
    
    # Train for more steps to see improvement
    losses = []
    recon_errors = []
    perplexities = []
    
    print("Training progress:")
    for step in range(50):  # More training steps
        # Use shared_eval method for training
        loss, vq_loss, recon_error, data_recon, perplexity, _, _, _ = model.shared_eval(
            train_data, optimizer, mode='train'
        )
        
        losses.append(loss.item())
        recon_errors.append(recon_error.item())
        perplexities.append(perplexity.item())
        
        if step % 10 == 0:  # Print every 10 steps
            print(f"  Step {step+1:2d}: Loss={loss.item():.4f}, Recon={recon_error.item():.4f}, Perplexity={perplexity.item():.2f}")
    
    print(f"\n✓ Completed 50 training steps")
    print(f"  - Initial loss: {losses[0]:.4f}")
    print(f"  - Final loss: {losses[-1]:.4f}")
    print(f"  - Loss improvement: {losses[0] - losses[-1]:.4f}")
    print(f"  - Initial recon error: {recon_errors[0]:.4f}")
    print(f"  - Final recon error: {recon_errors[-1]:.4f}")
    print(f"  - Recon improvement: {recon_errors[0] - recon_errors[-1]:.4f}")
    
    return train_data, data_recon


def test_evaluation_mode(model):
    """测试评估模式"""
    print("\n" + "=" * 50)
    print("测试 6: 评估模式")
    print("=" * 50)
    
    test_data = generate_test_data(4, 256)
    
    # 测试验证模式
    loss, vq_loss, recon_error, data_recon, perplexity, _, _, _ = model.shared_eval(
        test_data, optimizer=None, mode='val'
    )
    
    print(f"✓ 验证模式测试成功")
    print(f"  - 总损失: {loss.item():.4f}")
    print(f"  - VQ损失: {vq_loss.item():.4f}")
    print(f"  - 重构误差: {recon_error.item():.4f}")
    print(f"  - 困惑度: {perplexity.item():.4f}")
    
    # 测试测试模式
    loss, vq_loss, recon_error, data_recon, perplexity, _, _, _ = model.shared_eval(
        test_data, optimizer=None, mode='test'
    )
    
    print(f"✓ 测试模式测试成功")
    print(f"  - 总损失: {loss.item():.4f}")


def visualize_reconstruction(original, reconstructed, encoding_indices, save_path=None, title_suffix=""):
    """Visualize reconstruction results"""
    print("\n" + "=" * 50)
    print("Test 7: Visualization of Reconstruction Results")
    print("=" * 50)
    
    # Select first two samples for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i in range(2):
        # Original vs Reconstructed signals
        axes[i, 0].plot(original[i].numpy(), label='Original', color='blue', linewidth=2)
        axes[i, 0].plot(reconstructed[i].detach().numpy(), label='Reconstructed', color='red', alpha=0.8, linewidth=2)
        axes[i, 0].set_title(f'Sample {i+1}: Original vs Reconstructed{title_suffix}')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Encoding indices
        axes[i, 1].plot(encoding_indices[i].numpy(), marker='o', markersize=3, linewidth=1)
        axes[i, 1].set_title(f'Sample {i+1}: Encoding Indices')
        axes[i, 1].set_ylabel('Codebook Index')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Reconstruction error over time
        error = (original[i] - reconstructed[i].detach()).numpy()
        axes[i, 2].plot(error, color='green', linewidth=1)
        axes[i, 2].set_title(f'Sample {i+1}: Reconstruction Error')
        axes[i, 2].set_ylabel('Error')
        axes[i, 2].grid(True, alpha=0.3)
        
        # Calculate and display metrics
        mse = np.mean(error**2)
        mae = np.mean(np.abs(error))
        axes[i, 2].text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                       transform=axes[i, 2].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
    else:
        plt.show()
        print(f"✓ Visualization completed")
    
    plt.close()


def run_comprehensive_test():
    """Run comprehensive VQ-VAE tests"""
    print("🚀 Starting Comprehensive VQ-VAE Tests")
    print("=" * 60)
    
    try:
        # 1. Model initialization test
        model = test_model_initialization()
        
        # 2. Forward pass test (before training)
        print("\n" + "=" * 50)
        print("BEFORE TRAINING:")
        print("=" * 50)
        original, reconstructed, encoding_indices = test_forward_pass(model)
        
        # Visualize before training
        visualize_reconstruction(original, reconstructed, encoding_indices, 
                               save_path='../vqvae_before_training.png', 
                               title_suffix=" (Before Training)")
        
        # 3. Encode/decode test
        test_encode_decode(model)
        
        # 4. Different compression factors test
        test_different_compression_factors()
        
        # 5. Training step test
        train_data, trained_recon = test_training_step(model)
        
        # Test forward pass after training
        print("\n" + "=" * 50)
        print("AFTER TRAINING:")
        print("=" * 50)
        with torch.no_grad():
            recon_after, vq_loss_after, quantized_after, perplexity_after, _, encoding_indices_after, _ = model(train_data[:2])
        
        print(f"✓ Forward pass after training")
        print(f"  - VQ loss after training: {vq_loss_after.item():.4f}")
        print(f"  - Perplexity after training: {perplexity_after.item():.4f}")
        
        # Visualize after training
        visualize_reconstruction(train_data[:2], recon_after, encoding_indices_after, 
                               save_path='../vqvae_after_training.png',
                               title_suffix=" (After Training)")
        
        # 6. Evaluation mode test
        test_evaluation_mode(model)
        
        print("\n" + "=" * 60)
        print("🎉 All tests completed! Check the generated images to see improvement:")
        print("   - ../vqvae_before_training.png: Reconstruction before training")
        print("   - ../vqvae_after_training.png: Reconstruction after training")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    run_comprehensive_test()