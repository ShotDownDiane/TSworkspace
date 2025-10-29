# VQ-VAE tokenizer for TSFlow

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, List, Optional, Tuple
from tsflow.tokenizer._base import BaseTSTokenizer
from tsflow.module.vqvae import VQVAE


class LatentTSTokenizer(BaseTSTokenizer):
    """
    基于 VQ-VAE 的时间序列分词器
    
    该分词器使用 VQ-VAE 模型将连续的时间序列数据转换为离散的 token 序列，
    并支持从 token 序列重构回原始时间序列。
    """
    
    def __init__(self, config: dict, vq_vae_model: VQVAE):
        """
        初始化 LatentTSTokenizer
        
        Args:
            config: 配置字典，包含特殊 token 等信息
            vq_vae_model: 训练好的 VQ-VAE 模型
        """
        super().__init__(config)
        self.vq_vae_model = vq_vae_model
        self.vq_vae_model.eval()  # 设置为评估模式
        
        # 获取配置信息
        self.special_tokens = getattr(config, 'special_tokens', {})
        self.device = next(vq_vae_model.parameters()).device
        
        # 计算词汇表大小
        self.vocab_size = vq_vae_model.vq._num_embeddings + len(self.special_tokens)
        self.compression_factor = vq_vae_model.compression_factor
        
    def _validate_input(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """验证和预处理输入数据"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        elif not isinstance(data, torch.Tensor):
            raise TypeError(f"输入数据必须是 torch.Tensor 或 np.ndarray，得到 {type(data)}")
            
        # 确保数据在正确的设备上
        data = data.to(self.device)
        
        # 确保数据是 2D 的 (batch_size, sequence_length)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        elif data.dim() > 2:
            raise ValueError(f"输入数据维度应为 1D 或 2D，得到 {data.dim()}D")
            
        return data

    def encode(self, time_series: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        将时间序列编码为连续的潜在表示
        
        Args:
            time_series: 输入时间序列，形状为 (batch_size, sequence_length) 或 (sequence_length,)
            
        Returns:
            encoded: 编码后的潜在表示，形状为 (batch_size, embedding_dim, compressed_length)
        """
        time_series = self._validate_input(time_series)
        
        with torch.no_grad():
            encoded = self.vq_vae_model.encode(time_series)
            
        return encoded
    
    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        将潜在表示解码为时间序列
        
        Args:
            encoded: 潜在表示，形状为 (batch_size, embedding_dim, compressed_length)
            
        Returns:
            decoded: 解码后的时间序列，形状为 (batch_size, sequence_length)
        """
        with torch.no_grad():
            decoded = self.vq_vae_model.decode(encoded)
            
        return decoded

    def tokenize(self, time_series: Union[torch.Tensor, np.ndarray]) -> List[int]:
        """
        将时间序列转换为离散 token 序列
        
        Args:
            time_series: 输入时间序列，形状为 (sequence_length,) 或 (1, sequence_length)
            
        Returns:
            tokens: 离散 token 列表
        """
        time_series = self._validate_input(time_series)
        
        # 确保是单个序列
        if time_series.shape[0] != 1:
            raise ValueError("tokenize 方法只支持单个序列，请使用 batch_tokenize 处理批量数据")
        
        with torch.no_grad():
            # 编码
            encoded = self.vq_vae_model.encode(time_series)
            
            # 向量量化
            vq_loss, quantized, perplexity, _, encoding_indices, _ = self.vq_vae_model.vq(encoded)
            
            # 转换为 token 列表
            tokens = encoding_indices.squeeze(0).tolist()
            
        return tokens

    def detokenize(self, tokens: List[int]) -> np.ndarray:
        """
        将离散 token 序列转换回时间序列
        
        Args:
            tokens: 离散 token 列表
            
        Returns:
            reconstructed: 重构的时间序列，numpy 数组
        """
        if not isinstance(tokens, (list, np.ndarray, torch.Tensor)):
            raise TypeError(f"tokens 必须是 list、np.ndarray 或 torch.Tensor，得到 {type(tokens)}")
            
        # 转换为 tensor
        if isinstance(tokens, list):
            indices = torch.tensor(tokens, dtype=torch.long, device=self.device)
        elif isinstance(tokens, np.ndarray):
            indices = torch.from_numpy(tokens).long().to(self.device)
        else:
            indices = tokens.long().to(self.device)
            
        # 确保是 2D 张量 (1, sequence_length)
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)
            
        with torch.no_grad():
            batch_size, seq_len = indices.shape
            
            # 将 ID 转换为 one-hot 编码
            one_hot = torch.zeros(batch_size, seq_len, self.vq_vae_model.vq._num_embeddings, device=self.device)
            one_hot.scatter_(2, indices.unsqueeze(-1), 1)
            
            # 获取对应的嵌入向量
            quantized = torch.matmul(one_hot, self.vq_vae_model.vq._embedding.weight)
            quantized = quantized.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
            
            # 解码
            reconstructed = self.vq_vae_model.decode(quantized)
            
        return reconstructed.squeeze(0).cpu().numpy()
    
    def batch_tokenize(self, time_series_batch: Union[torch.Tensor, np.ndarray]) -> List[List[int]]:
        """
        批量处理时间序列转换为 token 序列
        
        Args:
            time_series_batch: 批量时间序列，形状为 (batch_size, sequence_length)
            
        Returns:
            tokens_batch: 批量 token 列表
        """
        time_series_batch = self._validate_input(time_series_batch)
        
        with torch.no_grad():
            # 编码
            encoded = self.vq_vae_model.encode(time_series_batch)
            
            # 向量量化
            vq_loss, quantized, perplexity, _, encoding_indices, _ = self.vq_vae_model.vq(encoded)
            
            # 转换为 token 列表
            tokens_batch = []
            for i in range(encoding_indices.shape[0]):
                tokens = encoding_indices[i].tolist()
                tokens_batch.append(tokens)
                
        return tokens_batch
    
    def batch_detokenize(self, tokens_batch: List[List[int]]) -> np.ndarray:
        """
        批量处理 token 序列转换回时间序列
        
        Args:
            tokens_batch: 批量 token 列表
            
        Returns:
            reconstructed_batch: 重构的时间序列批量，numpy 数组
        """
        if not tokens_batch:
            raise ValueError("tokens_batch 不能为空")
            
        # 转换为 tensor
        max_len = max(len(tokens) for tokens in tokens_batch)
        batch_size = len(tokens_batch)
        
        indices = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        for i, tokens in enumerate(tokens_batch):
            indices[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            
        with torch.no_grad():
            # 将 ID 转换为 one-hot 编码
            one_hot = torch.zeros(batch_size, max_len, self.vq_vae_model.vq._num_embeddings, device=self.device)
            one_hot.scatter_(2, indices.unsqueeze(-1), 1)
            
            # 获取对应的嵌入向量
            quantized = torch.matmul(one_hot, self.vq_vae_model.vq._embedding.weight)
            quantized = quantized.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)
            
            # 解码
            reconstructed = self.vq_vae_model.decode(quantized)
            
        return reconstructed.cpu().numpy()
    
    def get_compression_info(self) -> dict:
        """获取压缩相关信息"""
        return {
            'compression_factor': self.compression_factor,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.vq_vae_model.vq._embedding_dim,
            'num_embeddings': self.vq_vae_model.vq._num_embeddings,
            'device': str(self.device)
        }