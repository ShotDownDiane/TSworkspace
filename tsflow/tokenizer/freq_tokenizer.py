import pdb
from statistics import mean
import numpy as np
import pywt
from scipy.stats import iqr
from typing import List, Tuple, Dict, Optional, Union
from ._base import BaseTSTokenizer

class WaveletTokenizer(BaseTSTokenizer):
    """
    基于小波分解和量化（Quantization）的时间序列 Tokenizer。
    
    实现了以下步骤:
    1. 缩放 (Z-score normalization)
    2. 小波分解 (DWT)
    3. 阈值处理 (No-thresholding, CDF-thresholding, VisuShrink, FDRC)
    4. 量化 (使用Freedman-Diaconis规则)
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 从配置中获取信息
        self.wavelet = config.get('wavelet', 'db4')  # 小波基，默认为 db4
        self.level = config.get('level', 3)          # 分解层数
        self.time_tokens = config.get('time_tokens', 1022) # 1024 - 2 (PAD, EOS)
        self.special_tokens: Dict[str, int] = config.get('special_tokens', {'PAD': 0, 'EOS': 1, 'TIME_START': 2})
        self.vocab_size = len(self.special_tokens) + self.time_tokens

        # 阈值处理方法: no_threshold, cdf_threshold, visushrink_soft, visushrink_hard, fdrc
        self.threshold_method = config.get('threshold_method', 'visushrink_soft')

        # === 核心：量化参数（需要在训练时学习并固定） ===
        # min_val/max_val: 整个训练集小波系数的最小/最大值
        self.min_val = config.get('min_val', None) 
        self.max_val = config.get('max_val', None) 
        # bin_size: Freedman-Diaconis 规则计算出的 bin 宽度
        self.bin_size = config.get('bin_size', None)
        # bin边界
        self.bin_edges = config.get('bin_edges', None)
            
        self.time_start_token = self.special_tokens['TIME_START']
        self.pad_token_id = self.special_tokens['PAD']
        self.eos_token_id = self.special_tokens['EOS']
        
        # 实际用于量化的 bin 数量（可能会比 config.time_tokens 少）
        if self.bin_size is not None and self.min_val is not None and self.max_val is not None:
             self.actual_bins = int(np.ceil((self.max_val - self.min_val) / self.bin_size))
             if self.actual_bins > self.time_tokens:
                 print(f"警告：计算出的 bin 数量 {self.actual_bins} 超过了设定的最大 token数 {self.time_tokens}。将裁剪到最大值。")
                 self.actual_bins = self.time_tokens
        else:
            self.actual_bins = self.time_tokens


    # --- 阶段 0：学习量化参数（只在训练时运行一次） ---
    def learn_quantization_params(self, training_data: List[List[float]]):
        """
        根据训练数据的经验分布，计算量化所需的 min_val, max_val 和 bin_size。
        使用Freedman-Diaconis规则优化bin宽度，以最小化重构误差。
        """
        all_coeffs = []
        for ins in training_data:
            # 1. 缩放 (Z-score)
            ts = np.concatenate((ins['past_target'],ins['future_target']))
            
            ts_scaled, _, _ = self._z_score_normalize(ts)
            
            # 2. 小波分解 (DWT)
            level = min(self.level, pywt.dwt_max_level(ts_scaled.shape[0], pywt.Wavelet(self.wavelet).dec_len))
            coeffs = pywt.wavedec(ts_scaled, self.wavelet, level=level, mode='periodization')
            
            # 3. 阈值处理
            coeffs_thresholded = self._apply_thresholding(coeffs)

            # 将所有系数展平收集
            for c in coeffs_thresholded:
                all_coeffs.extend(c.tolist())
        
        coeffs_array = np.array(all_coeffs)
        
        # 计算 min_val 和 max_val (取 1% 和 99% 分位数以避免极端值污染)
        self.min_val = np.percentile(coeffs_array, 1)
        self.max_val = np.percentile(coeffs_array, 99)

        # 使用Freedman-Diaconis规则计算最优bin宽度
        N = len(coeffs_array)
        IQR = iqr(coeffs_array)
        
        if IQR > 0:
            # Freedman-Diaconis规则: bin_size = 2 * IQR / (N^(1/3))
            fd_bin_size = 2 * IQR / (N ** (1/3))
            fd_bins = int(np.ceil((self.max_val - self.min_val) / fd_bin_size))
            
            # 如果FD规则产生的bins数量在合理范围内，使用FD规则
            if fd_bins <= self.time_tokens:
                self.bin_size = fd_bin_size
                print(f"使用Freedman-Diaconis规则: {fd_bins} bins")
            else:
                # 否则，直接使用均匀分布
                self.bin_size = (self.max_val - self.min_val) / self.time_tokens
                print(f"FD规则产生{fd_bins} bins超过限制，使用均匀分布: {self.time_tokens} bins")
        else:
            # 如果IQR为0，使用均匀分布
            self.bin_size = (self.max_val - self.min_val) / self.time_tokens
            print("IQR为0，使用均匀分布")
            
        # 确保 bin size 不为零
        if self.bin_size < 1e-9:
            self.bin_size = (self.max_val - self.min_val) / self.time_tokens
            
        # 计算实际 bin 数量和边界
        self.actual_bins = min(int(np.ceil((self.max_val - self.min_val) / self.bin_size)), self.time_tokens)
        
        # 计算bin边界
        self.bin_edges = np.linspace(self.min_val, self.max_val, self.actual_bins + 1)
        
        print(f"量化参数学习完成：")
        print(f"  Bin Size: {self.bin_size:.6f}")
        print(f"  Min Val: {self.min_val:.4f}")
        print(f"  Max Val: {self.max_val:.4f}")
        print(f"  实际Bins: {self.actual_bins}")
        print(f"  总词汇表大小: {len(self.special_tokens) + self.actual_bins}")


    # --- 辅助方法 ---
    def _z_score_normalize(self, ts: np.ndarray, z_score_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, float]:
        """
        1. 缩放: Z-score 标准化。
        """
        if z_score_mask is None:
            z_score_mask = np.ones(ts.shape, dtype=bool)
        mean = np.mean(ts[z_score_mask])
        std = np.std(ts[z_score_mask]) 
        # 确保标准差不为零
        std_safe = std if std != 0 else 1.0
        ts_scaled = (ts - mean) / std_safe
        return ts_scaled, mean, std
    
    def _apply_thresholding(self, coeffs: List[np.ndarray]) -> List[np.ndarray]:
        """
        3. 阈值处理: 根据选择的方法对小波系数进行阈值处理。
        
        支持的方法:
        - no_threshold: 不进行阈值处理
        - cdf_threshold: 基于系数经验分布的阈值处理
        - visushrink_soft: VisuShrink软阈值处理
        - visushrink_hard: VisuShrink硬阈值处理
        - fdrc: False Discovery Rate Control阈值处理
        """
        # 保留近似系数不变
        approx_coeffs = coeffs[0]
        detail_coeffs = coeffs[1:]
        
        # 根据不同方法处理细节系数
        if self.threshold_method == 'no_threshold':
            # 不进行阈值处理
            return coeffs
            
        elif self.threshold_method == 'cdf_threshold':
            # CDF阈值处理: 根据系数在各层的分布设置阈值
            thresholded_details = []
            for j, d_coeffs in enumerate(detail_coeffs):
                # 计算当前层的阈值 (基于分位数)
                # 阈值随层数指数增长: b^(J-j+1)，其中J是总层数，j是当前层
                level_idx = len(detail_coeffs) - j
                cutoff_percentile = min(5 * 2**(level_idx), 50)  # 指数增长但不超过50%
                threshold = np.percentile(np.abs(d_coeffs), cutoff_percentile)
                
                # 应用阈值
                mask = np.abs(d_coeffs) <= threshold
                thresholded = d_coeffs.copy()
                thresholded[mask] = 0
                thresholded_details.append(thresholded)
                
        elif self.threshold_method.startswith('visushrink'):
            # VisuShrink阈值处理 (Donoho, 1995)
            thresholded_details = []
            for d_coeffs in detail_coeffs:
                # 计算VisuShrink阈值: sigma * sqrt(2*log(N))
                # 使用MAD估计噪声标准差: sigma = median(|d|) / 0.6745
                sigma = np.median(np.abs(d_coeffs)) / 0.6745
                N = len(d_coeffs)
                threshold = sigma * np.sqrt(2 * np.log(N)) if N > 0 else 0
                
                if self.threshold_method == 'visushrink_soft':
                    # 软阈值: sign(x) * max(|x| - threshold, 0)
                    thresholded = pywt.threshold(d_coeffs, threshold, mode='soft')
                else:  # visushrink_hard
                    # 硬阈值: x if |x| > threshold else 0
                    thresholded = pywt.threshold(d_coeffs, threshold, mode='hard')
                    
                thresholded_details.append(thresholded)
                
        elif self.threshold_method == 'fdrc':
            # FDRC (False Discovery Rate Control) 阈值处理
            thresholded_details = []
            for d_coeffs in detail_coeffs:
                thresholded = self._apply_fdrc(d_coeffs)
                thresholded_details.append(thresholded)
        
        else:
            # 默认不进行阈值处理
            return coeffs
            
        # 组合近似系数和处理后的细节系数
        return [approx_coeffs] + thresholded_details
    
    def _apply_fdrc(self, coeffs: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        应用FDRC (False Discovery Rate Control) 阈值处理。
        
        参考: Abramovich & Benjamini (1996)
        """
        N = len(coeffs)
        if N == 0:
            return coeffs
            
        # 1. 计算系数的p值 (基于标准正态分布)
        # 使用MAD估计噪声标准差
        sigma = np.median(np.abs(coeffs)) / 0.6745
        if sigma == 0:
            return coeffs
            
        z_scores = np.abs(coeffs) / sigma
        p_values = 2 * (1 - np.abs(np.minimum(np.maximum(z_scores, -8), 8)))
        
        # 2. 排序p值
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # 3. 找到最大的k使得p_(k) <= (k/N) * alpha
        k = 0
        for i in range(N):
            if sorted_p[i] <= ((i + 1) / N) * alpha:
                k = i + 1
                
        # 4. 如果没有找到这样的k，则所有系数都被阈值处理为0
        if k == 0:
            return np.zeros_like(coeffs)
            
        # 5. 设置阈值为第k个排序后的系数的绝对值
        threshold = np.abs(coeffs[sorted_idx[k-1]])
        
        # 6. 应用硬阈值
        thresholded = pywt.threshold(coeffs, threshold, mode='hard')
        
        return thresholded

    # 兼容性方法
    def tokenize(self, time_series: List[float], z_score_mask: Optional[np.ndarray] = None) -> Tuple[List[int], float, float, List[int]]:
        """
        将时间序列编码为token序列。
        
        步骤:
        1. Z-score标准化
        2. 小波分解 (DWT)
        3. 阈值处理
        4. 量化为离散token
        """
        if self.min_val is None or self.bin_size is None or self.bin_edges is None:
            raise ValueError("量化参数未初始化。请先调用 learn_quantization_params。")

        ts_array = np.array(time_series, dtype=np.float64)
        
        # 1. 缩放 (Z-score)
        ts_scaled, mean, std = self._z_score_normalize(ts_array, z_score_mask)
        
        # 2. 小波分解 (DWT)
        level = min(self.level, pywt.dwt_max_level(len(ts_scaled), pywt.Wavelet(self.wavelet).dec_len))
        coeffs = pywt.wavedec(ts_scaled, self.wavelet, level=level, mode='periodization')
        coeff_lengths = [len(c) for c in coeffs]    
        
        # 3. 阈值处理
        coeffs_thresholded = self._apply_thresholding(coeffs)
        
        # 4. 量化
        # 将系数打平成单数组
        flat_coeffs = np.concatenate([c.flatten() for c in coeffs_thresholded])
        
        # 处理NaN/Inf
        mask_nan = ~np.isfinite(flat_coeffs)
        flat_coeffs_safe = flat_coeffs.copy()
        flat_coeffs_safe[mask_nan] = 0.0
        
        # 数字化: 将系数映射到bin索引
        digitized = np.digitize(flat_coeffs_safe, self.bin_edges) - 1
        digitized = np.clip(digitized, 0, len(self.bin_edges) - 2)
        
        # 添加起始偏移，设置PAD
        tokens = digitized + self.time_start_token
        tokens[mask_nan] = self.pad_token_id
        
        # 添加EOS标记
        tokens = np.append(tokens, self.eos_token_id).tolist()
        
        return tokens, mean, std, coeff_lengths
    
    def detokenize(self, tokens: List[int], mean: float, std: float, coeff_lengths: List[int]) -> List[float]:
        """
        兼容性方法，执行解码并应用保存的均值和标准差。
        """
        # 解码
        if self.min_val is None or self.bin_size is None or self.bin_edges is None:
            raise ValueError("量化参数未初始化。请先调用 learn_quantization_params。")
            
        # 移除EOS标记
        if tokens and tokens[-1] == self.eos_token_id:
            tokens = tokens[:-1]
            
        # 将tokens转换为numpy数组
        tokens_array = np.array(tokens)
        
        # 1. 反量化：将token转换回系数值
        # 移除偏移，得到bin索引
        bin_indices = tokens_array.copy()
        bin_indices[tokens_array >= self.time_start_token] -= self.time_start_token
        bin_indices[tokens_array < self.time_start_token] = 0  # 处理特殊token
        
        # 计算bin的中心值
        bin_centers = self.min_val + (bin_indices + 0.5) * self.bin_size
        
        # 处理PAD标记
        bin_centers[tokens_array == self.pad_token_id] = 0
        
        # 如果没有提供原始形状信息，尝试估计一个合理的形状
        if original_shape is None:
            # 估计原始时间序列长度
            est_length = len(bin_centers)
            # 估计小波分解层数
            est_level = min(self.level, pywt.dwt_max_level(est_length, pywt.Wavelet(self.wavelet).dec_len))
            # 创建一个临时时间序列进行分解，获取系数形状
            temp_ts = np.zeros(est_length)
            temp_coeffs = pywt.wavedec(temp_ts, self.wavelet, level=est_level, mode='periodization')
            original_shape = [len(c) for c in temp_coeffs]
        
        # 2. 重构小波系数
        reconstructed_coeffs = []
        start_idx = 0
        for length in original_shape:
            end_idx = start_idx + length
            if end_idx <= len(bin_centers):
                reconstructed_coeffs.append(bin_centers[start_idx:end_idx])
            else:
                # 如果长度不匹配，用零填充
                coeff_part = np.zeros(length)
                remaining_len = len(bin_centers) - start_idx
                if remaining_len > 0:
                    coeff_part[:remaining_len] = bin_centers[start_idx:]
                reconstructed_coeffs.append(coeff_part)
            start_idx = end_idx
        
        # 3. 小波逆变换
        try:
            ts_scaled = pywt.waverec(reconstructed_coeffs, self.wavelet, mode='periodization')
        except Exception as e:
            # 如果逆变换失败，尝试修复系数长度
            print(f"警告：小波逆变换失败 ({e})，尝试修复系数长度...")
            # 使用一个参考长度来重新构建系数
            est_length = sum(original_shape)
            temp_ts = np.zeros(est_length)
            temp_coeffs = pywt.wavedec(temp_ts, self.wavelet, level=self.level, mode='periodization')
            
            # 用实际系数替换临时系数
            for i, (temp_coeff, recon_coeff) in enumerate(zip(temp_coeffs, reconstructed_coeffs)):
                min_len = min(len(temp_coeff), len(recon_coeff))
                temp_coeffs[i][:min_len] = recon_coeff[:min_len]
            
            ts_scaled = pywt.waverec(temp_coeffs, self.wavelet, mode='periodization')
        
        # 反标准化
        std_safe = std if std != 0 else 1.0
        original_ts = (np.array(ts_scaled) * std_safe) + mean
        
        return original_ts.tolist()