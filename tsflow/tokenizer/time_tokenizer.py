from .base import BaseTokenizer
import numpy as np

class TimeDomainTokenizer(BaseTokenizer):
    """
    一个更健壮的实现，假设输入数据已被缩放到 [min_val, max_val] 范围。
    这个类只负责“量化”（Quantization），不负责“缩放”（Scaling）。
    """
    def __init__(self, config):
        super().__init__(config)
        self.time_bins = config.time_bins  # 分箱的数量, e.g., 4096
        self.special_tokens = config.special_tokens
        
        # 假设配置中定义了缩放后的值范围
        # 例如, Chronos 论文中提到了 v_min = -15 和 v_max = 45
        self.min_val = config.min_val 
        self.max_val = config.max_val
        
        # 词汇表大小 = 特殊token + 用于表示时间序列的bin
        self.vocab_size = len(self.special_tokens) + self.time_bins
        
        # 计算每个bin实际代表的值范围大小
        self.bin_size = (self.max_val - self.min_val) / self.time_bins
        
        # 获取起始token的ID，所有时间token都将基于此偏移
        # 假设 special_tokens 是一个字典, e.g., {'pad': 0, 'eos': 1, 'time_start': 2}
        self.time_start_token = self.special_tokens['time_start']

    def tokenize(self, time_series):
        """
        将已经缩放过的时间序列量化为token ID。
        """
        tokens = []
        
        # 将numpy数组用于高效的批量操作
        ts_values = np.array(time_series)
        
        # 1. 裁剪 (Clip): 将值限制在定义的 [min_val, max_val] 范围内
        ts_values = np.clip(ts_values, self.min_val, self.max_val)
        
        # 2. 归一化 (Normalize): 将值从 [min_val, max_val] 映射到 [0, 1]
        normalized_values = (ts_values - self.min_val) / (self.max_val - self.min_val)
        
        # 3. 计算Bin索引 (Calculate Bins): 
        #    将 [0, 1] 范围的值映射到 [0, time_bins-1] 的整数索引
        #    我们乘以 (self.time_bins - 1e-9) 以确保 self.max_val 
        #    本身能被正确映射到最后一个bin (time_bins - 1)，而不是 time_bins。
        bin_indices = (normalized_values * (self.time_bins - 1e-9)).astype(int)
        
        # 4. 偏移 (Offset): 添加特殊token的偏移量
        token_ids = bin_indices + self.time_start_token
        
        return token_ids.tolist()
    
    def detokenize(self, tokens):
        """
        将token ID反量化为时间序列的值（的中心点）。
        注意：这只能还原“量化”的值，不能还原“缩放”前的值。
        """
        time_series = []
        for token in tokens:
            # 检查token是否在时间token的范围内
            if token >= self.time_start_token and token < (self.time_start_token + self.time_bins):
                
                # 1. 移除偏移 (Remove Offset)
                bin_index = token - self.time_start_token
                
                # 2. 计算bin的中心值 (Calculate Bin Center)
                #    (bin_index + 0.5) 是为了取到bin的中点，而不是下限
                normalized_value = (bin_index + 0.5) / self.time_bins
                
                # 3. 反归一化 (De-normalize): 
                #    将 [0, 1] 的值映射回 [min_val, max_val] 范围
                time_value = (normalized_value * (self.max_val - self.min_val)) + self.min_val
                
                time_series.append(time_value)
                
        return time_series