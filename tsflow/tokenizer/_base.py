
import numpy as np


class BaseTSTokenizer:
    def __init__(self, config):
        self.filepath_root = "/share/mas/zhangzhixiang/awd/generator/TSFlow/tsflow/tokenizer/saved" 

    def tokenize(self, text):
        raise NotImplementedError("Subclasses must implement this method.")

    def detokenize(self, tokens):
        raise NotImplementedError("Subclasses must implement this method.")

    def save(self, exp_id):
        filepath = f"{self.filepath_root}/tokenizer_{exp_id}.npz"
        np.savez_compressed(filepath, tokenizer=self)
