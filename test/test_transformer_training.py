import os
import token
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import pytorch_lightning as pl
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.util import to_pandas
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

from tsflow.PointNet.transformer import GPTEstimator
from tsflow.tokenizer.freq_tokenizer import WaveletTokenizer
from tsflow.utils.transforms import create_transforms

def test_transformer_training():
    # 获取电力数据集
    dataset = get_dataset("electricity")
    
    # 设置模型参数
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    context_length = 2 * prediction_length  # 使用2倍预测长度作为上下文长度
    
    # 创建tokenizer
    tokenizer_config = {
        'wavelet': 'db4',
        'level': 3,
        'time_tokens': 1022,
        'special_tokens': {'PAD': 0, 'EOS': 1, 'TIME_START': 2}
    }
    tokenizer = WaveletTokenizer(tokenizer_config)

    train_transformation = create_transforms(
        context_length=context_length,
        prediction_length=prediction_length,
    )

    train_dataset = train_transformation.apply(dataset.train)

    tokenizer.learn_quantization_params(train_dataset)
    
    # 创建estimator
    estimator = GPTEstimator(
        freq=freq,
        context_length=context_length,
        prediction_length=prediction_length,
        vocab_size=tokenizer.vocab_size,
        d_model=128,  # 使用较小的模型进行测试
        num_layers=3,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        learning_rate=1e-3,
        tokenizer=tokenizer,
        trainer_kwargs={
            "max_epochs": 10,
            "accelerator": "auto",
            "devices": 1,
        },
    )
    
    # 训练模型
    predictor = estimator.train(dataset.train)
    
    # 进行预测
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset.test,
        predictor=predictor,
        num_samples=100,
    )
    
    # 将预测结果转换为列表
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    # 评估结果
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    
    # 打印评估指标
    print("Aggregate metrics:")
    for metric, value in agg_metrics.items():
        print(f"{metric}: {value}")
    
    # 可视化一些预测结果
    test_data = list(dataset.test)
    for idx in range(min(5, len(test_data))):
        plt.figure(figsize=(15, 7))
        
        # 绘制实际值
        target = test_data[idx]["target"]
        plt.plot(target, label="Actual")
        
        # 绘制预测值
        forecast = forecasts[idx]
        mean_forecast = forecast.mean
        plt.plot(range(len(target)-prediction_length, len(target)), 
                mean_forecast, label="Forecast", color="red")
        
        # 添加置信区间
        plt.fill_between(
            range(len(target)-prediction_length, len(target)),
            forecast.quantile(0.1),
            forecast.quantile(0.9),
            color="red",
            alpha=0.1,
            label="80% Confidence Interval"
        )
        
        plt.title(f"Sample forecast {idx+1}")
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        save_dir = Path("test_results")
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / f"forecast_{idx+1}.png")
        plt.close()

if __name__ == "__main__":
    test_transformer_training()