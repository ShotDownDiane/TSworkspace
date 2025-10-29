from statistics import mean
from typing import Optional, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import math

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddConstFeature,
    AddTimeFeatures,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
)
from gluonts.torch.batchify import batchify
from gluonts.dataset.loader import TrainDataLoader, InferenceDataLoader
from gluonts.itertools import Cached
from gluonts.torch.model.estimator import PyTorchLightningEstimator

from tsflow.tokenizer._base import BaseTSTokenizer
from tsflow.tsl_models.DecoderOnlyTransformer import DecoderOnlyTransformer
from tsflow.tokenizer._base import BaseTSTokenizer

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear transformations
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to V
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GPT(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_seq_length=1024,
        dropout=0.1,
        learning_rate=1e-4,
        tokenizer: Optional[BaseTSTokenizer] = None,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate
        self.tokenizer = tokenizer
        
    def forward(self, x, mask=None):
        # Create causal mask if none provided
        if mask is None:
            mask = torch.triu(torch.ones((x.size(1), x.size(1))), diagonal=1).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.to(x.device)
            mask = ~mask
        
        # Embedding and positional encoding
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.final_norm(x)
        x = self.output_projection(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        # Tokenize the input
        # train step, need concate the past_tokens and future_tokens
        input_ts = torch.cat([past_target, future_target], dim=1)
        z_score_mask = torch.ones_like(input_ts, dtype=torch.bool)
        z_score_mask[:, :len(past_target)] = False

        input_tokens,_,_,_ = self.tokenizer.tokenize(input_ts.cpu().numpy(), z_score_mask=z_score_mask)

        # Get predictions
        outputs = self(input_tokens)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            input_tokens.view(-1)
        )
        
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        # Tokenize the input
        # val step, only need the past_tokens
        input_ts = torch.cat([past_target, future_target], dim=1)

        z_score_mask = torch.ones_like(input_ts, dtype=torch.bool)
        z_score_mask[:, :len(past_target)] = False

        input_tokens,_,_,_ = self.tokenizer.encode(input_ts.cpu().numpy(), z_score_mask=z_score_mask)
        
        # Get predictions
        outputs = self(input_tokens)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            input_tokens.view(-1)
        )
        
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def generate(self, input_ids, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get model predictions
                logits = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
        return input_ids


class GPTPredictor(PyTorchPredictor):
    def __init__(self, model: GPT, tokenizer: BaseTSTokenizer) -> None:
        super().__init__(model=model, tokenizer=tokenizer)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        # Tokenize the input
        # generate loop
        input_ts = past_target
        z_score_mask = torch.ones_like(input_ts, dtype=torch.bool)

        input_tokens,mean,std,coeff_lengths = self.tokenizer.encode(input_ts.cpu().numpy(), z_score_mask=z_score_mask)
        
        # Get predictions
        outputs = self.generate(input_tokens, max_length=input_tokens.size(1) + self.prediction_length)

        # Decode the predictions
        future_pred = self.tokenizer.decode(outputs[:, -self.prediction_length:], mean=mean, std=std, coeff_lengths=coeff_lengths)
        
        # Convert to torch tensor
        future_pred = torch.from_numpy(future_pred).to(past_target.device)
        
        return future_pred, future_target


class GPTEstimator(PyTorchLightningEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        vocab_size: int = 1000,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        training_length: Optional[int] = None,
        tokenizer: Optional[BaseTSTokenizer] = None,
        num_batches_per_epoch: int = 100,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        
        self.freq = freq
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.training_length = training_length or context_length
        self.tokenizer = tokenizer or FreqTokenizer(vocab_size=vocab_size)
        self.num_batches_per_epoch = num_batches_per_epoch
        self.batch_size = batch_size

    def create_transformation(self) -> Transformation:

        return Chain(
        [
            AsNumpyArray(
                field=FieldName.TARGET,
                expected_ndim=1,
            ),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            )
        ]
    )

    def create_training_data_loader(self, dataset,module,**kwargs) -> TrainDataLoader:
        train_sampler = ExpectedNumInstanceSampler(
                num_instances=1000,
                min_past=self.context_length,
                min_future=self.prediction_length,
            )

        train_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=train_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        
        return TrainDataLoader(
            Cached(dataset),  # Cache dataset for faster training
            batch_size=self.batch_size,  # Number of samples per batch
            stack_fn=batchify,  # Function to combine samples into batches
            transform=train_splitter,  # Preprocessing transformation
            num_batches_per_epoch=self.num_batches_per_epoch,  # Number of batches per training epoch
        )
    
    def create_validation_data_loader(self, dataset,module,**kwargs) -> TrainDataLoader:
        val_sampler = ValidationSplitSampler(min_future=self.prediction_length)

        val_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=val_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )
        return InferenceDataLoader(
            Cached(dataset),  # Cache dataset for faster training
            batch_size=self.batch_size,  # Number of samples per batch
            stack_fn=batchify,  # Function to combine samples into batches
            transform=val_splitter,  # Preprocessing transformation
            num_batches_per_epoch=self.num_batches_per_epoch,  # Number of batches per training epoch
        )   

    def create_lightning_module(self) -> pl.LightningModule:
        return GPT(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            tokenizer=self.tokenizer,
        )
        

    def create_predictor(
        self,
        transformation: Transformation,
        module: pl.LightningModule,
    ) -> PyTorchPredictor:

        # Create a predictor
        predictor = GPTPredictor(model=module, tokenizer=self.tokenizer)
        copy_parameters(module, predictor)

        prediction_splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            train_sampler=TestSplitSampler(),
            past_length=self.context_length,
            future_length=self.prediction_length,
            time_series_fields=[FieldName.OBSERVED_VALUES],
        )

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            module=predictor,
            batch_size=self.batch_size,
            freq=self.freq,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )