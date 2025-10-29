from typing import Optional, List, Dict
import numpy as np
import torch
import pytorch_lightning as pl

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
)
from gluonts.torch.model.estimator import PyTorchLightningEstimator

from tsflow.tsl_models.DecoderOnlyTransformer import DecoderOnlyTransformer
from tsflow.tokenizer.freq_tokenizer import FreqTokenizer

class TransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: DecoderOnlyTransformer,
        tokenizer: FreqTokenizer,
        context_length: int,
        prediction_length: int,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        # Tokenize the input
        past_tokens = self.tokenizer.encode(past_target.cpu().numpy())
        future_tokens = self.tokenizer.encode(future_target.cpu().numpy())
        
        # Convert to tensor and move to device
        past_tokens = torch.tensor(past_tokens, device=self.device)
        future_tokens = torch.tensor(future_tokens, device=self.device)
        
        # Concatenate for teacher forcing
        input_tokens = past_tokens
        target_tokens = future_tokens
        
        # Get predictions
        outputs = self(input_tokens)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            target_tokens.view(-1)
        )
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        past_target = batch["past_target"]
        future_target = batch["future_target"]
        
        # Tokenize the input
        past_tokens = self.tokenizer.encode(past_target.cpu().numpy())
        future_tokens = self.tokenizer.encode(future_target.cpu().numpy())
        
        # Convert to tensor and move to device
        past_tokens = torch.tensor(past_tokens, device=self.device)
        future_tokens = torch.tensor(future_tokens, device=self.device)
        
        # Get predictions
        outputs = self(past_tokens)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            future_tokens.view(-1)
        )
        
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class TransformerEstimator(PyTorchLightningEstimator):
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
        tokenizer: Optional[FreqTokenizer] = None,
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

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                InstanceSplitter(
                    target_field=FieldName.TARGET,
                    is_pad_field=FieldName.IS_PAD,
                    start_field=FieldName.START,
                    forecast_start_field=FieldName.FORECAST_START,
                    train_sampler=ValidationSplitSampler(min_future=self.prediction_length),
                    past_length=self.context_length,
                    future_length=self.prediction_length,
                    time_series_fields=[FieldName.OBSERVED_VALUES],
                ),
            ]
        )

    def create_lightning_module(self) -> pl.LightningModule:
        model = DecoderOnlyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
        )
        
        return TransformerLightningModule(
            model=model,
            tokenizer=self.tokenizer,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            learning_rate=self.learning_rate,
        )

    def create_predictor(
        self,
        transformation: Transformation,
        module: pl.LightningModule,
    ) -> PyTorchPredictor:
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
            module=module,
            batch_size=self.batch_size,
            freq=self.freq,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )