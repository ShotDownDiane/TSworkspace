# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import partial
import math
import logging
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from tqdm.auto import tqdm

from gluonts.mx import DeepAREstimator, TransformerEstimator
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.time_feature import (
    get_lags_for_frequency,
    time_features_from_frequency_str,
)
from gluonts.dataset.split import slice_data_entry
from gluonts.transform import AdhocTransform, Chain


from tsflow.utils import (
    ScaleAndAddMeanFeature,
    ScaleAndAddMinMaxFeature,
    GluonTSNumpyDataset,
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    get_next_file_num,
)
from tsflow.model import TSFlowUncond,LinearEstimator,TSFlowPS
from tsflow.dataset import get_gts_dataset
from tsflow.metrics import linear_pred_score, wasserstein

DOWNSTREAM_MODELS = ["linear"]


def load_model(config):
    setting = config["setting"]
    target_dim = config["model_params"]["target_dim"]
    model_params = config["model_params"]
    model = TSFlowPS(
        setting=setting,
        target_dim=target_dim,
        context_length=model_params["context_length"],
        prediction_length=model_params["prediction_length"],
        backbone_params=model_params["backbone_params"],
        prior_params=model_params["prior_params"],
        optimizer_params=model_params["optimizer_params"],
        ema_params=model_params["ema_params"],
        frequency=model_params["freq"],
        normalization=model_params["normalization"],
        use_lags=model_params["use_lags"],
        use_ema=model_params["use_ema"],
        num_steps=model_params["num_steps"],
        solver=model_params["solver"],
        matching=model_params["matching"]
    )

    model.load_state_dict(
        torch.load(config["ckpt"], map_location="cpu")['state_dict'],
        strict=True,
    )
    model = model.to(config["device"])
    return model


def sample_synthetic(
    model: TSFlowPS,
    num_samples: int = 10_000,
    batch_size: int = 1000,
):
    synth_samples = []

    n_iters = math.ceil(num_samples / batch_size)
    for _ in tqdm(range(n_iters)):
        samples = model.sample_n(num_samples=batch_size)
        synth_samples.append(samples)

    synth_samples = np.concatenate(synth_samples, axis=0)[:num_samples]

    return synth_samples


def sample_real(
    data_loader,
    n_timesteps: int,
    num_samples: int = 10_000,
    batch_size: int = 1000,
):
    real_samples = []
    data_iter = iter(data_loader)
    n_iters = math.ceil(num_samples / batch_size)
    for _ in tqdm(range(n_iters)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        ts = np.concatenate(
            [batch["past_target"], batch["future_target"]], axis=-1
        )[:, -n_timesteps:]
        real_samples.append(ts)

    real_samples = np.concatenate(real_samples, axis=0)[:num_samples]

    return real_samples


def evaluate_tstr(
    tstr_predictor,
    test_dataset,
    context_length,
    prediction_length,
    num_samples=100,
):
    total_length = context_length + prediction_length
    # Slice test set to be of the same length as context_length + prediction_length
    slice_func = partial(slice_data_entry, slice_=slice(-total_length, None))

    transformation = Chain(
        [
            AdhocTransform(slice_func),
            # Add scale to data entry for use later during evaluation

        ]
    )
    sliced_test_set = transformation.apply(test_dataset)

    fcst_iter, ts_iter = make_evaluation_predictions(
        dataset=sliced_test_set,
        predictor=tstr_predictor,
        num_samples=num_samples,

    )
    evaluator = Evaluator(num_workers=1)
    metrics, _ = evaluator(list(ts_iter), list(fcst_iter))
    
    return filter_metrics(metrics)


def train_and_evaluate(
    dataset,
    model_name,
    synth_samples,
    real_samples,
    config
):
    # NOTE: There's no notion of time for synthetic time series,
    # they are just "sequences".
    # A dummy timestamp is used for start time in synthetic time series.
    # Hence, time_features are set to [] in the models below.
    model_name = model_name.lower()
    freq = dataset.metadata.freq
    context_length = config["model_params"]["context_length"]
    prediction_length = config["model_params"]["prediction_length"]
    total_length = context_length + prediction_length
    if synth_samples.ndim == 3:
        synth_samples = synth_samples[:, :, 0]

    assert len(synth_samples) == len(real_samples)
    assert (
        synth_samples.shape[-1] == total_length
        and real_samples.shape[-1] == total_length
    )
    num_samples = len(real_samples)

    synthetic_dataset = GluonTSNumpyDataset(synth_samples)

    linear_metrics, _, _ = linear_pred_score(
        samples=synth_samples,
        context_length=context_length,
        prediction_length=prediction_length,
        test_dataset=dataset.test,
        num_samples=100
    )

    wasserstein_metrics = wasserstein(
        x0=torch.tensor(synth_samples),
        x1=torch.tensor(real_samples)
    )
    import pdb; pdb.set_trace()
    return dict(
        linear_metrics=linear_metrics,
        wasserstein_metrics=wasserstein_metrics,
    )



    '''
    if model_name == "linear":
        logger.info(f"Running TSTR for {model_name}")
        tstr_predictor = LinearEstimator(
            freq=freq,  # Not actually used in the estimator
            prediction_length=prediction_length,
            context_length=context_length,
            num_train_samples=num_samples,
            # Synthetic dataset is in the "scaled space"
            scaling=False,
        ).train(synthetic_dataset)
    elif model_name == "deepar":
        logger.info(f"Running TSTR for {model_name}")
        tstr_predictor = DeepAREstimator(
            freq=freq,
            prediction_length=prediction_length,
            # Synthetic dataset is in the "scaled space"
            scaling=False,
            time_features=[],
            lags_seq=get_lags_for_frequency(freq, lag_ub=context_length)
        ).train(synthetic_dataset)
    elif model_name == "transformer":
        logger.info(f"Running TSTR for {model_name}")
        tstr_predictor = TransformerEstimator(
            freq=freq,
            prediction_length=prediction_length,
            # Synthetic dataset is in the "scaled space"
            scaling=False,
            time_features=[],
            lags_seq=get_lags_for_frequency(freq, lag_ub=context_length)
        ).train(synthetic_dataset)

    tstr_metrics = evaluate_tstr(
        tstr_predictor=tstr_predictor,
        test_dataset=dataset.test,
        context_length=context_length,
        prediction_length=prediction_length
    )

    return dict(
        tstr_metrics=tstr_metrics,
    )
    '''


def main(config: dict, log_dir: str, samples_path: str):
    # Read global parameters
    dataset_name = config['dataset_params']["dataset"]
    context_length = config['model_params']["context_length"]
    prediction_length = config['model_params']["prediction_length"]

    # Create log_dir
    log_dir: Path = Path(log_dir)
    base_dirname = "tstr_log"
    run_num = get_next_file_num(
        base_dirname, log_dir, file_type="", separator="-"
    )
    log_dir = log_dir / f"{base_dirname}-{run_num}"
    log_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Logging to {log_dir}")

    # Load dataset and model
    logger.info("Loading model")
    dataset = get_gts_dataset(dataset_name)
    config["freq"] = dataset.metadata.freq
    assert prediction_length == dataset.metadata.prediction_length

    model = load_model(config)

    # Setup data transformation and loading
    transformation = create_transforms(
        time_features=time_features_from_frequency_str(config["model_params"]["freq"]),
        prediction_length=prediction_length,
        freq = config["model_params"]["freq"],
        train_length = len(dataset.train),
    )
    transformed_data = transformation.apply(list(dataset.train), is_train=True)
    training_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq),
        future_length=prediction_length,
        mode="train",
    )
    train_dataloader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=1000,
        stack_fn=batchify,
        transform=training_splitter,
    )

    # Generate real samples
    logger.info("Generating real samples")
    real_samples = sample_real(
        train_dataloader,
        n_timesteps=context_length + prediction_length,
        num_samples=10000,
    )
    np.save(log_dir / "real_samples.npy", real_samples)

    if samples_path is None:
        # Generate synthetic samples
        logger.info("Generating synthetic samples")
        synth_samples = sample_synthetic(model, num_samples=10000)
        np.save(log_dir / "synth_samples.npy", synth_samples)
    else:
        logger.info(f"Using synthetic samples from {samples_path}")
        synth_samples = np.load(samples_path)[:10000]
        synth_samples = synth_samples.reshape(
            (10000, context_length + prediction_length)
        )

    # Run TSTR experiment for each downstream model
    results = []

    for model_name in DOWNSTREAM_MODELS:
        logger.info(f"Training and evaluating {model_name}")
        metrics = train_and_evaluate(
            dataset=dataset,
            model_name=model_name,
            synth_samples=synth_samples,
            real_samples=real_samples,
            config=config
        )
        results.append({"model": model_name, **metrics})

    logger.info("Saving results")
    with open(log_dir / "results.yaml", "w") as fp:
        yaml.safe_dump(
            {"config": config, "metrics": results},
            fp,
            default_flow_style=False,
            sort_keys=False,
        )


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results", help="Path to results dir"
    )
    parser.add_argument(
        "--samples_path", type=str, help="Path to generated samples"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    main(config=config, log_dir=args.out_dir, samples_path=args.samples_path)