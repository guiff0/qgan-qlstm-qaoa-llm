"""
Master experiment runner.

Executes every baseline + ablation in the config, with:
  - per-model seeding (not just once at the top of the whole run)
  - real train/val/test data (loaded, not assumed to already exist as .npy)
  - structured logging (.log + .json + a row in results/all_results.csv) per run
  - real adversarial-attack and entanglement-tomography evaluation

Run with:  python -m src.experiments.run_all
See RUNNING.md for the full step-by-step, including expected runtime.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.classical_lstm import ClassicalLSTM
from src.baselines.classical_gan_llm import ClassicalGANLLM
from src.baselines.qgan_llm import QGANLLM
from src.baselines.qlstm_forecaster import QLSTMForecaster
from src.utils.config import load_config, merge_override
from src.utils.logging_utils import RunLogger, make_run_id
from src.utils.reproducibility import set_all_seeds


def load_processed_split(processed_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    X_path = os.path.join(processed_dir, f"X_{split}.npy")
    y_path = os.path.join(processed_dir, f"y_{split}.npy")
    if not (os.path.isfile(X_path) and os.path.isfile(y_path)):
        raise FileNotFoundError(
            f"\n\nProcessed data not found at {X_path} / {y_path}.\n"
            f"Run `python -m scripts.prepare_data` first (see RUNNING.md, Step 2)."
        )
    return np.load(X_path), np.load(y_path)


def run_one_model(model, model_name: str, splits: dict, cfg: dict, seed: int):
    set_all_seeds(seed)
    run_id = make_run_id(model_name)
    log_cfg = cfg.get("logging", {})
    run_logger = RunLogger(
        run_id,
        log_dir=log_cfg.get("log_dir", "logs"),
        results_dir=log_cfg.get("results_dir", "results"),
    )
    run_logger.log_config(model.config)
    run_logger.info(f"Starting training run for {model_name} (run_id={run_id})")

    model.train(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"], run_logger=run_logger)

    attack_cfg = cfg["adversarial"]
    # last_input_prices: last observed close price per test sequence — needed
    # by the attack success metric's directional-flip definition. Assumes the
    # first feature column is (or is derived from) the last observed close;
    # adjust the index below if your feature ordering differs.
    last_prices = splits["X_test"][:, 0]

    metrics = model.evaluate(
        splits["X_test"], splits["y_test"],
        attack_cfg=attack_cfg, last_input_prices=last_prices,
    )

    if hasattr(model, "measure_latency"):
        latency_report = model.measure_latency(splits["X_test"])
        metrics["latency_mean_ms"] = latency_report["mean_latency_ms"]
        metrics["latency_sd_ms"] = latency_report["sd_latency_ms"]
        metrics["latency_raw_samples_ms"] = latency_report["raw_samples_ms"]
        run_logger.info(
            f"Latency: mean={latency_report['mean_latency_ms']:.3f}ms "
            f"SD={latency_report['sd_latency_ms']:.3f}ms "
            f"(n={latency_report['n_repeats']} timed single-sample forward passes)"
        )

    run_logger.log_final_metrics(metrics)
    run_logger.append_to_results_csv(
        model_name,
        {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
    )
    # The full per-sample latency distribution (not just mean/SD) is kept in
    # the structured .json log via save_json() below, not the CSV -- this is
    # exactly what's needed to check a claim like "97% of runs fall below
    # 40ms" against the real distribution rather than assuming normality
    # from mean/SD alone.
    run_logger.save_json()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config YAML (default: config/default_config.yaml)")
    parser.add_argument("--only", default=None,
                         help="Comma-separated subset of model names to run (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    processed_dir = cfg["data"]["processed_dir"]

    splits = {}
    for split in ("train", "val", "test"):
        X, y = load_processed_split(processed_dir, split)
        splits[f"X_{split}"] = X
        splits[f"y_{split}"] = y

    models_to_run = {
        "Classical LSTM": lambda: ClassicalLSTM(cfg["classical_lstm"], seed=seed),
        "Classical GAN-LLM": lambda: ClassicalGANLLM(cfg["classical_gan_llm"], seed=seed),
        "QGAN-LLM": lambda: QGANLLM(cfg["qgan_llm"], seed=seed),
        "QLSTM Forecaster": lambda: QLSTMForecaster(cfg["qlstm_forecaster"], seed=seed),
    }
    for ablation in cfg["ablations"]:
        ablation_cfg = merge_override(cfg["qgan_llm"], {k: v for k, v in ablation.items() if k != "name"})
        models_to_run[ablation["name"]] = (lambda c=ablation_cfg: QGANLLM(c, seed=seed))

    if args.only:
        selected = set(name.strip() for name in args.only.split(","))
        models_to_run = {k: v for k, v in models_to_run.items() if k in selected}

    all_metrics = {}
    for name, factory in models_to_run.items():
        print("=" * 70)
        print(f"Running: {name}")
        print("=" * 70)
        model = factory()
        metrics = run_one_model(model, name, splits, cfg, seed)
        all_metrics[name] = metrics

    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE — summary:")
    print("=" * 70)
    for name, metrics in all_metrics.items():
        printable = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        print(f"{name}: {printable}")


if __name__ == "__main__":
    main()
