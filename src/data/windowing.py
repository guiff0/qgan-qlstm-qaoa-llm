"""
Sliding-window sequence construction for the Classical LSTM baseline.

METHODOLOGY GAP FOUND AND FIXED HERE:

Every model in the original pipeline (and, until this file, every model
here) consumed one row = one independent sample, with LSTMs given a
length-1 "sequence" via `.unsqueeze(1)`. That's not a bug exactly (it
runs, it doesn't crash, it doesn't detach gradients) — but it quietly
defeats the entire architectural point of choosing an LSTM. An LSTM's
whole value over a plain feedforward/gated layer is that it carries
state across a real sequence of past timesteps; fed length-1 input, its
recurrent weights never see more than one step and contribute nothing
an MLP couldn't. If "Classical LSTM" is meant to be a genuine baseline
(the manuscript's config even names a sequence_length: 60 lookback
window that nothing was using), it should actually look back 60 steps.

Implemented as an indexable Dataset that constructs each window
on-the-fly rather than materializing a (5.5M, 60, 32) array up front
(~1TB+ at float32 for the full training set) -- memory-safe at the
data scale this project describes.

The QGAN-LLM and Classical GAN-LLM generators are deliberately NOT
changed to use this: their job is to map noise -> one synthetic
feature vector resembling one real timestep, which is a single-step
generative task, not a forecasting task with a lookback window. Only
the point-forecasters (Classical LSTM's own forecast, and the
forecast heads already added to the GAN baselines) are sequence
models where a real lookback window matters.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedSequenceDataset(Dataset):
    """
    Wraps a (n_rows, n_features) array + (n_rows,) target array into a
    dataset of (sequence_length, n_features) -> scalar-target samples,
    without duplicating the underlying data in memory.

    Sample i corresponds to using rows [i, i+sequence_length) as the
    LSTM input sequence and predicting the target at row
    i+sequence_length (i.e. the next step after the window — a
    standard one-step-ahead forecasting setup, and consistent with the
    chronological, no-shuffling-across-time discipline used for the
    train/val/test split itself).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 60):
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        if len(X) <= sequence_length:
            raise ValueError(
                f"Not enough rows ({len(X)}) to build even one window of "
                f"length {sequence_length}. Check your date-range split in "
                f"config/default_config.yaml -- this usually means a split "
                f"(train/val/test) ended up empty or too small."
            )
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.sequence_length = sequence_length
        self.n_samples = len(X) - sequence_length

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        seq = self.X[idx: idx + self.sequence_length]          # (seq_len, n_features)
        target = self.y[idx + self.sequence_length]             # scalar: next step after the window
        last_price = self.X[idx + self.sequence_length - 1, 0]  # last observed value in the window
        return (
            torch.from_numpy(seq),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(last_price, dtype=torch.float32),
        )


def make_windowed_last_prices(X: np.ndarray, sequence_length: int) -> np.ndarray:
    """Standalone helper for callers (e.g. the adversarial-attack step)
    that need the same 'last observed value before the prediction
    target' array the windowed dataset produces internally, without
    constructing a full Dataset."""
    return np.asarray(X, dtype=np.float32)[sequence_length - 1: -1, 0]
