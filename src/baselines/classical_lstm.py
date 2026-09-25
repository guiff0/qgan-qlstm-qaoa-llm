"""
Standalone Classical LSTM baseline.

TWO ISSUES FIXED IN THIS FILE:

1. The original train() did:
       lstm_out, _ = self.model(X_train)     # X_train = ENTIRE training set
   passing all ~5.5 million rows through the LSTM in one forward pass per
   "epoch" -- infeasible at this scale and not a meaningful training loop
   (one gradient update per epoch instead of thousands). Fixed with a
   DataLoader-based mini-batch loop.

2. METHODOLOGY GAP: every row was treated as an independent, length-1
   "sequence" (X.unsqueeze(1)), which never lets the LSTM's recurrence
   see more than one timestep -- it contributes nothing over a plain
   gated feedforward layer. Fixed by using a genuine sliding window
   (src/data/windowing.py), matching the sequence_length: 60 lookback
   already named (but previously unused) in config/default_config.yaml.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseForecastingModel
from ..data.windowing import WindowedSequenceDataset
from ..evaluation.metrics import rmse as rmse_fn, mae as mae_fn
from ..evaluation.latency import measure_inference_latency
from ..utils.reproducibility import set_all_seeds, seeded_generator


class ClassicalLSTM(BaseForecastingModel):
    def __init__(self, config: Dict = None, seed: int = 42):
        default_config = {
            "input_size": 32,
            "hidden_size": 128,
            "num_layers": 2,
            "output_size": 1,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50,
            "early_stopping_patience": 10,
            "sequence_length": 60,
        }
        cfg = {**default_config, **(config or {})}
        super().__init__("Classical LSTM", cfg)
        self.seed = seed

    def build(self):
        set_all_seeds(self.seed)
        self.model = nn.LSTM(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"] if self.config["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(self.config["hidden_size"], self.config["output_size"])
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.fc.parameters()),
            lr=self.config["learning_rate"],
        )
        self.criterion = nn.MSELoss()

    def _forward(self, X_seq: torch.Tensor) -> torch.Tensor:
        """X_seq: (batch, sequence_length, n_features) -- a REAL sequence,
        not a length-1 stand-in."""
        lstm_out, _ = self.model(X_seq)
        return self.fc(lstm_out[:, -1, :])

    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        self.build()
        seq_len = self.config["sequence_length"]

        train_ds = WindowedSequenceDataset(X_train, y_train, sequence_length=seq_len)
        train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"], shuffle=True,
            generator=seeded_generator(self.seed),
        )
        val_ds = WindowedSequenceDataset(X_val, y_val, sequence_length=seq_len)
        val_loader = DataLoader(val_ds, batch_size=self.config["batch_size"], shuffle=False)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for X_batch, y_batch, _last_price in train_loader:
                self.optimizer.zero_grad()
                pred = self._forward(X_batch)
                loss = self.criterion(pred.squeeze(-1), y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            train_loss = epoch_loss / max(n_batches, 1)

            self.model.eval()
            val_loss_total, n_val_batches = 0.0, 0
            with torch.no_grad():
                for X_batch, y_batch, _last_price in val_loader:
                    val_pred = self._forward(X_batch)
                    val_loss_total += self.criterion(val_pred.squeeze(-1), y_batch).item()
                    n_val_batches += 1
            val_loss = val_loss_total / max(n_val_batches, 1)

            if run_logger:
                run_logger.log_epoch(epoch, train_loss=train_loss, val_loss=val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model("models/classical_lstm_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config["early_stopping_patience"]:
                    if run_logger:
                        run_logger.info(f"Early stopping at epoch {epoch}")
                    break

        self.is_trained = True
        self.load_model("models/classical_lstm_best.pt")

    def _window_inputs_only(self, X: np.ndarray) -> torch.Tensor:
        """Build (n_windows, seq_len, n_features) from a flat (n_rows,
        n_features) array, for predict()/evaluate() callers that only
        have X (no y) -- e.g. inference-time use."""
        seq_len = self.config["sequence_length"]
        X = np.asarray(X, dtype=np.float32)
        n_windows = len(X) - seq_len
        if n_windows <= 0:
            raise ValueError(
                f"Need more than {seq_len} rows to build a windowed sequence; got {len(X)}."
            )
        windows = np.stack([X[i: i + seq_len] for i in range(n_windows)])
        return torch.from_numpy(windows)

    def predict(self, X):
        self.model.eval()
        X_seq = self._window_inputs_only(X) if not torch.is_tensor(X) or X.dim() == 2 else X
        with torch.no_grad():
            pred = self._forward(X_seq)
        return pred.numpy()

    def measure_latency(self, X_test, n_repeats: int = 100) -> dict:
        """Timed on one full window (sequence_length rows in, one
        prediction out) -- the unit of a single 'input to output' pass
        for a sequence model, matching Ch.3's operational definition of
        DV4 (Response Latency). See ClassicalGANLLM.measure_latency for
        why this exists at all."""
        self.model.eval()
        X_windowed = self._window_inputs_only(X_test).numpy()  # (n_windows, seq_len, n_features)
        return measure_inference_latency(self._forward, X_windowed, n_repeats=n_repeats)

    def evaluate(self, X_test, y_test, **kwargs):
        """NOTE: predictions correspond to rows [sequence_length, len(X_test))
        of the input -- the first `sequence_length` rows have no full window
        and are dropped, so y_test is trimmed to match before scoring."""
        seq_len = self.config["sequence_length"]
        predictions = self.predict(X_test)
        y_test_arr = np.array(y_test).flatten()[seq_len:]
        predictions_arr = np.array(predictions).flatten()

        self.results = {
            "rmse": rmse_fn(y_test_arr, predictions_arr),
            "mae": mae_fn(y_test_arr, predictions_arr),
            "model_type": "Classical LSTM",
        }
        return self.results
