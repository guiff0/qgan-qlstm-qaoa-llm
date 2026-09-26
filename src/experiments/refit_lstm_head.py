"""
Recover a usable Classical LSTM from a LEGACY checkpoint (LSTM weights only).

WHY THIS EXISTS
---------------
Before the fix in classical_lstm.py, save_model() stored only the nn.LSTM
weights, never the output head `fc`. So models/classical_lstm_best.pt from
an earlier run holds the best-epoch LSTM but NOT the head that was trained
with it, and that head is not recoverable.

The LSTM (the expensive part -- hours of training) is intact. The head is a
single Linear(hidden_size -> 1) on the LSTM's last hidden state, so with the
LSTM frozen its optimal value has a closed form: ordinary least squares.
This script computes it in one forward-only pass over the training split
(no backprop, no epochs) by accumulating the normal equations, then writes
a complete checkpoint in the new format.

NOTE: this gives the *least-squares-optimal* head for the frozen epoch-N
LSTM, not the exact head that existed at epoch N. Validation error should
be very close to what was logged, but not bit-identical. It is fit on the
TRAIN split only; val/test are never touched for fitting.

Run:  python -m src.experiments.refit_lstm_head
      python -m src.experiments.refit_lstm_head --stride 4    # faster, ~4x fewer windows
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.classical_lstm import ClassicalLSTM, CHECKPOINT_PATH
from src.experiments.run_all import load_processed_split
from src.utils.config import load_config
from src.utils.reproducibility import set_all_seeds
from src.utils.progress import progress_bar


def _windows_and_targets(model: ClassicalLSTM, X, y, idxs):
    view = model._window_view(X)
    seq_len = model.config["sequence_length"]
    xb = torch.from_numpy(np.ascontiguousarray(view[idxs]))
    yb = np.asarray(y, dtype=np.float64)[idxs + seq_len]
    return xb, yb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth training window")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--ridge", type=float, default=1e-8, help="tiny relative ridge for numerical stability")
    ap.add_argument("--force", action="store_true", help="refit even if the checkpoint already has a head")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"]
    set_all_seeds(seed)
    processed_dir = cfg["data"]["processed_dir"]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    is_new_format = isinstance(ckpt, dict) and "lstm" in ckpt and "fc" in ckpt
    if is_new_format and not args.force:
        print(f"{args.checkpoint} already contains an LSTM and a head; nothing to do (use --force to refit).")
        return
    lstm_state = ckpt["lstm"] if is_new_format else ckpt

    model = ClassicalLSTM(cfg["classical_lstm"], seed=seed)
    model.build()
    model.model.load_state_dict(lstm_state)
    model.model.eval()
    H = model.config["hidden_size"]
    seq_len = model.config["sequence_length"]

    X_train, y_train = load_processed_split(processed_dir, "train")
    X_val, y_val = load_processed_split(processed_dir, "val")

    # ---- accumulate normal equations for [h, 1] @ w = y  (float64) ----
    n_windows = len(X_train) - seq_len
    all_idx = np.arange(0, n_windows, max(args.stride, 1))
    A = np.zeros((H + 1, H + 1), dtype=np.float64)
    b = np.zeros(H + 1, dtype=np.float64)
    n_chunks = max(1, -(-len(all_idx) // args.batch_size))
    fit_bar = progress_bar(total=n_chunks, desc="Refit: accumulating normal equations", unit="chunk")
    with torch.no_grad():
        for k in range(0, len(all_idx), args.batch_size):
            idxs = all_idx[k: k + args.batch_size]
            xb, yb = _windows_and_targets(model, X_train, y_train, idxs)
            h = model._features(xb).numpy().astype(np.float64)
            hb = np.concatenate([h, np.ones((len(h), 1))], axis=1)
            A += hb.T @ hb
            b += hb.T @ yb
            fit_bar.update(1)
            fit_bar.set_postfix(windows=f"{k + len(idxs):,}/{len(all_idx):,}")
    fit_bar.close()

    reg = args.ridge * np.trace(A) / A.shape[0]
    w = np.linalg.solve(A + reg * np.eye(H + 1), b)
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor(w[:H], dtype=torch.float32).unsqueeze(0))
        model.fc.bias.copy_(torch.tensor(w[H:], dtype=torch.float32))

    # ---- report (val is for reporting only, never used to fit) ----
    def mse(X, y, stride, desc):
        view_n = len(X) - seq_len
        idxs_all = np.arange(0, view_n, stride)
        n_chunks_mse = max(1, -(-len(idxs_all) // args.batch_size))
        se, n = 0.0, 0
        with torch.no_grad():
            for k in progress_bar(range(0, len(idxs_all), args.batch_size), total=n_chunks_mse,
                                  desc=desc, unit="chunk"):
                idxs = idxs_all[k: k + args.batch_size]
                xb, yb = _windows_and_targets(model, X, y, idxs)
                p = model._forward(xb).squeeze(-1).numpy().astype(np.float64)
                se += float(((p - yb) ** 2).sum())
                n += len(idxs)
        return se / n

    print(f"Train MSE (refit head): {mse(X_train, y_train, max(args.stride, 1), 'Refit: scoring train'):.3e}")
    print(f"Val   MSE (refit head): {mse(X_val, y_val, 1, 'Refit: scoring val'):.3e}"
          f"   <- compare to the val_loss logged at your best epoch")

    if not is_new_format:
        backup = args.checkpoint.replace(".pt", ".legacy.pt")
        if not os.path.exists(backup):
            shutil.copy2(args.checkpoint, backup)
            print(f"Backed up original checkpoint -> {backup}")
    model.save_model(args.checkpoint)
    print(f"Wrote complete checkpoint (LSTM + head) -> {args.checkpoint}")
    print('Next: python -m src.experiments.run_all --only "Classical LSTM" --reuse-checkpoints')


if __name__ == "__main__":
    main()
