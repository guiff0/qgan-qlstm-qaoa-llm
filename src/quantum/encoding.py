"""
Amplitude-encoding reference states.

FLAGGED ARCHITECTURE GAP: the six confirmed answers (from an earlier
planning exchange) assumed amplitude encoding is this study's PRIMARY
encoding scheme, with angle encoding as a comparison ablation. That is
not what's actually built: circuits.py's QLSTMGenerator and
QLSTMForecaster both encode features via angle encoding (RY+RZ
rotations) -- there is no trainable amplitude-encoding circuit
anywhere in this codebase. This module does NOT silently build a fake
"amplitude encoding primary path" to match that premise. What it does
provide is real and useful regardless: a genuine, deterministic
amplitude-encoding REFERENCE STATE (the ideal target), which
encoding_fidelity.py uses to ask a well-posed, honest question: how
closely does the circuit actually implemented here (angle encoding)
approximate ideal amplitude encoding of the same input? That is a
different, narrower claim than "we compared amplitude encoding to angle
encoding as two implemented alternatives" -- state it that way in the
paper, or build a real trainable amplitude-encoding circuit first if the
broader claim is what's needed.
"""
from __future__ import annotations

import numpy as np


def amplitude_encode(X: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Deterministic amplitude-encoding reference: pads/truncates each row
    of X to length 2^n_qubits and unit-normalizes it. This is the
    "ideal" target state -- what a perfect amplitude encoder would
    prepare -- computed directly from the classical data with no
    quantum circuit involved (this part has no approximation error by
    construction; the approximation error, if any, comes from whatever
    circuit is asked to reproduce it).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    target_dim = 2 ** n_qubits
    n_features = X.shape[1]

    if n_features < target_dim:
        X = np.pad(X, ((0, 0), (0, target_dim - n_features)))
    elif n_features > target_dim:
        X = X[:, :target_dim]

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms
