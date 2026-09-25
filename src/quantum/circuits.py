"""
QLSTM generator circuit.

*** CRITICAL BUG FOUND AND FIXED IN THIS FILE ***

The original QLSTMGenerator.forward() did this, per sample in a Python loop:

    inputs = z[i].detach().numpy()
    weights = self.theta.detach().numpy()
    result = self._quantum_circuit(inputs, weights)

`.detach().numpy()` strips both `z` and `self.theta` out of the PyTorch
autograd graph before the quantum circuit ever runs. That means when
`total_loss.backward()` was later called during training, no gradient
could reach `self.theta` — the circuit's own trainable parameters.
The generator's *classical* output_layer would still learn something
(it sees fixed, un-trainable quantum outputs as input), but the quantum
circuit itself would train to nothing but its random initialization.
Every result attributed to "20-qubit ring-entangled QGAN generator" vs.
"12-qubit" vs. "linear topology" would, in the original code, actually
just be measuring different random-but-frozen quantum embeddings, not
a trained effect of entanglement structure or qubit count.

THE FIX: use PennyLane's native torch interface (`interface="torch"`)
so the QNode is a differentiable PyTorch operation, and never call
.detach() or .numpy() on parameters that need gradients.
"""
from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


def build_qlstm_qnode(n_qubits: int, n_layers: int, entanglement: str,
                       dev_name: str = "default.qubit"):
    """
    Returns a torch-differentiable QNode implementing the encoding +
    variational-layer + measurement structure described in the
    dissertation's Materials/Instrumentation section.
    """
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop" if dev_name == "default.qubit" else "parameter-shift")
    def circuit(inputs, weights):
        # --- Encoding layer ---
        for i in range(min(n_qubits, inputs.shape[-1])):
            qml.RY(inputs[..., i], wires=i)
            qml.RZ(inputs[..., i] * 0.1, wires=i)

        # --- Variational layers ---
        n_params_per_layer = n_qubits * 3
        for layer in range(n_layers):
            start = layer * n_params_per_layer
            for q in range(n_qubits):
                idx = start + q * 3
                qml.RX(weights[..., idx], wires=q)
                qml.RY(weights[..., idx + 1], wires=q)
                qml.RZ(weights[..., idx + 2], wires=q)

            if entanglement == "ring":
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
            elif entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            elif entanglement == "linear":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            else:
                raise ValueError(f"Unknown entanglement topology: {entanglement}")

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def apply_circuit_gates_only(inputs_np, weights_np, n_qubits: int, n_layers: int, entanglement: str):
    """
    Gate-application-only version (no measurement) for use with
    src/quantum/tomography.py's statevector_from_circuit(), which needs
    a function that ends with the circuit still "open" so qml.state()
    can be appended by the caller. Takes plain numpy arrays since
    tomography is a post-hoc analysis step, not part of the training
    graph — detaching here is fine (unlike in the generator's forward pass).
    """
    def _apply(weights):
        for i in range(min(n_qubits, len(inputs_np))):
            qml.RY(inputs_np[i], wires=i)
            qml.RZ(inputs_np[i] * 0.1, wires=i)

        n_params_per_layer = n_qubits * 3
        for layer in range(n_layers):
            start = layer * n_params_per_layer
            for q in range(n_qubits):
                idx = start + q * 3
                qml.RX(weights[idx], wires=q)
                qml.RY(weights[idx + 1], wires=q)
                qml.RZ(weights[idx + 2], wires=q)

            if entanglement == "ring":
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
            elif entanglement == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])
            elif entanglement == "linear":
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

    return lambda weights: _apply(weights if weights is not None else weights_np)


class QLSTMGenerator(nn.Module):
    """
    Quantum LSTM-style generator. Same architectural intent as the
    original (parameterized quantum circuit -> classical projection
    layer), but now genuinely trainable end-to-end.
    """

    def __init__(self, n_qubits: int = 20, n_layers: int = 4, n_features: int = 32,
                 entanglement: str = "ring", noise_strength: float = 0.01,
                 quantum_device: str = "default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_features = n_features
        self.entanglement = entanglement
        self.noise_strength = noise_strength
        self.quantum_device = quantum_device

        self.n_params = n_layers * n_qubits * 3
        self.theta = nn.Parameter(torch.randn(self.n_params) * 0.1)

        self.qnode = build_qlstm_qnode(n_qubits, n_layers, entanglement, quantum_device)
        self.output_layer = nn.Linear(n_qubits, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, n_features). Truncated/padded to n_qubits before encoding.
        Returns: (batch, n_features)
        """
        batch_size = z.shape[0]
        inputs = z[:, : self.n_qubits]
        if inputs.shape[1] < self.n_qubits:
            pad = torch.zeros(batch_size, self.n_qubits - inputs.shape[1], device=z.device, dtype=z.dtype)
            inputs = torch.cat([inputs, pad], dim=1)

        if self.training and self.noise_strength > 0:
            # Noise-as-a-feature (RQ3): additive Gaussian noise on the encoding,
            # kept inside the autograd graph (no detach) so its regularizing
            # effect on the trained parameters is real, not cosmetic.
            inputs = inputs + torch.randn_like(inputs) * self.noise_strength

        # PennyLane's torch interface supports batched execution when the
        # QNode's non-batch dims are consistent; broadcast over batch here.
        weights = self.theta.unsqueeze(0).expand(batch_size, -1)
        raw_outputs = self.qnode(inputs, weights)          # list of n_qubits tensors, each (batch,)
        stacked = torch.stack(raw_outputs, dim=-1).float()  # (batch, n_qubits)

        return self.output_layer(stacked)
