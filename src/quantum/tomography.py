"""
Quantum state tomography metrics: entanglement entropy, purity, and
state fidelity.

THIS REPLACES THE ORIGINAL STUB:

    def _measure_entanglement(self):
        return np.random.uniform(2.5, 4.0)  # Placeholder

That stub returned a random number labeled as "entanglement entropy."
Any correlation the dissertation reports between this quantity and
detection accuracy (e.g. r=0.67) would have been a correlation with
noise, not with anything the model actually did — this is exactly
the kind of self-confirming result flagged earlier. Below is a real,
literature-standard calculation:

- Get the full statevector from the PennyLane circuit
  (qml.state(), not qml.expval — expval alone cannot reconstruct
  entanglement, which is a property of the joint state).
- Partition the qubits into two halves (or a caller-specified cut).
- Compute the reduced density matrix of one half via partial trace.
- Von Neumann entropy of that reduced density matrix IS the
  entanglement entropy between the two halves for a pure global state.
- Purity = Tr(rho^2), a companion metric already named in the
  dissertation's Table 49 (Quantum State Tomography Metrics).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def statevector_from_circuit(circuit_fn, params, n_qubits: int, dev_name: str = "default.qubit"):
    """
    Run `circuit_fn(params)` (a function that only applies gates, no
    measurement) on a fresh device configured to return the full
    statevector, and return that state as a numpy array of length 2**n_qubits.
    """
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev)
    def _state_qnode(p):
        circuit_fn(p)
        return qml.state()

    return np.asarray(_state_qnode(params))


def reduced_density_matrix(statevector: np.ndarray, n_qubits: int, subsystem_qubits: list[int]) -> np.ndarray:
    """
    Partial trace of |psi><psi| over the complement of `subsystem_qubits`,
    returning the reduced density matrix of the requested subsystem.
    """
    psi = statevector.reshape([2] * n_qubits)
    n_sub = len(subsystem_qubits)
    complement = [q for q in range(n_qubits) if q not in subsystem_qubits]

    # Move subsystem qubits to the front, complement to the back
    order = subsystem_qubits + complement
    psi = np.transpose(psi, axes=order)
    psi = psi.reshape(2 ** n_sub, 2 ** len(complement))

    rho = psi @ psi.conj().T  # (2^n_sub, 2^n_sub)
    return rho


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    """S(rho) = -Tr(rho log2 rho), computed via eigenvalues for numerical stability."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals.real, eps, 1.0)
    return float(-np.sum(eigvals * np.log2(eigvals)))


def purity(rho: np.ndarray) -> float:
    """Tr(rho^2). Purity = 1 for a pure state, 1/d for maximally mixed of dimension d."""
    return float(np.real(np.trace(rho @ rho)))


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Uhlmann fidelity between two density matrices. For two pure states
    reduced to the same subsystem this reduces to |<phi|psi>|^2.
    Used to compare the QGAN's synthetic-data-encoding state against
    a reference/target state if one is defined.
    """
    # sqrt(rho) via eigendecomposition (rho is Hermitian PSD)
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals.real, 0, None)
    sqrt_rho = (evecs * np.sqrt(evals)) @ evecs.conj().T

    inner = sqrt_rho @ sigma @ sqrt_rho
    evals_inner = np.linalg.eigvalsh(inner)
    evals_inner = np.clip(evals_inner.real, 0, None)
    return float(np.sum(np.sqrt(evals_inner)) ** 2)


def entanglement_metrics(circuit_fn, params, n_qubits: int,
                          dev_name: str = "default.qubit",
                          cut: int | None = None) -> dict:
    """
    Full tomography report for one circuit execution:
      - entanglement_entropy: von Neumann entropy across the bipartition
      - purity: purity of the same reduced subsystem
      - subsystem_size: how many qubits were on each side of the cut

    `cut` is the number of qubits assigned to subsystem A; defaults to
    an even bipartition (n_qubits // 2), which is the standard choice
    for reporting a single scalar entanglement-entropy value per circuit.
    """
    if cut is None:
        cut = n_qubits // 2
    subsystem_qubits = list(range(cut))

    state = statevector_from_circuit(circuit_fn, params, n_qubits, dev_name)
    rho_A = reduced_density_matrix(state, n_qubits, subsystem_qubits)

    return {
        "entanglement_entropy": von_neumann_entropy(rho_A),
        "purity": purity(rho_A),
        "subsystem_size": cut,
        "max_possible_entropy": float(cut),  # log2(2^cut) — for normalizing if needed
    }
