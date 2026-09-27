"""
QAOA-based feature selector.

======================================================================
SCOPE NOTE -- why this file exists and what it deliberately does NOT do
======================================================================

QAOA circuits use one qubit per binary decision variable. A QUBO over
the FULL pre-selection feature set used elsewhere in this project (32
features after PCA, 54 before) would need 32-54 qubits with dense
pairwise two-qubit connectivity for the quadratic terms below -- not
simulable on default.qubit/lightning.qubit at reasonable wall-clock
cost, and nowhere near a usable qubit/fidelity budget on a real QPU
either.

So this module does NOT run QAOA directly over the forecasting
feature set. Instead (see QAOAFeatureSelector.fit):
  1. Candidates are pre-filtered CLASSICALLY: the top `n_candidates`
     engineered features (by |Pearson correlation| with the regression
     target) become the QAOA problem's qubits. `n_candidates` (config
     default 16) is the actual circuit width -- never the full
     feature count.
  2. QAOA solves a QUBO over just those candidates: a feature
     SELECTION problem (which of the pre-filtered candidates to
     keep), not a feature EXTRACTION/transformation problem, and not
     the regression itself.
  3. The result is a boolean mask over the FULL original feature width
     (X_train.shape[1] columns), True only at indices QAOA selected
     among the candidates -- every non-candidate column is False by
     construction and was never considered.

See src/baselines/qaoa_llm.py's module docstring for why QAOA (a
combinatorial optimizer) is used for SELECTION only here, with the
actual price regression left to a classical linear head trained
afterward on the selected columns.

======================================================================
THE QUBO -- see _build_qubo for the exact algebra
======================================================================

Three additive terms, all in {0,1} decision variables x_i (1 =
candidate i selected):
  - importance:  reward |corr(feature_i, target)| -- pulls toward
    selecting predictive features
  - budget:      quadratic penalty pushing sum(x) toward target_k
  - redundancy:  penalty on pairs of candidates whose classical
    pairwise |corr| exceeds redundancy_edge_threshold -- discourages
    selecting two near-duplicate signals
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pennylane as qml
import torch


@dataclass
class QAOAFeatureSelectorConfig:
    n_candidates: int = 16
    target_k: int = 8
    n_layers: int = 3
    budget_penalty: float = 2.0
    redundancy_penalty: float = 1.0
    redundancy_edge_threshold: float = 0.3
    quantum_device: str = "lightning.qubit"
    n_steps: int = 150
    learning_rate: float = 0.1
    seed: int = 42


@dataclass
class QAOAFeatureSelectionResult:
    selected_mask: np.ndarray       # bool, length == n_features (FULL width)
    candidate_indices: np.ndarray   # int, length == n_candidates -- which full-width columns were even considered
    qubo_objective: float           # true QUBO value (lower is better) of the final selected bitstring
    final_probs: np.ndarray         # length 2**n_candidates, optimized circuit's basis-state probabilities (diagnostic)
    loss_history: List[float]       # cost-Hamiltonian expectation value at each optimizer step


class QAOAFeatureSelector:
    def __init__(self, config: QAOAFeatureSelectorConfig):
        self.config = config

    # ---- Step 1: classical pre-filter -------------------------------
    def _select_candidates(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Top `n_candidates` full-width columns by |corr| with the
        target, vectorized (no per-column Python loop over corrcoef)."""
        n_features = X.shape[1]
        n_candidates = min(self.config.n_candidates, n_features)
        y = np.asarray(y, dtype=np.float64).flatten()

        y_c = y - y.mean()
        X_c = X - X.mean(axis=0, keepdims=True)
        denom = np.sqrt((X_c ** 2).sum(axis=0)) * np.sqrt((y_c ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (X_c.T @ y_c) / denom
        corr = np.nan_to_num(corr, nan=0.0)  # zero-variance columns -> zero correlation, not selected preferentially

        # Top-|corr| indices; stable sort keeps ties in original column order (determinism).
        order = np.argsort(-np.abs(corr), kind="stable")
        candidate_indices = order[:n_candidates]
        return candidate_indices, np.abs(corr[candidate_indices])

    # ---- Step 2: QUBO construction -----------------------------------
    def _build_qubo(self, X_candidates: np.ndarray, importance: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Builds the QUBO over the n_candidates pre-filtered features as
        an (n, n) upper-triangular-plus-diagonal matrix Q, such that for
        binary x (n,):
            QUBO(x) = sum_i Q[i,i]*x_i  +  sum_{i<j} Q[i,j]*x_i*x_j
        (Q[i,j] for i<j already carries the FULL pairwise coefficient,
        not half of it; Q is upper-triangular, lower triangle is zero.)

        Terms:
          IMPORTANCE (reward -> negative cost, since QAOA MINIMIZES):
              Q[i,i] += -importance[i]
          BUDGET (push sum(x) toward target_k):
              lambda * (sum_i x_i - target_k)^2
              = lambda*sum_i x_i + 2*lambda*sum_{i<j} x_i*x_j
                - 2*lambda*target_k*sum_i x_i + lambda*target_k^2
              (using x_i^2 = x_i for binary x_i)
              -> Q[i,i] += lambda * (1 - 2*target_k)     for every i
              -> Q[i,j] += 2*lambda                      for EVERY pair i<j
              -> const  += lambda * target_k**2
          REDUNDANCY (discourage near-duplicate candidates):
              for pairs with |corr(cand_i, cand_j)| > redundancy_edge_threshold:
              -> Q[i,j] += mu
        The constant term doesn't affect which x minimizes QUBO(x) (it's
        dropped when building the Ising cost Hamiltonian in
        _qubo_to_ising) but IS added back in when reporting
        qubo_objective, so that number is the true QUBO value, not just
        the part that mattered for optimization.
        """
        n = X_candidates.shape[1]
        lam = self.config.budget_penalty
        mu = self.config.redundancy_penalty
        k = self.config.target_k

        Q = np.zeros((n, n))
        np.fill_diagonal(Q, -importance + lam * (1 - 2 * k))

        if n > 1:
            iu = np.triu_indices(n, k=1)
            Q[iu] += 2 * lam  # budget term: every pair, unconditionally

            pairwise_corr = np.corrcoef(X_candidates, rowvar=False)
            pairwise_corr = np.nan_to_num(pairwise_corr, nan=0.0)
            redundant = np.abs(pairwise_corr) > self.config.redundancy_edge_threshold
            redundant_upper = np.triu(redundant, k=1)
            Q[redundant_upper] += mu

        const = lam * (k ** 2)
        return Q, const

    @staticmethod
    def _qubo_value(Q: np.ndarray, const: float, x: np.ndarray) -> float:
        x = x.astype(float)
        return float(const + x @ np.diag(Q) + x @ np.triu(Q, k=1) @ x)

    # ---- Step 3: QUBO -> Ising, for the QAOA cost Hamiltonian ----------
    @staticmethod
    def _qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Substitute x_i = (1 - z_i)/2 (z_i in {+1,-1}: PauliZ eigenvalues,
        |0> -> +1 i.e. x_i=0, |1> -> -1 i.e. x_i=1 -- standard QAOA
        convention) into QUBO(x) and collect terms in z:
            h[i]    -- coefficient of Z_i
            J[i,j]  -- coefficient of Z_i Z_j, i<j (upper-triangular, 0 elsewhere)
        The dropped additive constant only shifts every eigenvalue by
        the same amount and is irrelevant to which state QAOA finds --
        _qubo_value (on the QUBO, not the Ising form) is what's used to
        report the true objective.
        """
        n = Q.shape[0]
        h = -np.diag(Q).copy() / 2.0
        J = np.zeros((n, n))
        iu = np.triu_indices(n, k=1)
        qij = Q[iu]
        nz = qij != 0.0
        rows, cols = iu[0][nz], iu[1][nz]
        vals = qij[nz]
        J[rows, cols] = vals / 4.0
        np.add.at(h, rows, -vals / 4.0)
        np.add.at(h, cols, -vals / 4.0)
        return h, J

    # ---- Step 4: QAOA circuit -------------------------------------------
    def _diff_method(self) -> str:
        # Same rationale as src/quantum/circuits.py's build_qlstm_qnode:
        # backprop through the state-vector simulator when using
        # default.qubit, adjoint (exact, one extra pass) on lightning.*,
        # parameter-shift as the generic fallback for anything else
        # (e.g. a real QPU backend, or a lightning device without
        # adjoint support).
        device = self.config.quantum_device
        if device == "default.qubit":
            return "backprop"
        if device.startswith("lightning"):
            return "adjoint"
        return "parameter-shift"

    def _build_qaoa_qnodes(self, h: np.ndarray, J: np.ndarray):
        n = len(h)
        dev = qml.device(self.config.quantum_device, wires=n)
        edges = list(zip(*np.nonzero(J)))  # only (i, j) with i<j and J[i,j] != 0
        diff_method = self._diff_method()

        def _ansatz(gammas, betas):
            for w in range(n):
                qml.Hadamard(wires=w)
            for layer in range(self.config.n_layers):
                gamma, beta = gammas[layer], betas[layer]
                for w in range(n):
                    if h[w] != 0.0:
                        qml.RZ(2.0 * gamma * h[w], wires=w)
                for (i, j) in edges:
                    qml.MultiRZ(2.0 * gamma * J[i, j], wires=[int(i), int(j)])
                for w in range(n):
                    qml.RX(2.0 * beta, wires=w)

        obs_terms = [qml.PauliZ(w) for w in range(n) if h[w] != 0.0]
        obs_terms += [qml.PauliZ(int(i)) @ qml.PauliZ(int(j)) for (i, j) in edges]
        coeffs = [h[w] for w in range(n) if h[w] != 0.0] + [J[i, j] for (i, j) in edges]

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def cost_qnode(gammas, betas):
            _ansatz(gammas, betas)
            if not obs_terms:
                # Degenerate QUBO (h and J both all-zero, e.g. target_k
                # pushed to n with budget_penalty=0) -- nothing to
                # optimize; return a constant so the training loop below
                # still runs (and simply leaves gammas/betas at init).
                return qml.expval(qml.Identity(0))
            return qml.expval(qml.Hamiltonian(coeffs, obs_terms))

        # probs_qnode is only ever called under torch.no_grad() (see fit()),
        # to read out the optimized circuit's basis-state distribution --
        # never backpropagated through. adjoint differentiation doesn't
        # support qml.probs as a measurement, so this qnode always uses
        # PennyLane's own "best" choice rather than the diff_method picked
        # for the (gradient-bearing) cost_qnode above.
        @qml.qnode(dev, interface="torch", diff_method="best")
        def probs_qnode(gammas, betas):
            _ansatz(gammas, betas)
            return qml.probs(wires=range(n))

        return cost_qnode, probs_qnode

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> QAOAFeatureSelectionResult:
        cfg = self.config
        X_train = np.asarray(X_train)
        candidate_indices, importance = self._select_candidates(X_train, y_train)
        X_candidates = X_train[:, candidate_indices]
        n = len(candidate_indices)

        Q, const = self._build_qubo(X_candidates, importance)
        h, J = self._qubo_to_ising(Q)
        cost_qnode, probs_qnode = self._build_qaoa_qnodes(h, J)

        torch.manual_seed(cfg.seed)
        gammas = torch.nn.Parameter(torch.rand(cfg.n_layers, dtype=torch.float64) * 0.1)
        betas = torch.nn.Parameter(torch.rand(cfg.n_layers, dtype=torch.float64) * 0.1)
        optimizer = torch.optim.Adam([gammas, betas], lr=cfg.learning_rate)

        loss_history: List[float] = []
        for _ in range(cfg.n_steps):
            optimizer.zero_grad()
            loss = cost_qnode(gammas, betas)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach()))

        with torch.no_grad():
            probs = probs_qnode(gammas, betas).detach().numpy()
        best_state = int(np.argmax(probs))
        # PennyLane's qml.probs bitstring convention: wire 0 is the
        # most-significant bit of the returned index.
        x_candidates = np.array([(best_state >> (n - 1 - w)) & 1 for w in range(n)], dtype=bool)

        qubo_objective = self._qubo_value(Q, const, x_candidates)

        full_mask = np.zeros(X_train.shape[1], dtype=bool)
        full_mask[candidate_indices[x_candidates]] = True

        return QAOAFeatureSelectionResult(
            selected_mask=full_mask,
            candidate_indices=candidate_indices,
            qubo_objective=qubo_objective,
            final_probs=probs,
            loss_history=loss_history,
        )
