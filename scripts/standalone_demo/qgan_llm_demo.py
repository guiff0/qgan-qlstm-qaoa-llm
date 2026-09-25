#!/usr/bin/env python3
"""
================================================================================
STANDALONE QGAN DEMO -- fully synthetic, NOT the dissertation pipeline
================================================================================
See README.md in this folder before using this file for anything. This is a
small, readable worked example of a quantum-generator / classical-
discriminator GAN with an adversarial-robustness check and a quantum-
tomography metric, run on made-up sine-wave-plus-noise data. It does not use
config/default_config.yaml, the real EUR/USD pipeline, or src/baselines/, and
its numbers are not Chapter 4 results.

Fixes applied relative to the version this is based on (see README.md,
"What was fixed", for the reasoning and verification behind each):
  1. SyntaxError in generate_synthetic_forex_data's signature -- fixed.
  2. compute_ancova_pvalue() no longer clamps to a manufactured "significant"
     range; it runs a real one-way ANCOVA (statsmodels: group + covariate,
     anova_lm) over synthetic per-sample errors.
  3. fid_proxy replaced with an actual Frechet distance (mean + covariance),
     not a bare squared difference of means.
  4. ASR is explicitly labeled with its definition (abs-error-above-
     threshold) so it is never confused with the real pipeline's directional-
     flip ASR.
  5. No invented hardware-timing numbers anywhere in this file or its README.
================================================================================
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import pennylane as qml
except ImportError:
    print("[ERROR] PennyLane is required. Install via 'pip install pennylane'")
    sys.exit(1)

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# ==============================================================================
# 1. STRUCTURED JSON EXECUTION LOGGER
# ==============================================================================
class DemoLogger:
    """Minimal JSON-lines step logger, local to this standalone demo.

    For the real pipeline's equivalent, see src/utils/step_tracer.py
    (StepTracer) -- this class is intentionally NOT that one, to keep this
    demo runnable with zero dependency on the rest of the repo.
    """

    def __init__(self, log_dir: str = "logs/standalone_demo"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"demo_{int(time.time())}.jsonl")

    def log_step(self, phase: str, step_id: int, execution_time_ms: float,
                metrics: Dict[str, Any], message: str, severity: str = "INFO") -> Dict[str, Any]:
        entry = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phase": phase,
            "step_id": step_id,
            "execution_time_ms": round(execution_time_ms, 3),
            "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
            "severity": severity,
            "message": message,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        print(f"[{phase}] step {step_id} | {execution_time_ms:.1f}ms | {message} | {entry['metrics']}")
        return entry


# ==============================================================================
# 2. SYNTHETIC DATA GENERATOR
# ==============================================================================
class SyntheticSeriesGenerator:
    """Generates a made-up multi-regime series and volatility-windowed
    samples. This is explicitly NOT financial data -- no claim is made that
    it resembles EUR/USD; it exists only to give the demo something to
    train and test on."""

    def __init__(self, sequence_length: int = 20, num_samples: int = 1000, seed: int = 42):
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X, regime_labels): X is (n_windows, sequence_length)
        normalized windows; regime_labels in {0, 1, 2} by window std-dev."""
        t = np.linspace(0, 100, self.num_samples + self.sequence_length)
        base = 1.1000 + 0.05 * np.sin(t / 5) + 0.02 * np.cos(t / 2)

        volatility = np.full_like(t, 0.001)
        volatility[1000:2000] = 0.004 if len(t) > 2000 else volatility[1000:2000]
        volatility[3500:4200] = 0.008 if len(t) > 4200 else volatility[3500:4200]

        series = base + self.rng.normal(0.0, volatility)

        X, regimes = [], []
        for i in range(len(series) - self.sequence_length):
            window = series[i: i + self.sequence_length]
            std = float(np.std(window))
            norm = (window - np.mean(window)) / (std + 1e-8)
            X.append(norm)
            regimes.append(0 if std < 0.002 else 1 if std < 0.005 else 2)

        return np.asarray(X, dtype=np.float32), np.asarray(regimes, dtype=np.int32)


# ==============================================================================
# 3. QUANTUM GENERATOR CIRCUIT
# ==============================================================================
class QuantumGeneratorCircuit:
    """Small parameterized quantum circuit generator (ring-topology
    entanglement), with optional depolarizing-style phase noise."""

    def __init__(self, n_qubits: int = 8, circuit_depth: int = 3, noise_rate: float = 0.01,
                device_name: str = "default.qubit", seed: Optional[int] = None):
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.noise_rate = noise_rate
        self.rng = np.random.default_rng(seed)
        self.dev = qml.device(device_name, wires=self.n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _circuit(self, weights: torch.Tensor, latent_noise: torch.Tensor):
        for i in range(self.n_qubits):
            qml.RY(latent_noise[i], wires=i)

        for layer in range(self.circuit_depth):
            for i in range(self.n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RY(weights[layer, i, 1], wires=i)
                qml.RZ(weights[layer, i, 2], wires=i)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            if self.noise_rate > 0.0:
                for i in range(self.n_qubits):
                    # Deterministic given self.rng (seeded), unlike the
                    # original's bare np.random.randn() -- reproducible runs.
                    qml.RZ(self.noise_rate * np.pi * float(self.rng.standard_normal()), wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def generate(self, weights: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        outputs = []
        for _ in range(batch_size):
            latent = torch.randn(self.n_qubits) * math.pi
            outputs.append(torch.stack(self.qnode(weights, latent)))
        # default.qubit's torch interface returns float64 regardless of input
        # dtype; cast back to float32 so this composes with float32 modules
        # (the discriminator, real_data_batch) without a dtype mismatch.
        return torch.stack(outputs).to(torch.float32)


# ==============================================================================
# 4. CLASSICAL DISCRIMINATOR
# ==============================================================================
class ClassicalDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ==============================================================================
# 5. FRECHET DISTANCE (real FID-style metric, not a squared mean difference)
# ==============================================================================
def frechet_distance(real: torch.Tensor, fake: torch.Tensor, eps: float = 1e-6) -> float:
    """Frechet distance between two multivariate Gaussians fit to `real`
    and `fake` (each (n_samples, n_features)):

        d^2 = ||mu_r - mu_f||^2 + Tr(C_r + C_f - 2*sqrt(C_r @ C_f))

    This is what FID actually computes (ordinarily on Inception features;
    here directly on the generator/discriminator's feature space, since
    this demo has no image-classifier feature extractor). A plain squared
    difference of means -- what the version this is based on computed --
    ignores the shape/spread of each distribution entirely: two batches
    with identical means but wildly different variances would score a
    "perfect" 0 under that version, and would not here.
    """
    r = real.detach().cpu().numpy().astype(np.float64)
    f = fake.detach().cpu().numpy().astype(np.float64)
    mu_r, mu_f = r.mean(axis=0), f.mean(axis=0)
    cov_r = np.cov(r, rowvar=False) + eps * np.eye(r.shape[1])
    cov_f = np.cov(f, rowvar=False) + eps * np.eye(f.shape[1])

    diff = mu_r - mu_f
    # Matrix square root of the product via eigendecomposition of the
    # symmetrized product, which stays numerically stable for the small,
    # well-conditioned covariances this demo produces.
    prod = cov_r @ cov_f
    eigvals, eigvecs = np.linalg.eig(prod)
    sqrt_prod = (eigvecs @ np.diag(np.sqrt(eigvals.astype(np.complex128))) @ np.linalg.inv(eigvecs)).real

    return float(diff @ diff + np.trace(cov_r + cov_f - 2 * sqrt_prod))


# ==============================================================================
# 6. TRAINING HARNESS
# ==============================================================================
class QGANTrainer:
    def __init__(self, q_generator: QuantumGeneratorCircuit, discriminator: ClassicalDiscriminator,
                logger: DemoLogger, lr_g: float = 0.01, lr_d: float = 0.001, seed: int = 42):
        self.q_gen = q_generator
        self.disc = discriminator
        self.logger = logger
        torch.manual_seed(seed)
        self.weights_g = torch.randn(
            (q_generator.circuit_depth, q_generator.n_qubits, 3), requires_grad=True
        )
        self.opt_g = optim.Adam([self.weights_g], lr=lr_g)
        self.opt_d = optim.Adam(self.disc.parameters(), lr=lr_d)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, real_data_batch: torch.Tensor, epoch: int) -> Dict[str, float]:
        t0 = time.time()
        batch_size = real_data_batch.size(0)

        self.opt_d.zero_grad()
        real_labels = torch.ones((batch_size, 1))
        loss_d_real = self.criterion(self.disc(real_data_batch), real_labels)
        fake_data = self.q_gen.generate(self.weights_g, batch_size=batch_size)
        fake_labels = torch.zeros((batch_size, 1))
        loss_d_fake = self.criterion(self.disc(fake_data.detach()), fake_labels)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        self.opt_d.step()

        self.opt_g.zero_grad()
        loss_g = self.criterion(self.disc(fake_data), real_labels)
        loss_g.backward()
        self.opt_g.step()

        fid = frechet_distance(real_data_batch, fake_data)
        metrics = {
            "loss_g": float(loss_g.item()),
            "loss_d": float(loss_d.item()),
            "frechet_distance": fid,
            "weight_norm": float(torch.norm(self.weights_g).item()),
        }
        self.logger.log_step("QGAN_TRAIN", epoch, (time.time() - t0) * 1000, metrics,
                             f"Epoch {epoch}: loss_g={loss_g.item():.4f} loss_d={loss_d.item():.4f} fid={fid:.4f}")
        return metrics


# ==============================================================================
# 7. ADVERSARIAL ATTACK CHECK (FGSM)
# ==============================================================================
class AdversarialAttackEngine:
    """FGSM check with an explicit, labeled ASR definition. NOTE: this is a
    *different* definition from src/attacks/adversarial.py's directional-
    flip ASR in the real pipeline -- the two numbers are not comparable and
    this class's output says so via `asr_definition`."""

    ASR_DEFINITION = "abs_error_above_threshold"

    def __init__(self, epsilon: float = 0.10):
        self.epsilon = epsilon

    def fgsm_attack(self, model: nn.Module, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        data_copy = data.clone().detach().requires_grad_(True)
        loss = nn.MSELoss()(model(data_copy), target)
        loss.backward()
        perturbed = data + self.epsilon * data_copy.grad.data.sign()
        return torch.clamp(perturbed, -3.0, 3.0)

    def evaluate_asr(self, forecaster: nn.Module, clean_data: torch.Tensor, targets: torch.Tensor,
                     threshold: float = 0.15) -> Dict[str, float]:
        with torch.no_grad():
            clean_preds = forecaster(clean_data)
        adv_data = self.fgsm_attack(forecaster, clean_data, targets)
        with torch.no_grad():
            adv_preds = forecaster(adv_data)
        success = (torch.abs(adv_preds - targets) > threshold).float()
        return {
            "asr_definition": self.ASR_DEFINITION,
            "asr_percent": float(torch.mean(success).item()) * 100.0,
            "clean_rmse": float(torch.sqrt(torch.mean((clean_preds - targets) ** 2)).item()),
            "adv_rmse": float(torch.sqrt(torch.mean((adv_preds - targets) ** 2)).item()),
            "epsilon": self.epsilon,
            "threshold": threshold,
        }


# ==============================================================================
# 8. QUANTUM TOMOGRAPHY + REAL ANCOVA
# ==============================================================================
class QuantumTomographyEngine:
    @staticmethod
    def calculate_entanglement_entropy(state_vector: np.ndarray) -> float:
        state = state_vector / (np.linalg.norm(state_vector) + 1e-12)
        probs = np.abs(state) ** 2
        probs = probs[probs > 1e-12]
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def run_ancova(group_errors: Dict[str, np.ndarray], covariate: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Real one-way ANCOVA: error ~ C(group) + covariate, via statsmodels
        OLS + anova_lm, returning the group effect's actual F-statistic and
        p-value.

        group_errors: {"classical": array of per-sample errors, "quantum": ...}
        covariate:    matching {"classical": array, "quantum": array}, e.g. a
                      per-sample volatility proxy, used to control for it.

        Unlike the function this replaces, there is no floor or ceiling
        applied to the result: a genuinely non-significant comparison will
        report p > 0.05, and the direction of the effect is not assumed.
        """
        if not _HAS_STATSMODELS:
            raise ImportError("statsmodels is required for a real ANCOVA; pip install statsmodels")

        rows = []
        for group, errs in group_errors.items():
            cov = covariate[group]
            if len(cov) != len(errs):
                raise ValueError(f"group '{group}': {len(errs)} errors but {len(cov)} covariate values")
            rows.extend({"error": float(e), "covariate": float(c), "group": group}
                       for e, c in zip(errs, cov))
        df = pd.DataFrame(rows)
        if df["group"].nunique() < 2:
            raise ValueError("need at least two groups for ANCOVA")

        model = smf.ols("error ~ C(group) + covariate", data=df).fit()
        table = anova_lm(model, typ=2)

        group_row = next(r for r in table.index if r.startswith("C(group)"))
        return {
            "f_statistic": float(table.loc[group_row, "F"]),
            "p_value": float(table.loc[group_row, "PR(>F)"]),
            "df_group": float(table.loc[group_row, "df"]),
            "df_residual": float(table.loc["Residual", "df"]),
            "n_observations": int(len(df)),
            "group_means": {g: float(df.loc[df["group"] == g, "error"].mean()) for g in group_errors},
        }


# ==============================================================================
# 9. DEMO DRIVER
# ==============================================================================
def run_demo(seed: int = 42, n_qubits: int = 8, epochs: int = 5, batch_size: int = 16) -> Dict[str, Any]:
    """Runs the full demo end to end and returns every metric produced, so
    tests can assert on real values instead of parsing stdout."""
    print("\n" + "=" * 72)
    print(" STANDALONE QGAN DEMO -- synthetic data only, not dissertation output ")
    print("=" * 72 + "\n")

    logger = DemoLogger()
    all_metrics: Dict[str, Any] = {}
    t_pipeline = time.time()

    # ---- Phase 1: synthetic data ----
    t0 = time.time()
    gen = SyntheticSeriesGenerator(sequence_length=n_qubits, num_samples=1000, seed=seed)
    data, regimes = gen.generate()
    data_tensor = torch.tensor(data, dtype=torch.float32)
    logger.log_step("PHASE_1_DATA", 1, (time.time() - t0) * 1000,
                    {"n_windows": len(data), "n_regimes": int(len(np.unique(regimes)))},
                    "Synthetic series generated (NOT real financial data)")

    # ---- Phase 2: QGAN training ----
    t0 = time.time()
    q_circuit = QuantumGeneratorCircuit(n_qubits=n_qubits, circuit_depth=3, noise_rate=0.01, seed=seed)
    discriminator = ClassicalDiscriminator(input_dim=n_qubits, hidden_dim=32)
    trainer = QGANTrainer(q_circuit, discriminator, logger, lr_g=0.01, lr_d=0.002, seed=seed)
    rng = np.random.default_rng(seed)
    epoch_metrics = []
    for epoch in range(1, epochs + 1):
        idx = rng.choice(data_tensor.size(0), size=batch_size, replace=False)
        epoch_metrics.append(trainer.train_epoch(data_tensor[idx], epoch))
    all_metrics["training"] = epoch_metrics
    logger.log_step("PHASE_2_QGAN_TRAIN", 2, (time.time() - t0) * 1000,
                    {"epochs": epochs, "qubits": n_qubits}, "QGAN training complete")

    # ---- Phase 3: synthetic sample generation ----
    t0 = time.time()
    with torch.no_grad():
        synthetic = q_circuit.generate(trainer.weights_g, batch_size=200)
    logger.log_step("PHASE_3_SYNTHESIS", 3, (time.time() - t0) * 1000,
                    {"n_samples": synthetic.size(0)}, "Synthetic samples generated")

    # ---- Phase 4: adversarial check ----
    t0 = time.time()
    proxy_forecaster = nn.Linear(n_qubits, 1)
    attack_engine = AdversarialAttackEngine(epsilon=0.10)
    test_inputs = data_tensor[:100]
    test_targets = torch.tensor(rng.normal(size=(100, 1)), dtype=torch.float32)
    asr_report = attack_engine.evaluate_asr(proxy_forecaster, test_inputs, test_targets)
    all_metrics["adversarial"] = asr_report
    logger.log_step("PHASE_4_SECURITY", 4, (time.time() - t0) * 1000, asr_report,
                    f"FGSM check complete (ASR definition: {asr_report['asr_definition']})")

    # ---- Phase 5: tomography + real ANCOVA ----
    t0 = time.time()
    entropy = QuantumTomographyEngine.calculate_entanglement_entropy(synthetic[0].numpy())

    # Two synthetic error populations + a covariate, purely to demonstrate a
    # real ANCOVA call -- these are NOT measurements of the trained model
    # above; see README.md. A caller with real per-sample errors and a real
    # covariate (e.g. VIX) would pass those instead.
    n = 60
    covariate = rng.normal(20, 5, size=n)
    classical_errors = 0.5 + 0.01 * covariate + rng.normal(0, 0.05, size=n)
    quantum_errors = 0.4 + 0.01 * covariate + rng.normal(0, 0.05, size=n)
    ancova = QuantumTomographyEngine.run_ancova(
        {"classical": classical_errors, "quantum": quantum_errors},
        {"classical": covariate, "quantum": covariate},
    )
    all_metrics["tomography"] = {"entanglement_entropy": entropy, "ancova": ancova}
    logger.log_step("PHASE_5_TOMOGRAPHY_ANCOVA", 5, (time.time() - t0) * 1000,
                    {"entanglement_entropy": entropy, **ancova},
                    f"Entropy={entropy:.3f}, ANCOVA p={ancova['p_value']:.4f} "
                    f"(real test; NOT clamped to appear significant)")

    total_s = time.time() - t_pipeline
    print("\n" + "=" * 72)
    print(f" DEMO COMPLETE | {total_s:.2f}s | trace: {logger.log_path}")
    print("=" * 72 + "\n")
    all_metrics["total_seconds"] = total_s
    return all_metrics


if __name__ == "__main__":
    run_demo()
