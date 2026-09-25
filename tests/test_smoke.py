"""
End-to-end smoke test on tiny synthetic data.

This does NOT validate that the dissertation's numbers are correct --
it can't, without real market data and real GPU-scale training. What
it DOES validate: every stage of the pipeline (windowing, LSTM
training, GAN training, quantum generator training + gradient flow,
adversarial attacks, entanglement tomography, statistical tests)
actually executes without crashing, on data shaped like the real
thing but tiny enough to run in seconds on CPU.

Run with:  pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines.classical_lstm import ClassicalLSTM
from src.baselines.classical_gan_llm import ClassicalGANLLM
from src.baselines.qgan_llm import QGANLLM
from src.baselines.qlstm_forecaster import QLSTMForecaster
from src.data.windowing import WindowedSequenceDataset
from src.evaluation.metrics import rmse, mae, frechet_distance, maximum_mean_discrepancy
from src.evaluation.statistical_tests import ancova, chi_square_test, independent_ttest, pearson_correlation_with_ci
from src.quantum.circuits import QLSTMGenerator
from src.quantum.tomography import entanglement_metrics
from src.attacks.adversarial import compute_attack_success_rate


N_ROWS = 400
N_FEATURES = 8   # small on purpose -- real config uses 32
SEQ_LEN = 10      # small on purpose -- real config uses 60


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(N_ROWS, N_FEATURES)).astype(np.float32)
    y = X[:, 0] + 0.1 * rng.normal(size=N_ROWS).astype(np.float32)
    split = N_ROWS // 2
    return {
        "X_train": X[:split], "y_train": y[:split],
        "X_val": X[split: split + N_ROWS // 4], "y_val": y[split: split + N_ROWS // 4],
        "X_test": X[split + N_ROWS // 4:], "y_test": y[split + N_ROWS // 4:],
    }


def test_windowed_dataset(synthetic_data):
    ds = WindowedSequenceDataset(synthetic_data["X_train"], synthetic_data["y_train"], sequence_length=SEQ_LEN)
    assert len(ds) == len(synthetic_data["X_train"]) - SEQ_LEN
    seq, target, last_price = ds[0]
    assert seq.shape == (SEQ_LEN, N_FEATURES)
    assert target.dim() == 0
    assert last_price.dim() == 0


def test_classical_lstm_trains(synthetic_data):
    model = ClassicalLSTM(config={
        "input_size": N_FEATURES, "hidden_size": 8, "num_layers": 1,
        "epochs": 2, "batch_size": 16, "sequence_length": SEQ_LEN,
        "early_stopping_patience": 5,
    })
    os.makedirs("models", exist_ok=True)
    model.train(synthetic_data["X_train"], synthetic_data["y_train"],
                synthetic_data["X_val"], synthetic_data["y_val"])
    metrics = model.evaluate(synthetic_data["X_test"], synthetic_data["y_test"])
    assert "rmse" in metrics and metrics["rmse"] >= 0
    assert not np.isnan(metrics["rmse"])


def test_classical_gan_llm_trains(synthetic_data):
    model = ClassicalGANLLM(config={
        "latent_dim": N_FEATURES, "output_dim": N_FEATURES,
        "generator_hidden": 8, "discriminator_hidden": 8,
        "epochs": 2, "batch_size": 16,
    })
    model.train(synthetic_data["X_train"], synthetic_data["y_train"],
                synthetic_data["X_val"], synthetic_data["y_val"])
    attack_cfg = {"fgsm_epsilon": 0.1, "pgd_epsilon": 0.1, "pgd_alpha": 0.01,
                  "pgd_steps": 2, "cw_c": 1.0, "cw_steps": 2, "attacks": ["fgsm"]}
    last_prices = synthetic_data["X_test"][:, 0]
    metrics = model.evaluate(synthetic_data["X_test"], synthetic_data["y_test"],
                              attack_cfg=attack_cfg, last_input_prices=last_prices)
    assert "rmse" in metrics
    assert "asr" in metrics
    assert 0 <= metrics["asr"] <= 100


def test_qgan_generator_gradients_actually_flow():
    """The specific bug this whole rebuild started from: does backward()
    actually reach the quantum circuit's parameters, or does it silently
    stop at the classical output layer? This test fails loudly if the
    original detach-to-numpy bug ever creeps back in."""
    torch.manual_seed(0)
    gen = QLSTMGenerator(n_qubits=4, n_layers=1, n_features=N_FEATURES,
                          entanglement="ring", noise_strength=0.0)
    theta_before = gen.theta.detach().clone()

    z = torch.randn(4, N_FEATURES, requires_grad=False)
    out = gen(z)
    loss = out.sum()
    loss.backward()

    assert gen.theta.grad is not None, "theta.grad is None -- gradients did not reach the quantum circuit"
    assert torch.any(gen.theta.grad != 0), "theta.grad is all zeros -- gradients reached but vanished"

    # confirm an optimizer step actually changes theta (end-to-end trainability)
    opt = torch.optim.SGD(gen.parameters(), lr=1.0)
    opt.step()
    assert not torch.allclose(theta_before, gen.theta.detach()), "theta did not update after an optimizer step"


def test_qgan_llm_trains_end_to_end(synthetic_data):
    model = QGANLLM(config={
        "n_qubits": 4, "n_layers": 1, "n_features": N_FEATURES,
        "entanglement": "ring", "noise_strength": 0.01,
        "discriminator_hidden": 8, "epochs": 2, "batch_size": 16,
    })
    model.train(synthetic_data["X_train"], synthetic_data["y_train"],
                synthetic_data["X_val"], synthetic_data["y_val"])
    attack_cfg = {"fgsm_epsilon": 0.1, "pgd_epsilon": 0.1, "pgd_alpha": 0.01,
                  "pgd_steps": 2, "cw_c": 1.0, "cw_steps": 2, "attacks": ["fgsm"]}
    last_prices = synthetic_data["X_test"][:, 0]
    metrics = model.evaluate(synthetic_data["X_test"], synthetic_data["y_test"],
                              attack_cfg=attack_cfg, last_input_prices=last_prices)
    assert "rmse" in metrics
    assert "entanglement_entropy" in metrics
    assert metrics["entanglement_entropy"] >= 0  # entropy is non-negative by definition


def test_entanglement_entropy_is_zero_for_unentangled_circuit():
    """Sanity check on the tomography math itself: a circuit with NO
    entangling gates at all must report ~0 entanglement entropy. If this
    fails, the partial-trace/von-Neumann-entropy implementation is wrong,
    independent of anything about the GAN."""
    import pennylane as qml

    n_qubits = 4

    def unentangled_circuit(weights):
        for i in range(n_qubits):
            qml.RY(weights[i], wires=i)
        # no CNOTs at all -- product state, zero entanglement across any cut

    weights = np.array([0.3, 0.7, 1.1, 0.4])
    report = entanglement_metrics(unentangled_circuit, weights, n_qubits, cut=2)
    assert report["entanglement_entropy"] < 1e-6, \
        f"Expected ~0 entropy for an unentangled circuit, got {report['entanglement_entropy']}"
    assert abs(report["purity"] - 1.0) < 1e-6, "Reduced state of a product state should be pure (purity=1)"


def test_qlstm_forecaster_trains_end_to_end(synthetic_data):
    """The class that answers 'where is the standalone quantum
    forecaster' -- no GAN, no discriminator, the quantum circuit
    predicts directly."""
    model = QLSTMForecaster(config={
        "n_qubits": 4, "n_layers": 1, "n_features": N_FEATURES,
        "entanglement": "ring", "noise_strength": 0.01,
        "epochs": 2, "batch_size": 16,
    })
    model.train(synthetic_data["X_train"], synthetic_data["y_train"],
                synthetic_data["X_val"], synthetic_data["y_val"])
    attack_cfg = {"fgsm_epsilon": 0.1, "pgd_epsilon": 0.1, "pgd_alpha": 0.01,
                  "pgd_steps": 2, "cw_c": 1.0, "cw_steps": 2, "attacks": ["fgsm"]}
    last_prices = synthetic_data["X_test"][:, 0]
    metrics = model.evaluate(synthetic_data["X_test"], synthetic_data["y_test"],
                              attack_cfg=attack_cfg, last_input_prices=last_prices)
    assert "rmse" in metrics
    assert "entanglement_entropy" in metrics
    assert "asr" in metrics


def test_metrics_run():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 5))
    b = rng.normal(size=(50, 5)) + 0.5
    assert rmse(a[:, 0], b[:, 0]) > 0
    assert mae(a[:, 0], b[:, 0]) > 0
    assert np.isfinite(frechet_distance(a, b))
    assert np.isfinite(maximum_mean_discrepancy(a, b))


def test_statistical_tests_run():
    rng = np.random.default_rng(0)
    dv = np.concatenate([rng.normal(0, 1, 50), rng.normal(1, 1, 50)])
    group = np.array([0] * 50 + [1] * 50)
    covariate = rng.normal(0, 1, 100)
    result = ancova(dv, group, covariate)
    assert "p_value" in result and 0 <= result["p_value"] <= 1

    table = np.array([[20, 30], [25, 25]])
    chi_result = chi_square_test(table)
    assert 0 <= chi_result["p_value"] <= 1

    t_result = independent_ttest(rng.normal(0, 1, 30), rng.normal(1, 1, 30))
    assert "cohens_d" in t_result

    x = rng.normal(0, 1, 100)
    y = x * 0.5 + rng.normal(0, 1, 100)
    corr_result = pearson_correlation_with_ci(x, y)
    assert -1 <= corr_result["r"] <= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
