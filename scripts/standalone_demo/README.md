# Standalone QGAN Demo (NOT the dissertation pipeline)

`qgan_llm_demo.py` in this folder is a small, self-contained, fully
synthetic demonstration of a quantum-generator / classical-discriminator
GAN with an adversarial-robustness check and a quantum-tomography metric.

**It is intentionally kept separate from `src/`.** It does not use
`config/default_config.yaml`, the real EUR/USD data pipeline, or the
project's baseline models (`src/baselines/`), and its numbers must never
be cited as, or confused with, Chapter 4 results. Its only purpose is to
be a small, readable, *honest* worked example of the mechanics (a
PennyLane PQC generator, a spectral-normalized discriminator, an FGSM
check, entanglement entropy, and a real ANCOVA) on made-up data.

## What was fixed from the version this was based on

1. **Syntax error.** `def generate_synthetic_forex_data((self) -> ...` had
   an extra `(` and could not be imported, let alone run.
2. **Fabricated "ANCOVA" p-value.** The original `compute_ancova_pvalue`
   computed `p = exp(-effect_size * 45 / 2)` and then clamped it with
   `max(min(p, 0.05), 0.001)` — the upper clamp means it returns a
   "significant" p-value (≤ 0.05) for almost any input, including cases
   where the "quantum" model is *worse* than the baseline (verified: with
   `clean_rmse=0.49, qgan_rmse=10.0`, the old function still returned
   `p=0.05`). It was asserting significance, not testing for it. This
   version instead runs a real one-way ANCOVA (`statsmodels`, group +
   covariate, `anova_lm`) over synthetic per-sample errors and reports
   whatever p-value that actually produces, including a non-significant
   one when the synthetic data doesn't support significance.
3. **Fake FID.** `fid_proxy` was the squared difference of the two
   batches' means (`torch.mean(...) ** 2`) — not FID, which requires each
   distribution's mean *and* covariance (Fréchet distance between two
   Gaussians). Replaced with an actual Fréchet distance over the
   generator's and real data's per-feature mean and covariance.
4. **ASR definition mismatch with the real pipeline.** The original used
   "perturbation pushes absolute error above a fixed threshold." The real
   pipeline (`src/attacks/adversarial.py`) uses a directional-flip
   definition instead. Both are legitimate but different, and reporting
   one number as if it were the other would be misleading. This demo now
   labels its ASR explicitly as `asr_definition: "abs_error_above_threshold"`
   in its output so the two are never conflated.
5. **Undocumented, made-up performance numbers in the accompanying
   summary** ("320.5 minutes down to 49.8 minutes on `lightning.gpu`",
   "<3.2ms live inference") are not reproduced anywhere in this folder.
   `default.qubit` vs `lightning.gpu` timing depends entirely on your
   hardware; if you want that comparison, time it yourself with
   `time python -m scripts.standalone_demo.qgan_llm_demo` on both device
   strings and record what you actually measured.

## Run

```bash
python -m scripts.standalone_demo.qgan_llm_demo
python -m pytest tests/test_standalone_demo.py -v
```

Everything in this script runs on synthetic sine-wave-plus-noise data
generated in-process — no external files, no network, no GPU required
(though `qml.device("default.qubit", ...)` can be swapped for
`lightning.qubit`/`lightning.gpu` if installed).
