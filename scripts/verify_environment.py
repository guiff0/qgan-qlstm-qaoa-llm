"""
Run this FIRST, before anything else. Checks that the environment is
actually ready, and tells you specifically what's missing rather than
failing partway through a multi-hour run.

Run with:  python -m scripts.verify_environment
"""
from __future__ import annotations

import importlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS_PASSED = []
CHECKS_FAILED = []


def check(description: str, fn):
    try:
        result = fn()
        if result is False:
            CHECKS_FAILED.append(description)
            print(f"  [FAIL] {description}")
        else:
            CHECKS_PASSED.append(description)
            print(f"  [ OK ] {description}" + (f" — {result}" if isinstance(result, str) else ""))
    except Exception as e:
        CHECKS_FAILED.append(f"{description}: {e}")
        print(f"  [FAIL] {description} — {e}")


def main():
    print("=" * 70)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 70)

    print("\n-- Python / packages --")
    check("Python >= 3.10", lambda: sys.version_info >= (3, 10))
    for pkg in ["torch", "pennylane", "numpy", "pandas", "scipy", "sklearn",
                "statsmodels", "yaml", "requests"]:
        check(f"Package installed: {pkg}", lambda p=pkg: importlib.import_module(p) is not None)

    print("\n-- Compute --")
    def gpu_check():
        import torch
        if torch.cuda.is_available():
            return f"CUDA available ({torch.cuda.get_device_name(0)})"
        return ("No CUDA GPU detected. Classical LSTM/GAN training will be slow on CPU; "
                "the QGAN's quantum simulator (default.qubit) is CPU-only regardless, "
                "but a GPU is strongly recommended for the classical components at "
                "5.5M-row scale. See RUNNING.md for cloud-GPU options.")
    check("GPU availability", gpu_check)

    print("\n-- Config --")
    def config_check():
        from src.utils.config import load_config
        cfg = load_config()
        return f"Loaded, {len(cfg)} top-level sections"
    check("config/default_config.yaml loads", config_check)

    print("\n-- Data files (raw) --")
    from src.utils.config import load_config
    cfg = load_config()
    for key in ["dukascopy_file", "forexsb_file", "fred_file"]:
        path = cfg["data"][key]
        check(f"Raw data file present: {path}", lambda p=path: os.path.isfile(p))

    print("\n-- Data files (processed) --")
    for split in ("train", "val", "test"):
        for prefix in ("X", "y"):
            path = os.path.join(cfg["data"]["processed_dir"], f"{prefix}_{split}.npy")
            check(f"Processed file present: {path}", lambda p=path: os.path.isfile(p))

    print("\n-- API keys --")
    check("NVIDIA_API_KEY environment variable set",
          lambda: bool(os.environ.get("NVIDIA_API_KEY")))

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(CHECKS_PASSED)} passed, {len(CHECKS_FAILED)} failed")
    print("=" * 70)
    if CHECKS_FAILED:
        print("\nFailed checks (see RUNNING.md / SETUP.md for how to resolve each):")
        for c in CHECKS_FAILED:
            print(f"  - {c}")
        sys.exit(1)
    else:
        print("\nEnvironment looks ready.")


if __name__ == "__main__":
    main()
