"""
Reproducibility utilities.

The original code only seeded torch and numpy inside
ExperimentRunner.__init__. That's incomplete: Python's own `random`
module (used by some sampling paths) and PennyLane's internal RNG
were left unseeded, and the seed was set once at runner construction
rather than before *every* model's build/train call — meaning model
N's training run consumes random state left over from model N-1,
so results are not independently reproducible per-model, only as a
whole fixed sequence. This module fixes both problems: call
set_all_seeds() immediately before building/training each individual
model, not just once at the top of the run.
"""
import os
import random
import numpy as np
import torch


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic CuDNN (slower, but required for reproducibility claims)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int = 42) -> torch.Generator:
    """For DataLoader shuffling — pass this as generator= so shuffling order is reproducible too."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
