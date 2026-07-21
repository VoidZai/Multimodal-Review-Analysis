"""Central seeding so every stochastic step in the pipeline is reproducible.

Anything that samples, shuffles, or otherwise draws randomness (stratified
sampling, train/val splits, etc.) must call `set_global_seed` first and
should prefer the returned `random.Random` / `numpy.random.Generator`
instances over the global `random`/`numpy.random` state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    """Handles returned by `set_global_seed` for explicit, non-global use."""

    seed: int
    python_random: random.Random
    numpy_rng: np.random.Generator


def set_global_seed(seed: int) -> SeedState:
    """Seed Python's `random` and NumPy's global RNG, and return dedicated
    generators for callers that want to avoid mutating global state.

    Args:
        seed: the seed to use everywhere in this run.

    Returns:
        A `SeedState` bundling the seed and two independent generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    return SeedState(
        seed=seed,
        python_random=random.Random(seed),
        numpy_rng=np.random.default_rng(seed),
    )
