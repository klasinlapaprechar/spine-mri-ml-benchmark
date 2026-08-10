"""Tiny synthetic mid-slice tensors for smoke tests (no clinical data)."""
from __future__ import annotations
import numpy as np

def make_batch(n: int = 8, n_classes: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3, 32, 32)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    return x, y
