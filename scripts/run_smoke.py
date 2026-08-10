#!/usr/bin/env python3
"""Public smoke: synthetic tensors + balanced-accuracy wiring."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.synthetic import make_batch
from src.metrics import balanced_accuracy

def main():
    x, y = make_batch()
    # trivial "model": predict majority class in-batch
    pred = np_full = __import__("numpy").full_like(y, int((y == 0).sum() >= (y == 1).sum()))
    # slightly better than chance stub: copy labels with noise
    rng = __import__("numpy").random.default_rng(1)
    pred = y.copy(); pred[rng.random(len(y)) < 0.25] ^= 1
    bal = balanced_accuracy(y, pred, n_classes=2)
    print(f"smoke ok — batch={x.shape} balanced_accuracy={bal:.3f} (synthetic only)")

if __name__ == "__main__":
    main()
