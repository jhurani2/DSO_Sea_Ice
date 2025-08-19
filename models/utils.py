"""Utility functions: metrics and checkpoint helpers."""
from pathlib import Path
from typing import Tuple

import numpy as np


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def r2_score(pred: np.ndarray, truth: np.ndarray) -> float:
    # simple R2
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.state_dict(), str(path))


def load_checkpoint(model, path: Path):
    import torch
    state = torch.load(str(path), map_location='cpu')
    model.load_state_dict(state)
    return model
