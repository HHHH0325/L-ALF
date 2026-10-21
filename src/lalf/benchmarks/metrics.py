"""Post-hoc evaluation only; full-library labels never reach the selectors."""

import numpy as np

from .core import COLS, front_mask, hard
from .data import prepare_frame
from .settings import P


def evaluate(selected, reference):
    """Reference labels set HV normalization after acquisition is complete."""
    y = prepare_frame(selected)[COLS].to_numpy(float)
    global_y = prepare_frame(reference)[COLS].to_numpy(float)
    feasible = hard(y)
    global_feasible = global_y[hard(global_y)]
    maxima = (
        np.log1p(np.maximum(global_feasible[:, 2:], 0)).max(0)
        if len(global_feasible)
        else np.ones(2)
    )
    vals = np.log1p(np.maximum(y[feasible, 2:], 0)) / np.maximum(maxima, 1e-12)
    hv, last = 0.0, 0.0
    if len(vals):
        vals = vals[front_mask(vals)]
        for f, ratio in vals[np.argsort(vals[:, 0])]:
            hv += max(f - last, 0) * ratio
            last = f
    strict = feasible & (y[:, 2] > P["strict"]["f"]) & (y[:, 3] > P["strict"]["ratio"])
    relaxed = feasible & (y[:, 2] > P["relaxed"]["f"]) & (y[:, 3] > P["relaxed"]["ratio"])
    first = None
    if strict.any():
        if "round" not in selected:
            raise ValueError("round is required to compute the first-hit batch budget")
        first = int(selected.iloc[np.flatnonzero(strict)[0]]["round"]) * 20
    return dict(
        n_labels=len(y),
        feasible_count=int(feasible.sum()),
        strict_count=int(strict.sum()),
        relaxed_count=int(relaxed.sum()),
        hypervolume=float(hv),
        first_strict_batch_budget=first,
        strict_observed=bool(strict.any()),
        censor_budget=len(y),
    )
