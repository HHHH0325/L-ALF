"""Feasibility-aware non-dominated sorting and Pareto-UCB batch selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from lalf.models import Thresholds


def hard_feasible_mask(df: pd.DataFrame, thresholds: Thresholds) -> pd.Series:
    """Return the predicted photophysical feasibility mask."""
    s1 = pd.to_numeric(df["mean_s1"], errors="coerce")
    est = pd.to_numeric(df["mean_est"], errors="coerce")
    return s1.between(thresholds.s1_low, thresholds.s1_high) & (
        est < thresholds.est_max
    )


def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left >= right) and np.any(left > right))


def pareto_layers(df: pd.DataFrame) -> pd.Series:
    """Assign successive Pareto layers while maximizing f and SOCME/DeltaEST.

    Layer zero is the current non-dominated boundary.  Removing it and repeating
    yields layer one, layer two, and so on.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype="Int64")

    values = df[["mean_f", "mean_ratio"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    valid = np.isfinite(values).all(axis=1)
    layers = np.full(len(df), -1, dtype=int)
    remaining = {index for index, is_valid in enumerate(valid) if is_valid}
    layer = 0

    while remaining:
        front = []
        for candidate in sorted(remaining):
            if not any(
                _dominates(values[other], values[candidate])
                for other in remaining
                if other != candidate
            ):
                front.append(candidate)
        if not front:
            break
        for candidate in front:
            layers[candidate] = layer
            remaining.remove(candidate)
        layer += 1

    return pd.Series(layers, index=df.index, dtype="Int64")


def select_pareto_ucb(
    df: pd.DataFrame,
    top_k: int,
    beta: float,
    thresholds: Thresholds,
) -> pd.DataFrame:
    """Select a DFT batch by feasibility, Pareto layer, and UCB.

    The mean score rewards predicted performance, while the standard-deviation
    term preserves candidates whose repeated LLM evaluations are less stable.
    """
    if top_k < 1 or df.empty:
        return df.iloc[0:0].copy()

    work = df.copy()
    work["predicted_feasible"] = hard_feasible_mask(work, thresholds)
    work["ucb_score"] = pd.to_numeric(work["mean_score"], errors="coerce") + float(
        beta
    ) * pd.to_numeric(work["std_score"], errors="coerce").fillna(0.0)
    work["pareto_layer"] = -1

    feasible = work.loc[work["predicted_feasible"]].copy()
    feasible["pareto_layer"] = pareto_layers(feasible)
    feasible = feasible.sort_values(
        ["pareto_layer", "ucb_score", "molecule_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )

    needed = max(0, int(top_k) - len(feasible))
    infeasible = work.loc[~work["predicted_feasible"]].copy()
    infeasible = infeasible.sort_values(
        ["ucb_score", "molecule_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(needed)

    selected = pd.concat([feasible.head(top_k), infeasible], ignore_index=True).head(
        top_k
    )
    selected["selection_rank"] = np.arange(1, len(selected) + 1, dtype=int)
    return selected
