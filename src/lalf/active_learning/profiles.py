"""Construct the three complementary search directions from the verified frontier.

The profiles are not three independent objectives.  They are three views of the
same feasible f-ratio Pareto frontier: one advances oscillator strength, one
advances the ISC ratio, and one fills the trade-off region between them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lalf.active_learning.selection import pareto_layers
from lalf.models import TargetProfile, Thresholds


_FOCI = {
    "f_focus": "Prioritize oscillator strength while preserving a useful ISC ratio.",
    "ratio_focus": "Prioritize SOCME/DeltaEST while preserving useful absorption.",
    "balance_focus": "Improve the trade-off between oscillator strength and ISC ratio.",
}


def _record(row: pd.Series) -> dict[str, object]:
    return {
        "molecule_id": str(row.get("molecule_id", "")),
        "smiles": str(row.get("smiles", "")),
        "s1": float(row.get("s1", np.nan)),
        "est": float(row.get("est", np.nan)),
        "f": float(row.get("f", np.nan)),
        "ratio": float(row.get("ratio", np.nan)),
    }


def build_profiles(
    history: pd.DataFrame, thresholds: Thresholds = Thresholds()
) -> list[TargetProfile]:
    """Build f-focus, ratio-focus, and balance-focus profiles.

    Only verified molecules inside the configured S1/DeltaEST window are used
    when possible.  A non-dominated frontier is extracted before anchors are
    chosen, preventing dominated but numerically extreme records from steering
    the next LLM design step.
    """
    if history.empty:
        return [TargetProfile(name=name, focus=focus) for name, focus in _FOCI.items()]

    work = history.copy()
    for column in ["s1", "est", "f", "ratio"]:
        work[column] = pd.to_numeric(work.get(column), errors="coerce")
    feasible = work.loc[
        work["s1"].between(thresholds.s1_low, thresholds.s1_high)
        & (work["est"] < thresholds.est_max)
    ].dropna(subset=["smiles", "f", "ratio"])
    if feasible.empty:
        feasible = work.dropna(subset=["smiles", "f", "ratio"])
    if feasible.empty:
        return [TargetProfile(name=name, focus=focus) for name, focus in _FOCI.items()]

    layer_input = feasible.rename(columns={"f": "mean_f", "ratio": "mean_ratio"})
    frontier = feasible.loc[pareto_layers(layer_input).to_numpy() == 0].copy()
    if not frontier.empty:
        feasible = frontier

    f_anchor = feasible.sort_values(["f", "ratio"], ascending=False).iloc[0]
    ratio_anchor = feasible.sort_values(["ratio", "f"], ascending=False).iloc[0]
    f_span = max(float(feasible["f"].max() - feasible["f"].min()), 1e-12)
    ratio_span = max(float(feasible["ratio"].max() - feasible["ratio"].min()), 1e-12)
    balance_score = (feasible["f"] - feasible["f"].min()) / f_span + (
        feasible["ratio"] - feasible["ratio"].min()
    ) / ratio_span
    balance_anchor = feasible.loc[balance_score.idxmax()]

    return [
        TargetProfile("f_focus", _FOCI["f_focus"], (_record(f_anchor),)),
        TargetProfile("ratio_focus", _FOCI["ratio_focus"], (_record(ratio_anchor),)),
        TargetProfile(
            "balance_focus",
            _FOCI["balance_focus"],
            (_record(f_anchor), _record(ratio_anchor), _record(balance_anchor)),
        ),
    ]
