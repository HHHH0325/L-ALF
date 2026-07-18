"""Translate frontier geometry and local structure-property evidence into prompts."""

from __future__ import annotations

import json
from collections.abc import Sequence

from lalf.models import TargetProfile


def build_design_prompt(profile: TargetProfile) -> str:
    """Ask the LLM for one directional search target around verified anchors."""
    anchors = json.dumps(profile.anchors, ensure_ascii=False)
    return (
        "You are designing a chemically valid molecular search target. "
        f"Direction: {profile.name}. Objective: {profile.focus} "
        f"Verified anchor context: {anchors}. "
        "Return exactly one SMILES string and no explanation."
    )


def build_prediction_prompt(
    smiles: str,
    global_frontier: Sequence[dict[str, object]] = (),
    local_positive: Sequence[dict[str, object]] = (),
    local_negative: Sequence[dict[str, object]] = (),
) -> str:
    """Build a comparative prediction prompt instead of an isolated prediction.

    Global frontier samples describe the current multi-objective boundary;
    local positives and negatives expose nearby structure-property contrasts.
    """
    context = {
        "global_frontier": list(global_frontier),
        "local_positive": list(local_positive),
        "local_negative": list(local_negative),
    }
    return (
        "Estimate S1_eV, DeltaEST_eV, oscillator_strength, SOCME_over_DeltaEST, "
        "and an overall score from 0 to 10 for the candidate. "
        f"Candidate SMILES: {smiles}. Context: {json.dumps(context, ensure_ascii=False)}. "
        "Return exactly five comma-separated numbers in that order."
    )
