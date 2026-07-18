"""Map LLM design directions back into a real molecular candidate library.

Morgan fingerprints provide structural relevance.  Greedy MMR then penalizes
redundancy so that each DFT batch explores several related structural variants
instead of repeatedly selecting the same local analogue family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from lalf.models import Thresholds


_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return None
    return _GENERATOR.GetFingerprint(molecule)


def valid_smiles(smiles: str) -> bool:
    return fingerprint(smiles) is not None


def local_references(
    history: pd.DataFrame,
    target_smiles: str,
    thresholds: Thresholds,
    positive_k: int = 2,
    negative_k: int = 4,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Retrieve nearby verified positive and negative examples for LLM context."""
    target_fp = fingerprint(target_smiles)
    if target_fp is None or history.empty:
        return [], []
    rows = []
    for _, row in history.iterrows():
        fp = fingerprint(str(row.get("smiles", "")))
        if fp is None:
            continue
        record = row.to_dict()
        record["similarity"] = float(DataStructs.TanimotoSimilarity(target_fp, fp))
        s1 = pd.to_numeric(pd.Series([row.get("s1")]), errors="coerce").iloc[0]
        est = pd.to_numeric(pd.Series([row.get("est")]), errors="coerce").iloc[0]
        record["_feasible"] = bool(
            pd.notna(s1)
            and pd.notna(est)
            and thresholds.s1_low <= float(s1) <= thresholds.s1_high
            and float(est) < thresholds.est_max
        )
        rows.append(record)
    rows.sort(key=lambda item: (-float(item["similarity"]), str(item.get("molecule_id", ""))))
    positive = [dict(row) for row in rows if row["_feasible"]][:positive_k]
    negative = [dict(row) for row in rows if not row["_feasible"]][:negative_k]
    for row in positive + negative:
        row.pop("_feasible", None)
    return positive, negative


def expand_candidates(
    pool: pd.DataFrame,
    target_smiles: str,
    n_pick: int,
    lambda_value: float,
    excluded_ids: set[str] | None = None,
    similarity_threshold: float = 0.0,
) -> pd.DataFrame:
    """Expand one target SMILES into a relevance-diversity candidate set.

    At each greedy step the score is lambda times target similarity minus
    one-minus-lambda times similarity to the already selected set.
    """
    target_fp = fingerprint(target_smiles)
    if target_fp is None or n_pick < 1:
        return pd.DataFrame(
            columns=["molecule_id", "smiles", "similarity", "mmr_score", "candidate_rank"]
        )

    excluded_ids = {str(value) for value in (excluded_ids or set())}
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for _, row in pool.iterrows():
        molecule_id = str(row.get("molecule_id", "")).strip()
        smiles = str(row.get("smiles", "")).strip()
        if not molecule_id or molecule_id in excluded_ids or molecule_id in seen_ids:
            continue
        fp = fingerprint(smiles)
        if fp is None:
            continue
        similarity = float(DataStructs.TanimotoSimilarity(target_fp, fp))
        if similarity < float(similarity_threshold):
            continue
        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "similarity": similarity,
                "_fingerprint": fp,
            }
        )
        seen_ids.add(molecule_id)

    if not rows:
        return pd.DataFrame(
            columns=["molecule_id", "smiles", "similarity", "mmr_score", "candidate_rank"]
        )

    selected: list[dict[str, object]] = []
    remaining = list(rows)
    while remaining and len(selected) < int(n_pick):
        best_index = 0
        best_score = -np.inf
        for index, candidate in enumerate(remaining):
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    float(
                        DataStructs.TanimotoSimilarity(
                            candidate["_fingerprint"], prior["_fingerprint"]
                        )
                    )
                    for prior in selected
                )
            score = float(lambda_value) * float(candidate["similarity"]) - (
                1.0 - float(lambda_value)
            ) * diversity_penalty
            tie_key = (score, float(candidate["similarity"]), str(candidate["molecule_id"]))
            best = remaining[best_index]
            best_tie = (
                best_score,
                float(best["similarity"]),
                str(best["molecule_id"]),
            )
            if tie_key > best_tie:
                best_index = index
                best_score = score
        chosen = remaining.pop(best_index)
        chosen["mmr_score"] = float(best_score)
        selected.append(chosen)

    result = pd.DataFrame(selected)
    result["candidate_rank"] = np.arange(1, len(result) + 1, dtype=int)
    return result.drop(columns=["_fingerprint"])
