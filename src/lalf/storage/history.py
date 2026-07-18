"""Maintain the verified label memory that drives every subsequent AL round."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_COLUMNS = [
    "molecule_id",
    "smiles",
    "s1",
    "est",
    "f",
    "socme",
    "ratio",
    "t_state",
]


def _reject_internal_id_conflicts(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        return
    conflicts = df.groupby("molecule_id")["smiles"].nunique(dropna=False)
    bad = conflicts.loc[conflicts > 1].index.astype(str).tolist()
    if bad:
        raise ValueError(
            f"{context} maps molecule IDs to different SMILES: {bad[:5]}"
        )


def load_history(path: str | Path) -> pd.DataFrame:
    history_path = Path(path)
    if not history_path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    history = pd.read_csv(history_path)
    missing = sorted(set(HISTORY_COLUMNS) - set(history.columns))
    if missing:
        raise ValueError(f"History file is missing columns: {missing}")
    history["molecule_id"] = history["molecule_id"].astype(str).str.strip()
    history["smiles"] = history["smiles"].astype(str).str.strip()
    _reject_internal_id_conflicts(history, "History")
    return history


def _validate_results(results: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(HISTORY_COLUMNS) - set(results.columns))
    if missing:
        raise ValueError(f"DFT results are missing columns: {missing}")
    clean = results[HISTORY_COLUMNS].copy()
    clean["molecule_id"] = clean["molecule_id"].astype(str).str.strip()
    clean["smiles"] = clean["smiles"].astype(str).str.strip()
    clean = clean.loc[(clean["molecule_id"] != "") & (clean["smiles"] != "")]
    for column in ["s1", "est", "f", "socme", "ratio", "t_state"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    required_numeric = clean[["s1", "est", "f", "socme", "ratio"]].to_numpy(float)
    if len(clean) and not np.isfinite(required_numeric).all():
        raise ValueError("DFT results contain missing or non-finite required labels")
    return clean.drop_duplicates("molecule_id", keep="last")


def merge_history_atomic(
    history_path: str | Path, results: pd.DataFrame
) -> pd.DataFrame:
    """Merge new oracle labels by stable ID without allowing structure mismatch."""
    path = Path(history_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_history(path)
    new_rows = _validate_results(results)
    _reject_internal_id_conflicts(new_rows, "DFT results")
    if not existing.empty and not new_rows.empty:
        existing_smiles = existing.set_index("molecule_id")["smiles"].to_dict()
        conflicts = [
            molecule_id
            for molecule_id, smiles in new_rows.set_index("molecule_id")["smiles"].items()
            if molecule_id in existing_smiles and existing_smiles[molecule_id] != smiles
        ]
        if conflicts:
            raise ValueError(
                "History update maps existing molecule IDs to different SMILES: "
                f"{conflicts[:5]}"
            )
    if existing.empty:
        merged = new_rows.copy()
    elif new_rows.empty:
        merged = existing[HISTORY_COLUMNS].copy()
    else:
        merged = pd.concat([existing[HISTORY_COLUMNS], new_rows], ignore_index=True)
    merged = merged.drop_duplicates("molecule_id", keep="last").reset_index(drop=True)

    temporary = path.with_name(path.name + ".tmp")
    merged.to_csv(temporary, index=False, encoding="utf-8")
    verified = load_history(temporary)
    if len(verified) != len(merged):
        raise RuntimeError("Atomic history verification changed the row count")
    os.replace(temporary, path)
    return verified
