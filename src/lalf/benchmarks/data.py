"""Main-package CSV schema at the boundary; label-free selector features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

LABELS = ["s1", "est", "f", "ratio"]
ALIASES = {"smiles": "SMILES", "s1": "S1", "est": "EST", "ratio": "SOCME/EST"}


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate user input; retain first duplicate SMILES as in the paper."""
    required = ["molecule_id", "smiles", *LABELS]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    work = frame[required + [c for c in ("socme", "t_state") if c in frame]].copy()
    for column in ("molecule_id", "smiles"):
        if work[column].isna().any():
            raise ValueError(f"Missing {column}")
        work[column] = work[column].astype(str).str.strip()
        if work[column].eq("").any():
            raise ValueError(f"Empty {column}")
    if work["molecule_id"].duplicated().any():
        raise ValueError("molecule_id must be unique")
    work[LABELS] = work[LABELS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(work[LABELS].to_numpy(float)).all():
        raise ValueError("All four property labels must be finite")
    work = work.drop_duplicates("smiles", keep="first").reset_index(drop=True)
    if any(Chem.MolFromSmiles(s) is None for s in work.smiles):
        raise ValueError("Invalid SMILES in candidate library")
    work.insert(0, "id", np.arange(len(work)))
    return work.rename(columns=ALIASES)


def public_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={v: k for k, v in ALIASES.items()}).drop(
        columns=["id"], errors="ignore"
    )


class Oracle:
    """A software boundary: full labels stay here, never in Library."""

    def __init__(self, frame: pd.DataFrame):
        self.__frame = frame.copy(deep=True)
        self.queried: set[int] = set()

    def query(self, ids) -> pd.DataFrame:
        ids = list(map(int, ids))
        if len(ids) != len(set(ids)) or self.queried.intersection(ids):
            raise ValueError("Repeated label query")
        if any(i < 0 or i >= len(self.__frame) for i in ids):
            raise ValueError("Unknown molecule position")
        result = self.__frame.iloc[ids].copy()
        self.queried.update(ids)
        return result


class Library:
    """Structures only; Morgan/PCA features never use target labels."""

    def __init__(self, structures: pd.DataFrame):
        self.structures = structures[["id", "molecule_id", "SMILES"]].copy()
        self.smiles = self.structures.SMILES.tolist()
        self.gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.fps = [self.gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in self.smiles]
        self.bits = np.asarray([np.asarray(fp) for fp in self.fps], dtype=np.float32)
        self._x = None

    @property
    def x(self):
        if self._x is None:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            dim = min(128, len(self.bits), self.bits.shape[1])
            projected = (
                PCA(n_components=dim, svd_solver="randomized", random_state=0)
                .fit_transform(self.bits)
                .astype(np.float32)
            )
            self._x = StandardScaler().fit_transform(projected).astype(np.float64)
        return self._x

    def similarity(self, smiles, ids):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid generated SMILES")
        fp = self.gen.GetFingerprint(mol)
        return np.asarray(DataStructs.BulkTanimotoSimilarity(fp, [self.fps[int(i)] for i in ids]))

    def mmr(self, smiles, ids, k, lam):
        ids = np.asarray(ids, dtype=int)
        rel = self.similarity(smiles, ids)
        penalty = np.zeros(len(ids))
        selected = []
        for step in range(min(k, len(ids))):
            score = rel.copy() if step == 0 else lam * rel - (1 - lam) * penalty
            score[selected] = -np.inf
            pos = int(np.argmax(score))
            selected.append(pos)
            sims = np.asarray(
                DataStructs.BulkTanimotoSimilarity(
                    self.fps[int(ids[pos])], [self.fps[int(i)] for i in ids]
                )
            )
            penalty = np.maximum(penalty, sims)
        return ids[selected].tolist()
