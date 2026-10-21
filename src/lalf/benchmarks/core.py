"""Queried-label-only primitives ported from the manuscript comparisons.

Public I/O uses the main package schema. Internal column aliases retain the
paper implementation so its prompts and surrogate targets remain auditable.
"""

from pathlib import Path
import hashlib
import json
import uuid
import numpy as np
from rdkit import Chem
from scipy.spatial.distance import pdist
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from .settings import P
from .references import SuccessfulExamplesManager
from .prompts import calculate_target_profiles, enrich_target_profiles

COLS = ["S1", "EST", "f", "SOCME/EST"]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temp.replace(path)


def atomic_csv(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + "." + uuid.uuid4().hex + ".tmp")
    data.to_csv(temp, index=False)
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hard(y, threshold=None):
    t = threshold or P["constraints"]
    y = np.asarray(y)
    return (y[..., 0] >= t["S1_low"]) & (y[..., 0] <= t["S1_high"]) & (y[..., 1] < t["EST_max"])


def common_score(y):
    y = np.asarray(y)
    ok = hard(y)
    objective = 3 * np.clip(np.log1p(np.maximum(y[..., 2], 0)) / np.log1p(0.3), 0, 1) + 3 * np.clip(
        np.log1p(np.maximum(y[..., 3], 0)) / np.log1p(70), 0, 1
    )
    return 4 * ok + objective * np.where(ok, 1.0, 0.3)


def transform(y):
    y = np.asarray(y).copy()
    y[..., 2:] = np.log1p(np.maximum(y[..., 2:], 0))
    return y


def untransform(y):
    y = np.asarray(y).copy()
    y[..., 2:] = np.expm1(np.clip(y[..., 2:], -15, 15))
    return y


def front_mask(y):
    y = np.asarray(y)
    out = np.ones(len(y), dtype=bool)
    for i in range(len(y)):
        out[i] = not np.any(np.all(y >= y[i], axis=1) & np.any(y > y[i], axis=1))
    return out


def rank_candidates(ids, draws, beta=0.6):
    ids = np.asarray(ids, dtype=int)
    draws = np.asarray(draws)
    mu = draws.mean(0)
    sd = draws.std(0, ddof=0)
    scores = common_score(draws)
    ucb = scores.mean(0) + beta * scores.std(0, ddof=0)
    feasible = hard(mu)
    pool = np.flatnonzero(feasible)
    if not len(pool):
        pool = np.arange(len(ids))
    front = pool[front_mask(mu[pool, 2:4])]
    rest = np.array([i for i in pool if i not in set(front)], dtype=int)
    extra = np.array([i for i in range(len(ids)) if i not in set(pool)], dtype=int)
    sort = lambda pos: sorted(pos, key=lambda i: (-ucb[i], int(ids[i])))
    ordering = np.array(sort(front) + sort(rest) + sort(extra), dtype=int)
    return ids[ordering].tolist(), mu, sd, ucb


def fit_predict(kind, lib, history, ids, seed):
    """Only history carries targets; label-free Library and requested ids."""
    train_ids = history.id.to_numpy(int)
    x = lib.x[train_ids]
    test = lib.x[ids]
    y = transform(history[COLS].to_numpy(float))
    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=96, max_features=0.7, min_samples_leaf=2, n_jobs=2, random_state=seed
        )
        model.fit(lib.bits[train_ids], y)
        pred = np.array([tree.predict(lib.bits[ids]) for tree in model.estimators_])
        return untransform(pred), {
            "kind": kind,
            "ensemble_members": len(pred),
            "uncertainty": "tree dispersion",
        }
    if kind == "xgb":
        from xgboost import XGBRegressor

        predictions = []
        for trial in range(3):
            rng = np.random.default_rng(seed * 17 + trial)
            boot = rng.choice(len(y), len(y), replace=True)
            model = MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.07,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    n_jobs=2,
                    random_state=seed + trial,
                    verbosity=0,
                )
            )
            model.fit(lib.bits[train_ids[boot]], y[boot])
            predictions.append(model.predict(lib.bits[ids]))
        return untransform(np.array(predictions)), {
            "kind": kind,
            "ensemble_members": 3,
            "uncertainty": "bootstrap dispersion",
        }
    if kind == "gp":
        scale = np.maximum(y.std(0), 1e-6)
        center = y.mean(0)
        length = max(float(np.median(pdist(x))), 1e-3)
        model = GaussianProcessRegressor(
            kernel=Matern(length_scale=length, nu=2.5) + WhiteKernel(noise_level=0.05),
            optimizer=None,
            alpha=1e-6,
        )
        model.fit(x, (y - center) / scale)
        mean, std = model.predict(test, return_std=True)
        rng = np.random.default_rng(seed)
        draws = mean[None] + rng.standard_normal((32, len(ids), 4)) * std[None]
        return untransform(draws * scale + center), {
            "kind": kind,
            "posterior_draws": 32,
            "kernel_length": length,
            "uncertainty": "marginal GP standard deviation; independent ranking draws",
        }
    if kind == "graph":
        from .graph_model import graph_predict

        return graph_predict(lib, history, ids, seed)
    raise ValueError(kind)


def observed_anchors(history):
    y = history[COLS].to_numpy(float)
    score = common_score(y)
    feasible = np.flatnonzero(hard(y))
    pool = feasible if len(feasible) else np.arange(len(y))
    a = pool[np.argmax(y[pool, 2])]
    b = pool[np.argmax(y[pool, 3])]
    c = int(np.argmax(score))
    return [str(history.iloc[i].SMILES) for i in [a, b, c]]


def mutate(smiles, rng):
    """Small valence-checked terminal substitution / aromatic atom swap."""
    mol = Chem.MolFromSmiles(smiles)
    variants = []
    for a in mol.GetAtoms():
        if a.GetDegree() == 1 and a.GetAtomicNum() in [6, 7, 8, 9, 17, 35]:
            for elem in [6, 7, 8, 9, 17, 35]:
                if elem == a.GetAtomicNum():
                    continue
                rw = Chem.RWMol(mol)
                rw.GetAtomWithIdx(a.GetIdx()).SetAtomicNum(elem)
                try:
                    Chem.SanitizeMol(rw)
                    s = Chem.MolToSmiles(rw)
                    if s != smiles:
                        variants.append(s)
                except Exception:
                    pass
        if a.GetIsAromatic() and a.GetDegree() == 2 and a.GetAtomicNum() in [6, 7]:
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(a.GetIdx()).SetAtomicNum(7 if a.GetAtomicNum() == 6 else 6)
            try:
                Chem.SanitizeMol(rw)
                variants.append(Chem.MolToSmiles(rw))
            except Exception:
                pass
    variants = sorted(set(variants))
    return variants[int(rng.integers(len(variants)))] if variants else smiles


def generator_profiles(history, seed):
    manager = SuccessfulExamplesManager()
    frontier, _, guidance = manager.get_generation_memory(history)
    profiles = enrich_target_profiles(
        calculate_target_profiles(frontier, rnd_state=seed), guidance, rnd_state=seed
    )
    return profiles, manager


def retrieve(lib, targets, remaining, rnd):
    lam = 0.5 + 0.3 * (rnd - 1) / 19
    chosen = []
    for target, k in zip(targets, [15, 15, 20]):
        available = [i for i in remaining if i not in set(chosen)]
        chosen.extend(lib.mmr(target, available, k, lam))
    if len(chosen) != 50:
        raise ValueError("Incorrect candidate budget")
    return chosen
