"""Literature-based, finite-library adaptations. Only queried history has labels.

These are controlled adaptations, not reproductions of the original benchmarks.
The third-party Graph-GA callable operators and MIT license are preserved.
Upstream __main__ demonstrations are omitted; see vendor source records.
"""

import contextlib
import hashlib
import importlib.util
import json
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from rdkit import Chem, DataStructs
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

from pathlib import Path
from .core import COLS, common_score, transform, untransform

_FEATURE_LOCK = threading.Lock()
_OPERATOR_LOCK = threading.Lock()
_WL = {}
_OPERATORS = None


def wl_counts(smiles, depth=3):
    """Bond-labeled WL subtree counts; ignores atom indexing and stereochemistry."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid structure in WL features")
    labels = [
        str((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsAromatic(), a.GetTotalNumHs()))
        for a in mol.GetAtoms()
    ]
    counts = Counter("0:" + x for x in labels)
    for iteration in range(1, depth + 1):
        refined = []
        for atom in mol.GetAtoms():
            neighbors = sorted(
                (str(b.GetBondType()), labels[b.GetOtherAtomIdx(atom.GetIdx())])
                for b in atom.GetBonds()
            )
            token = json.dumps([labels[atom.GetIdx()], neighbors], separators=(",", ":"))
            refined.append(hashlib.sha256(token.encode()).hexdigest())
        labels = refined
        counts.update(str(iteration) + ":" + x for x in labels)
    return dict(counts)


def wl_features(lib):
    key = hashlib.sha256(("wl-bond-h3-v1\n" + "\n".join(lib.smiles)).encode()).hexdigest()
    with _FEATURE_LOCK:
        if key not in _WL:
            features = DictVectorizer(dtype=np.float64).fit_transform(
                [wl_counts(s) for s in lib.smiles]
            )
            features = normalize(features, norm="l2", axis=1).tocsr()
            _WL[key] = features
        return _WL[key]


def graph_gp_predict(lib, history, ids, seed):
    features = wl_features(lib)
    train = features[history.id.to_numpy(int)]
    y = transform(history[COLS].to_numpy(float))
    center = y.mean(0)
    scale = np.maximum(y.std(0), 1e-6)
    kernel = (train @ train.T).toarray()
    # Fixed parameters: no full-dataset targets used for fitting or tuning.
    factor = cho_factor(kernel + (0.05 + 1e-6) * np.eye(len(y)), lower=True)
    alpha = cho_solve(factor, (y - center) / scale)
    mean, std = [], []
    for start in range(0, len(ids), 512):
        cross = (features[np.asarray(ids[start : start + 512], dtype=int)] @ train.T).toarray()
        mean.append(cross @ alpha)
        solved = solve_triangular(factor[0], cross.T, lower=True)
        std.append(np.sqrt(np.maximum(1 - (solved * solved).sum(0), 1e-12)))
    mean, std = np.concatenate(mean), np.concatenate(std)
    rng = np.random.default_rng(seed)
    draws = mean[None] + rng.standard_normal((32, len(ids), 4)) * std[None, :, None]
    return untransform(draws * scale + center), {
        "kind": "graph_gp",
        "kernel": "L2-normalized bond-labeled WL subtree dot product",
        "wl_depth": 3,
        "observation_noise_variance_standardized": 0.05,
        "jitter": 1e-6,
        "optimizer": None,
        "posterior_draws": 32,
        "uncertainty": "latent marginal GP SD; independent marginal ranking draws",
        "citations": ["shervashidze2011wl", "griffiths2023gauche"],
        "implementation": "native WL/GP; not the GAUCHE software or a published BO benchmark reproduction",
        "representation_limit": "atom/bond graph; stereochemistry omitted",
    }


def graph_ga_operators():
    global _OPERATORS
    if _OPERATORS is None:
        source = Path(__file__).parent / "vendor" / "graph_ga"
        modules = []
        for name in ["crossover", "mutate"]:
            spec = importlib.util.spec_from_file_location(
                "lalf_graphga_" + name, source / (name + ".py")
            )
            module = importlib.util.module_from_spec(spec)
            previous = sys.modules.get("crossover")
            if modules:
                sys.modules["crossover"] = modules[0]
            try:
                spec.loader.exec_module(module)
            finally:
                if previous is None:
                    sys.modules.pop("crossover", None)
                else:
                    sys.modules["crossover"] = previous
            modules.append(module)
        _OPERATORS = tuple(modules)
    return _OPERATORS


@contextlib.contextmanager
def seeded_operators(seed, parent_smiles):
    # Upstream uses module-level random generators. Serialize and restore both.
    with _OPERATOR_LOCK:
        co, mutation = graph_ga_operators()
        py_state, np_state = random.getstate(), np.random.get_state()
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32))
        sizes = np.array([Chem.MolFromSmiles(s).GetNumAtoms() for s in parent_smiles])
        co.average_size = float(sizes.mean())
        co.size_stdev = float(max(sizes.std(), 1))
        try:
            yield co, mutation
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def population(history):
    scores = common_score(history[COLS].to_numpy(float))
    elite = np.argsort(-scores, kind="stable")[: min(100, len(history))]
    parents = history.iloc[elite].reset_index(drop=True)
    fitness = scores[elite]
    # A tiny floor makes a zero-score startup population sampleable.
    probability = (fitness + 1e-8) / (fitness + 1e-8).sum()
    return parents, fitness, probability


def parent_pairs(history, seed, n=60):
    parents, fitness, probability = population(history)
    rng = np.random.default_rng(seed)
    indices = [rng.choice(len(parents), 2, replace=False, p=probability) for _ in range(n)]
    pairs = []
    for a, b in indices:
        pairs.append(
            [{"SMILES": str(parents.iloc[i].SMILES), "fitness": float(fitness[i])} for i in [a, b]]
        )
    return parents, pairs


def llm_crossovers(client, pairs, scope):
    """MolLEO's two-parent prompt, batched in 10 independent tagged tasks.

    Invalid chemical proposals use the paper's Graph-GA fallback. HTTP or
    malformed-JSON failures are errors and cannot silently become GA results.
    """
    from .client import APIError

    objective = (
        "Maximize observed fitness: 4*H + (3*clip(log(1+max(f,0))/log(1.3),0,1) "
        "+3*clip(log(1+max(SOCME/EST,0))/log(71),0,1))*(1 if H else 0.3). "
        "H=1 when 1.3<=S1<=3.2 eV and EST<0.3 eV, with no additional EST lower bound; otherwise H=0. "
        "EST is the processed absolute gap |E(S1)-E(Tn)|: maximize SOCME/gap over "
        "available SOC channels in 0.01<=gap<=0.30 eV; if none exist, use the nearest "
        "channel with SOC and floor its gap at 0.01 eV. SOCME/EST uses that same gap. "
        "The two desirable descriptors are oscillator strength f and SOCME/EST. "
        "Strict reference targets: f>0.3 and SOCME/EST>70. "
    )
    tasks = [{"id": i, "parents": p} for i, p in enumerate(pairs)]
    chunks = [tasks[i : i + 10] for i in range(0, len(tasks), 10)]

    def request(item):
        chunk_no, chunk = item
        prompt = (
            objective + "\nFor EACH independent task, two molecules and their observed fitness "
            "scores are supplied. Propose one new molecule with a higher fitness by making "
            "crossover and mutations based on those two parents. Return JSON array "
            '[{"id":integer,"smiles":string}], all requested IDs exactly once. '
            "Do not supply predicted scores.\nTASKS_JSON=" + json.dumps(chunk)
        )
        for attempt in range(3):
            content = client.call(
                prompt, 0.2, scope + f"/crossovers/chunk{chunk_no}/attempt{attempt}", 8000
            )
            try:
                output = json.loads(content[content.find("[") : content.rfind("]") + 1])
                wanted = {t["id"] for t in chunk}
                got = [int(row["id"]) for row in output]
                if len(got) != len(set(got)) or set(got) != wanted:
                    raise ValueError("Missing/duplicate/unrequested crossover ID")
                return {int(row["id"]): row.get("smiles") for row in output}
            except (TypeError, ValueError, KeyError):
                continue
        raise APIError("MolLEO crossover JSON failed: " + scope)

    result = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        for part in executor.map(request, enumerate(chunks)):
            result.update(part)
    return result


def canonical_mol(value):
    if not isinstance(value, str):
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    return mol


def evolutionary_search(method, lib, history, remaining, seed, client=None, scope=None):
    parents, pairs = parent_pairs(history, seed)
    raw = llm_crossovers(client, pairs, scope) if method == "molleo_search" else {}
    children = []
    invalid_llm = 0
    failed_graph = 0
    mutation_failures = 0
    operator_errors = 0
    with seeded_operators(seed, parents.SMILES):
        co, mutation = graph_ga_operators()
        for i, pair in enumerate(pairs):
            child = canonical_mol(raw.get(i)) if method == "molleo_search" else None
            if child is not None and Chem.MolToSmiles(child) in [p["SMILES"] for p in pair]:
                child = None
            if method == "molleo_search" and child is None:
                invalid_llm += 1
            if child is None:
                try:
                    child = co.crossover(
                        Chem.MolFromSmiles(pair[0]["SMILES"]), Chem.MolFromSmiles(pair[1]["SMILES"])
                    )
                except (IndexError, ValueError, KeyError, RuntimeError):
                    # The archived operator indexes empty RDKit reaction outputs for
                    # some structures. Record the failure without editing its source.
                    operator_errors += 1
                    child = None
            if child is None:
                failed_graph += 1
                continue
            try:
                mutated = mutation.mutate(Chem.Mol(child), 0.067)
            except (IndexError, ValueError, KeyError, RuntimeError):
                operator_errors += 1
                mutated = None
            if mutated is None:
                mutation_failures += 1
                continue
            smiles = Chem.MolToSmiles(mutated)
            if smiles not in children:
                children.append(smiles)
    if not children:
        raise RuntimeError(
            "No valid evolutionary offspring; cannot silently replace method with random search"
        )
    best = str(parents.iloc[0].SMILES)
    if method == "molleo_search":
        fp = lib.gen.GetFingerprint(Chem.MolFromSmiles(best))
        relevance = [
            DataStructs.TanimotoSimilarity(fp, lib.gen.GetFingerprint(Chem.MolFromSmiles(s)))
            for s in children
        ]
        children = [children[i] for i in np.argsort(-np.asarray(relevance), kind="stable")]
    # Nearest unused finite-library structure for each child. Continue to the
    # second-nearest unused structure on duplicate mappings. No true fitness.
    available = np.asarray(remaining, dtype=int)
    candidate = []
    mappings = []
    for smiles in children:
        sims = lib.similarity(smiles, available)
        pos = next(
            j for j in np.argsort(-sims, kind="stable") if int(available[j]) not in candidate
        )
        idx = int(available[pos])
        candidate.append(idx)
        mappings.append(
            {"generated_smiles": smiles, "database_id": idx, "tanimoto": float(sims[pos])}
        )
        if len(candidate) == 50:
            break
    if len(candidate) < 50:
        # Explicit structure-only filling, with the observed best parent as anchor.
        sims = lib.similarity(best, available)
        for pos in np.argsort(-sims, kind="stable"):
            idx = int(available[pos])
            if idx not in candidate:
                candidate.append(idx)
                if len(candidate) == 50:
                    break
    if method == "molleo_search":
        values = lib.similarity(best, candidate)
        selected = np.asarray(candidate)[np.argsort(-values, kind="stable")[:20]].tolist()
    else:
        selected = np.random.default_rng(seed + 7919).choice(candidate, 20, replace=False).tolist()
    metadata = {
        "kind": method,
        "citations": ["jensen2019graphga"] + (["wang2025molleo"] if raw else []),
        "population": "top 100 observed fitness; fitness-proportional distinct parent sampling",
        "pairs": pairs,
        "offspring_attempts": len(pairs),
        "mutation_probability": 0.067,
        "valid_unique_offspring": len(children),
        "invalid_llm_fallbacks": invalid_llm,
        "failed_graph_crossovers": failed_graph,
        "failed_mutations": mutation_failures,
        "upstream_operator_exceptions": operator_errors,
        "mappings": mappings,
        "structure_only_fillers": 50 - len(mappings),
        "selected_by": "best-observed-parent Tanimoto pressure"
        if raw
        else "uniform offspring batch",
        "adaptation": "finite-library mapping, 50 proposals/20 queried per round, common scalar fitness, 100-parent elite survival",
        "model": client.model if raw else None,
        "llm_batching": "10 independent two-parent tasks per API response" if raw else None,
        "not_exact_reproduction": True,
    }
    return candidate, None, metadata, selected
