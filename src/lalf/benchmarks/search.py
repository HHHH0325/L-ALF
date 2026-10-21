"""Whole-search baselines; selectors see structures and queried history only."""

import numpy as np
from .core import COLS, common_score, observed_anchors, fit_predict, rank_candidates


def search_candidates(method, lib, history, remaining, seed, client=None, scope=None):
    rng = np.random.default_rng(seed)
    anchors = observed_anchors(history)
    if method == "similarity_search":
        rel = np.stack([lib.similarity(s, remaining) for s in anchors])
        score = rel.max(0)
        ranked = np.array(remaining)[np.argsort(-score, kind="stable")]
        return (
            ranked[:50].tolist(),
            np.stack([np.zeros((50, 4))] * 3),
            {"mode": "nearest observed objective anchors"},
            ranked[:20].tolist(),
        )
    if method == "evolutionary_search":
        # Genuine genotype operations on Morgan fingerprints, mapped to valid finite-library structures.
        # Each evaluated generation has only queried parents; intermediate offspring fitness is an
        # observed-label interpolation proxy, not an extra oracle evaluation.
        train_scores = common_score(history[COLS].to_numpy(float))
        hist_ids = history.id.to_numpy(int)
        parent_order = np.argsort(-train_scores, kind="stable")[: min(40, len(history))]
        parents = hist_ids[parent_order]
        parent_scores = train_scores[parent_order]
        probability = np.exp((parent_scores - parent_scores.max()) / 2)
        probability /= probability.sum()
        available = np.asarray(remaining)
        bits = lib.bits[available]
        bit_counts = bits.sum(1)
        pool = []
        for child in range(15):
            a, b = rng.choice(len(parents), 2, replace=False, p=probability)
            mask = rng.random(2048) < 0.5
            genotype = np.where(mask, lib.bits[parents[a]], lib.bits[parents[b]]).copy()
            flips = rng.choice(2048, 4, replace=False)
            genotype[flips] = 1 - genotype[flips]
            intersection = bits @ genotype
            similarity = intersection / np.maximum(bit_counts + genotype.sum() - intersection, 1)
            chosen = np.argsort(-similarity, kind="stable")
            additions = [int(available[i]) for i in chosen if int(available[i]) not in set(pool)][
                :3
            ]
            pool.extend(additions)
        pool += rng.choice([i for i in remaining if i not in set(pool)], 5, replace=False).tolist()
        ids = pool
        weights = np.stack([lib.similarity(lib.smiles[i], hist_ids) for i in ids]) ** 4
        fitness = (weights @ train_scores) / np.maximum(weights.sum(1), 1e-9)
        order = np.argsort(-fitness, kind="stable")
        chosen = np.asarray(ids)[order[:20]].tolist()
        return (
            ids,
            np.zeros((3, 50, 4)),
            {
                "mode": "discrete-library GA: queried-history elite parents, uniform Morgan-bit crossover, four-bit mutation, three-neighbor genotype mapping per child, five random immigrants, similarity-interpolated observed fitness",
                "version": 2,
            },
            chosen,
        )
    if method in ["graphga_search", "molleo_search"]:
        from .literature import evolutionary_search

        return evolutionary_search(method, lib, history, remaining, seed, client, scope)
    kind = method.removesuffix("_search")
    if kind == "graph_gp":
        from .literature import graph_gp_predict

        draws, meta = graph_gp_predict(lib, history, remaining, seed)
    else:
        draws, meta = fit_predict(kind, lib, history, remaining, seed)
    # All available structures may be screened by external surrogate search.
    scores = common_score(draws)
    ucb = scores.mean(0) + 0.6 * scores.std(0)
    prescreen = np.argsort(-ucb, kind="stable")[:500]
    ids = np.asarray(remaining)[prescreen].tolist()
    panel = draws[:, prescreen]
    ordered, *_ = rank_candidates(ids, panel)
    selected = ordered[:20]
    # Log all candidate predictions, not only selected or prescreened candidates.
    return remaining, draws, dict(meta, prescreen_size=500), selected
