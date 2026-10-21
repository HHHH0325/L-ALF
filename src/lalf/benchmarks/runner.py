"""Portable, code-only manuscript comparisons on user-supplied labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lalf.config import LLMConfig, _reject_secret_fields
from lalf.llm.client import OpenAIChatBackend

from .core import (
    COLS,
    atomic_csv,
    atomic_json,
    common_score,
    digest,
    fit_predict,
    generator_profiles,
    mutate,
    observed_anchors,
    rank_candidates,
    retrieve,
)
from .data import Library, Oracle, prepare_frame, public_frame
from .registry import METHODS, needs_llm
from .search import search_candidates
from .settings import P


def choose_batch(method, lib, history, remaining, seed, round_index, client=None, device="cpu"):
    """No oracle or full label table is available to this selector."""
    if method not in METHODS:
        raise ValueError(f"Unregistered comparison: {method}")
    if history.empty:
        raise ValueError("Comparison models start after the budgeted random first batch")
    roundseed = seed * 10000 + round_index
    scope = f"{method}/seed{seed}/round{round_index}"
    if method in {"qpots", "qlognparego"}:
        from .mobo_adapter import select_mobo

        return select_mobo(method, lib, history, remaining, roundseed, device=device)
    if method.endswith("_search"):
        return search_candidates(method, lib, history, remaining, roundseed, client, scope)[3]
    profiles, _ = generator_profiles(history, roundseed)
    if method in {"similarity_gen_llm", "rule_gen_llm", "evolution_gen_llm"}:
        targets = observed_anchors(history)
        if method == "rule_gen_llm":
            targets = [
                mutate(s, np.random.default_rng(roundseed + i)) for i, s in enumerate(targets)
            ]
        elif method == "evolution_gen_llm":
            rng = np.random.default_rng(roundseed)
            variants = [mutate(s, rng) for s in targets for _ in range(6)]
            scores = common_score(history[COLS].to_numpy(float))
            ranked = []
            for smiles in sorted(set(variants)):
                weights = lib.similarity(smiles, history.id.to_numpy(int)) ** 4
                ranked.append((float(weights @ scores / max(weights.sum(), 1e-9)), smiles))
            targets = [s for _, s in sorted(ranked, reverse=True)[:3]]
            while len(targets) < 3:
                targets.append(targets[-1])
    else:
        targets = client.targets(
            profiles,
            scope + "/generator",
            no_context=method == "llm_no_context",
        )
    ids = retrieve(lib, targets, remaining, round_index)
    evaluator = method.removeprefix("llm_gen_") if method.startswith("llm_gen_") else "llm"
    if evaluator == "llm":
        draws, _ = client.evaluate(
            lib,
            history,
            ids,
            scope,
            no_context=method == "llm_no_context",
        )
    else:
        draws, _ = fit_predict(evaluator, lib, history, ids, roundseed)
    return rank_candidates(ids, draws, beta=P["ucb_beta"])[0][: P["batch_size"]]


def random_plan(size, seed, rounds):
    """Original Fig. 4 protocol: one draw, then slice into batches."""
    return np.random.default_rng(seed).choice(size, rounds * P["batch_size"], replace=False)


def run_comparison(frame, method, seed, rounds=20, output=None, backend=None, device="cpu"):
    """Return one table; runtime outputs are local and never bundled for release.

    This portable runner refactors retained source, not an exact replay of the
    historical five L-ALF trajectories. It never calls ORCA.
    """
    if method not in METHODS:
        raise ValueError(f"Unregistered comparison: {method}")
    if not 1 <= rounds <= P["rounds"]:
        raise ValueError("rounds must be between 1 and 20")
    full = prepare_frame(frame)
    budget = rounds * P["batch_size"]
    minimum = max(
        budget,
        (rounds - 1) * 20 + 50
        if rounds > 1 and method not in {"random", "qpots", "qlognparego"}
        else budget,
    )
    if len(full) < minimum:
        raise ValueError(f"Need at least {minimum} unique molecules for this run")
    oracle = Oracle(full)
    lib = Library(full[["id", "molecule_id", "SMILES"]])
    history = full.iloc[0:0].copy()
    plan = random_plan(len(full), seed, rounds) if method == "random" else None
    client = None
    out = Path(output) if output is not None else None
    if out is not None:
        # Never overwrite an existing run or silently change its configuration.
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(
            out / "config.json",
            {
                "method": method,
                "seed": seed,
                "rounds": rounds,
                "batch_size": 20,
                "protocol": P,
                "protocol_sha256": digest(Path(__file__).with_name("paper_protocol.json")),
                "structures_sha256": hashlib.sha256(
                    full[["molecule_id", "SMILES"]].to_csv(index=False).encode()
                ).hexdigest(),
            },
        )
    for rnd in range(1, rounds + 1):
        remaining = [i for i in range(len(full)) if i not in oracle.queried]
        if plan is not None:
            ids = plan[(rnd - 1) * 20 : rnd * 20].tolist()
        elif rnd == 1:
            ids = np.random.default_rng(seed).choice(len(full), 20, replace=False).tolist()
        else:
            if needs_llm(method) and client is None:
                from .client import Client

                if backend is None:
                    llm_config = LLMConfig(
                        api_key=os.environ.get("LALF_LLM_API_KEY", ""),
                        base_url=os.environ.get("LALF_LLM_BASE_URL", LLMConfig().base_url),
                        model=os.environ.get("LALF_LLM_MODEL", LLMConfig().model),
                    )
                    backend = OpenAIChatBackend(llm_config)
                client = Client(backend, os.environ.get("LALF_LLM_MODEL", "configured-backend"))
            ids = choose_batch(
                method, lib, history.copy(deep=True), remaining, seed, rnd, client, device
            )
        if len(ids) != 20 or len(set(ids)) != 20 or not set(ids).issubset(remaining):
            raise ValueError(
                "Invalid selection: require 20 distinct, previously unqueried candidates"
            )
        if out is not None:
            atomic_json(
                out / f"selection_{rnd:03d}.json",
                {
                    "molecule_ids": full.iloc[ids].molecule_id.tolist(),
                    "history_count": len(history),
                },
            )
        # The only access to new labels, after selection is fixed.
        labels = oracle.query(ids)
        labels["round"] = rnd
        labels["selection_rank"] = np.arange(1, 21)
        labels["method"] = method
        history = pd.concat([history, labels], ignore_index=True)
        if out is not None:
            atomic_csv(out / "selected_verified_all.csv", public_frame(history))
    return public_frame(history)


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Benchmark config must be a mapping")
    _reject_secret_fields(raw)
    if set(raw) - {"paths", "comparison"}:
        raise ValueError("Only paths and comparison sections are accepted")
    paths, comparison = raw.get("paths", {}), raw.get("comparison", {})
    if not isinstance(paths, dict) or not isinstance(comparison, dict):
        raise ValueError("paths and comparison must be mappings")
    if set(paths) != {"labels_csv", "runs_dir"}:
        raise ValueError("paths requires labels_csv and runs_dir")
    if any(not isinstance(value, str) or not value.strip() for value in paths.values()):
        raise ValueError("paths must contain nonempty file paths")
    if set(comparison) - {"methods", "seeds", "rounds", "device"}:
        raise ValueError("Unknown comparison field")
    methods = comparison.get("methods", list(METHODS))
    seeds = comparison.get("seeds", P["seeds"])
    rounds = comparison.get("rounds", 20)
    device = comparison.get("device", "cpu")
    if not isinstance(methods, list) or not methods or set(methods) - set(METHODS):
        raise ValueError("methods must contain only retained comparisons")
    if len(methods) != len(set(methods)):
        raise ValueError("methods must be unique")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be nonnegative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if type(rounds) is not int or not 1 <= rounds <= 20 or device not in {"cpu", "cuda", "auto"}:
        raise ValueError("Invalid rounds or device")
    resolved = {name: (path.parent / str(value)).resolve() for name, value in paths.items()}
    return resolved, dict(methods=methods, seeds=seeds, rounds=rounds, device=device)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Code-only manuscript baseline runner")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    if args.list_methods:
        print(json.dumps(METHODS, indent=2))
        return 0
    if args.config is None:
        parser.error("--config is required")
    paths, config = load_config(args.config)
    if args.check_config:
        print(json.dumps(config, indent=2))
        return 0
    frame = pd.read_csv(paths["labels_csv"], dtype={"molecule_id": str})
    for method in config["methods"]:
        for seed in config["seeds"]:
            run_comparison(
                frame,
                method,
                seed,
                config["rounds"],
                paths["runs_dir"] / method / f"seed{seed}",
                device=config["device"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
