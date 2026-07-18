"""Resumable orchestration of the complete L-ALF discovery loop.

Each round advances through five scientific stages:

1. derive three target profiles from the verified feasible frontier;
2. retrieve diverse real-library candidates and predict their properties;
3. select a small Pareto-UCB batch and prepare its DFT oracle inputs;
4. collect and parse the high-cost labels;
5. update history and recompute the frontier used by the next round.

The persisted stage is an execution detail that protects this scientific loop
from interruption without changing its selection logic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lalf.active_learning.profiles import build_profiles
from lalf.active_learning.prompts import build_design_prompt, build_prediction_prompt
from lalf.active_learning.selection import pareto_layers, select_pareto_ucb
from lalf.chemistry.fingerprints import (
    expand_candidates,
    local_references,
    valid_smiles,
)
from lalf.config import AppConfig
from lalf.dft.builder import build_orca_bundle
from lalf.dft.parser import parse_td_output
from lalf.hpc.base import HPCBackend, JobHandle, JobStatus
from lalf.llm.client import LLMBackend, generate_target_smiles, predict_properties
from lalf.models import Thresholds
from lalf.storage.history import load_history, merge_history_atomic
from lalf.storage.state import RunStateStore


@dataclass(frozen=True)
class RunSummary:
    completed_rounds: int
    total_rounds: int
    runs_dir: Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_pool(path: Path) -> pd.DataFrame:
    pool = pd.read_csv(path)
    required = {"molecule_id", "smiles"}
    missing = sorted(required - set(pool.columns))
    if missing:
        raise ValueError(f"Candidate pool is missing columns: {missing}")
    pool = pool[["molecule_id", "smiles"]].copy()
    pool["molecule_id"] = pool["molecule_id"].astype(str).str.strip()
    pool["smiles"] = pool["smiles"].astype(str).str.strip()
    conflicts = pool.groupby("molecule_id")["smiles"].nunique(dropna=False)
    bad = conflicts.loc[conflicts > 1].index.astype(str).tolist()
    if bad:
        raise ValueError(
            f"Candidate pool maps molecule IDs to different SMILES: {bad[:5]}"
        )
    return pool.drop_duplicates("molecule_id", keep="first").reset_index(drop=True)


def _fallback_target(
    profile_anchors: tuple[dict[str, object], ...],
    pool: pd.DataFrame,
    excluded_ids: set[str],
) -> str:
    for anchor in profile_anchors:
        smiles = str(anchor.get("smiles", "")).strip()
        if smiles and valid_smiles(smiles):
            return smiles
    available = pool.loc[~pool["molecule_id"].isin(excluded_ids)]
    for smiles in available["smiles"].astype(str):
        if valid_smiles(smiles):
            return smiles
    return ""


def _prepare_round(
    config: AppConfig,
    round_dir: Path,
    llm_backend: LLMBackend,
) -> None:
    """Perform target design, diversified expansion, prediction, and selection."""
    history = load_history(config.paths.history_csv)
    pool = _load_pool(config.paths.pool_csv)
    excluded_ids = set(history["molecule_id"].astype(str))
    thresholds = Thresholds(
        s1_low=config.active_learning.s1_low,
        s1_high=config.active_learning.s1_high,
        est_max=config.active_learning.est_max,
    )
    profiles = build_profiles(history, thresholds)
    candidate_rows: list[dict[str, object]] = []

    for profile in profiles:
        target = generate_target_smiles(
            llm_backend,
            build_design_prompt(profile),
            max_retries=config.active_learning.max_target_retries,
            validator=valid_smiles,
            temperature=config.llm.temperature,
        )
        if not target:
            target = _fallback_target(profile.anchors, pool, excluded_ids)
        if not target:
            continue
        expanded = expand_candidates(
            pool,
            target,
            n_pick=config.active_learning.candidates_per_profile,
            lambda_value=config.active_learning.mmr_lambda,
            excluded_ids=excluded_ids,
            similarity_threshold=config.active_learning.similarity_threshold,
        )
        for _, row in expanded.iterrows():
            record = row.to_dict()
            record["source_profile"] = profile.name
            record["target_smiles"] = target
            candidate_rows.append(record)

    if not candidate_rows:
        raise RuntimeError("No candidates remained after target expansion and history exclusion")
    candidates = pd.DataFrame(candidate_rows)
    source_profiles = (
        candidates.groupby("molecule_id")["source_profile"]
        .apply(lambda values: "|".join(sorted(set(map(str, values)))))
        .to_dict()
    )
    candidates = candidates.sort_values(
        ["similarity", "mmr_score"], ascending=False
    ).drop_duplicates("molecule_id", keep="first")
    candidates["source_profiles"] = candidates["molecule_id"].map(source_profiles)
    candidates.to_csv(round_dir / "expanded_candidates.csv", index=False)

    frontier_context = []
    for profile in profiles:
        frontier_context.extend(profile.anchors)
    prediction_rows = []
    for _, candidate in candidates.iterrows():
        local_positive, local_negative = local_references(
            history, str(candidate["smiles"]), thresholds
        )
        stats = predict_properties(
            llm_backend,
            build_prediction_prompt(
                str(candidate["smiles"]),
                global_frontier=frontier_context,
                local_positive=local_positive,
                local_negative=local_negative,
            ),
            trials=config.active_learning.llm_trials,
            temperature=config.llm.temperature,
        )
        prediction_rows.append(
            {
                "molecule_id": str(candidate["molecule_id"]),
                "smiles": str(candidate["smiles"]),
                "source_profiles": candidate["source_profiles"],
                "mean_s1": stats.mean.s1,
                "mean_est": stats.mean.est,
                "mean_f": stats.mean.oscillator_strength,
                "mean_ratio": stats.mean.ratio,
                "mean_score": stats.mean.score,
                "std_s1": stats.std.s1,
                "std_est": stats.std.est,
                "std_f": stats.std.oscillator_strength,
                "std_ratio": stats.std.ratio,
                "std_score": stats.std.score,
                "prediction_samples": stats.samples,
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(round_dir / "predictions.csv", index=False)
    selected = select_pareto_ucb(
        predictions,
        top_k=config.active_learning.select_top_k,
        beta=config.active_learning.beta,
        thresholds=thresholds,
    )
    if selected.empty:
        raise RuntimeError("Pareto-UCB did not select any candidates")
    selected.to_csv(round_dir / "selected.csv", index=False)
    manifest = build_orca_bundle(selected, round_dir / "bundle", config.dft)
    if manifest.valid_count == 0:
        raise RuntimeError("All selected molecules failed 3D conformer generation")
    _write_json(
        round_dir / "prepare_summary.json",
        {
            "candidate_count": int(len(candidates)),
            "selected_count": int(len(selected)),
            "bundle_valid_count": manifest.valid_count,
            "bundle_invalid_count": manifest.invalid_count,
        },
    )


def _wait_for_job(
    config: AppConfig,
    backend: HPCBackend,
    handle: JobHandle,
) -> JobStatus:
    started = time.monotonic()
    while True:
        status = backend.poll(handle)
        if status.is_terminal:
            return status
        if time.monotonic() - started > config.workflow.wait_timeout_seconds:
            raise TimeoutError(f"Slurm job {handle.job_id} exceeded the configured timeout")
        time.sleep(max(0.01, float(config.workflow.poll_interval_seconds)))


def _collect_round(
    config: AppConfig,
    round_dir: Path,
    backend: HPCBackend,
    handle: JobHandle,
) -> int:
    """Parse returned oracle labels and rebuild the verified Pareto frontier."""
    results_dir = round_dir / "downloaded"
    backend.download(handle, results_dir)
    mapping_path = results_dir / "mapping.csv"
    if not mapping_path.exists():
        mapping_path = round_dir / "bundle" / "mapping.csv"
    mapping = pd.read_csv(mapping_path)
    valid = mapping["is_valid"].astype(str).str.lower().isin({"true", "1", "yes"})
    parsed_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for _, row in mapping.loc[valid].iterrows():
        file_number = int(row["file_number"])
        output_path = results_dir / f"{file_number}_td.out"
        if not output_path.exists():
            failures.append(
                {
                    "file_number": file_number,
                    "molecule_id": row["molecule_id"],
                    "status": "missing_output",
                }
            )
            continue
        parsed = parse_td_output(
            output_path.read_text(encoding="utf-8", errors="replace"),
            est_min=config.dft.est_min,
            est_max=config.dft.est_max,
        )
        if parsed.status != "complete":
            failures.append(
                {
                    "file_number": file_number,
                    "molecule_id": row["molecule_id"],
                    "status": parsed.status,
                }
            )
            continue
        parsed_rows.append(
            {
                "molecule_id": str(row["molecule_id"]),
                "smiles": str(row["smiles"]),
                "s1": parsed.s1,
                "est": parsed.est,
                "f": parsed.oscillator_strength,
                "socme": parsed.socme,
                "ratio": parsed.ratio,
                "t_state": parsed.t_state,
            }
        )

    pd.DataFrame(failures).to_csv(round_dir / "failed_results.csv", index=False)
    parsed_df = pd.DataFrame(parsed_rows)
    if parsed_df.empty:
        raise RuntimeError("No complete ORCA results were available for history update")
    parsed_df.to_csv(round_dir / "parsed_results.csv", index=False)
    updated_history = merge_history_atomic(config.paths.history_csv, parsed_df)
    feasible = updated_history.loc[
        pd.to_numeric(updated_history["s1"], errors="coerce").between(
            config.active_learning.s1_low, config.active_learning.s1_high
        )
        & (
            pd.to_numeric(updated_history["est"], errors="coerce")
            < config.active_learning.est_max
        )
    ].copy()
    if feasible.empty:
        feasible.to_csv(round_dir / "pareto_front.csv", index=False)
    else:
        layer_input = feasible.rename(
            columns={"f": "mean_f", "ratio": "mean_ratio"}
        )
        feasible["pareto_layer"] = pareto_layers(layer_input).to_numpy()
        feasible.loc[feasible["pareto_layer"] == 0].to_csv(
            round_dir / "pareto_front.csv", index=False
        )
    return len(parsed_df)


def run_workflow(
    config: AppConfig,
    llm_backend: LLMBackend,
    hpc_backend: HPCBackend,
) -> RunSummary:
    """Run all configured active-learning rounds, resuming safe stage boundaries."""
    config.paths.runs_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for round_index in range(1, int(config.workflow.rounds) + 1):
        round_id = f"round_{round_index:03d}"
        round_dir = config.paths.runs_dir / round_id
        round_dir.mkdir(parents=True, exist_ok=True)
        store = RunStateStore(round_dir / "state.json")
        state = store.load()
        if state.stage == "complete":
            completed += 1
            continue
        if state.stage == "failed":
            raise RuntimeError(
                f"{round_id} is marked failed: {state.error_category or 'unknown'}"
            )

        try:
            if state.stage == "new":
                _prepare_round(config, round_dir, llm_backend)
                state = store.mark_stage("prepared")

            if state.stage == "prepared":
                handle = hpc_backend.upload_submit(round_dir / "bundle", round_id)
                state = store.mark_submitted(handle)

            if state.stage == "submitted":
                handle = JobHandle(state.job_id, state.remote_dir)
                terminal = _wait_for_job(config, hpc_backend, handle)
                if terminal == JobStatus.FAILED:
                    store.mark_terminal_error("slurm_failed")
                    raise RuntimeError(f"Slurm job {handle.job_id} failed")
                state = store.mark_stage("ready_to_collect")

            if state.stage == "ready_to_collect":
                handle = JobHandle(state.job_id, state.remote_dir)
                parsed_count = _collect_round(config, round_dir, hpc_backend, handle)
                _write_json(
                    round_dir / "collect_summary.json",
                    {"parsed_count": parsed_count, "history_csv": str(config.paths.history_csv)},
                )
                store.mark_stage("complete")

            completed += 1
        except Exception as exc:
            current = store.load()
            if current.stage not in {"failed", "complete"}:
                store.mark_error(type(exc).__name__)
            raise

    return RunSummary(
        completed_rounds=completed,
        total_rounds=int(config.workflow.rounds),
        runs_dir=config.paths.runs_dir,
    )
