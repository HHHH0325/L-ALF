from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lalf.config import load_config
from lalf.hpc.ssh_slurm import SSHSlurmBackend
from lalf.llm.client import OpenAIChatBackend
from lalf.storage.state import RunStateStore
from lalf.workflow import run_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lalf", description="Run the sanitized L-ALF active-learning/DFT workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["run", "status", "check-config"]:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)
    return parser


def _status(config_path: Path) -> int:
    config = load_config(config_path)
    rows = []
    for round_index in range(1, config.workflow.rounds + 1):
        round_id = f"round_{round_index:03d}"
        state = RunStateStore(config.paths.runs_dir / round_id / "state.json").load()
        rows.append(
            {
                "round": round_id,
                "stage": state.stage,
                "job_id": state.job_id,
                "attempts": state.attempts,
                "error_category": state.error_category,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "check-config":
            print(
                json.dumps(
                    {
                        "pool_csv": str(config.paths.pool_csv),
                        "history_csv": str(config.paths.history_csv),
                        "runs_dir": str(config.paths.runs_dir),
                        "rounds": config.workflow.rounds,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "status":
            return _status(args.config)
        summary = run_workflow(
            config,
            OpenAIChatBackend(config.llm),
            SSHSlurmBackend(config.hpc),
        )
        print(
            f"Completed {summary.completed_rounds}/{summary.total_rounds} rounds in "
            f"{summary.runs_dir}"
        )
        return 0
    except Exception as exc:
        print(f"lalf: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
