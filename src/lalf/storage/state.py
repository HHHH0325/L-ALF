"""Persist round-stage checkpoints without storing scientific data or credentials."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from lalf.hpc.base import JobHandle


@dataclass(frozen=True)
class RunState:
    stage: str = "new"
    job_id: str = ""
    remote_dir: str = ""
    attempts: int = 0
    error_category: str = ""
    updated_at: str = ""


class RunStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def load(self) -> RunState:
        if not self.path.exists():
            return RunState()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        allowed = set(RunState.__dataclass_fields__)
        return RunState(**{key: raw[key] for key in allowed if key in raw})

    def save(self, state: RunState) -> RunState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        safe_state = RunState(
            stage=state.stage,
            job_id=state.job_id,
            remote_dir=state.remote_dir,
            attempts=int(state.attempts),
            error_category=state.error_category,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(asdict(safe_state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        return safe_state

    def mark_submitted(self, handle: JobHandle) -> RunState:
        previous = self.load()
        return self.save(
            RunState(
                stage="submitted",
                job_id=handle.job_id,
                remote_dir=handle.remote_dir,
                attempts=previous.attempts + 1,
            )
        )

    def mark_stage(self, stage: str) -> RunState:
        previous = self.load()
        return self.save(
            RunState(
                stage=str(stage),
                job_id=previous.job_id,
                remote_dir=previous.remote_dir,
                attempts=previous.attempts,
                error_category=previous.error_category,
            )
        )

    def mark_error(self, category: str) -> RunState:
        previous = self.load()
        return self.save(
            RunState(
                stage=previous.stage,
                job_id=previous.job_id,
                remote_dir=previous.remote_dir,
                attempts=previous.attempts,
                error_category=str(category),
            )
        )

    def mark_terminal_error(self, category: str) -> RunState:
        previous = self.load()
        return self.save(
            RunState(
                stage="failed",
                job_id=previous.job_id,
                remote_dir=previous.remote_dir,
                attempts=previous.attempts,
                error_category=str(category),
            )
        )
