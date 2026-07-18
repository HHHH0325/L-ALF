from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStatus.COMPLETED, JobStatus.PARTIAL, JobStatus.FAILED}


class JobHandle:
    def __init__(self, job_id: str, remote_dir: str) -> None:
        self.job_id = str(job_id)
        self.remote_dir = str(remote_dir)

    def __repr__(self) -> str:
        return f"JobHandle(job_id={self.job_id!r}, remote_dir={self.remote_dir!r})"


class HPCBackend(Protocol):
    def upload_submit(self, bundle_dir: Path, run_id: str) -> JobHandle: ...

    def poll(self, handle: JobHandle) -> JobStatus: ...

    def download(self, handle: JobHandle, local_dir: Path) -> None: ...

