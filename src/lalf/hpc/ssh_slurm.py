"""Transport the selected DFT oracle batch and observe its calculation state.

The remote directory is keyed by a content digest, tying scheduler results to
the exact molecular mapping that produced them and avoiding cross-round label
mix-ups.
"""

from __future__ import annotations

import hashlib
import posixpath
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import paramiko

from lalf.config import HPCConfig
from lalf.hpc.base import JobHandle, JobStatus


_T = TypeVar("_T")


def bundle_digest(bundle_dir: str | Path) -> str:
    directory = Path(bundle_dir).resolve()
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def classify_job_status(
    queued_text: str,
    accounting_text: str,
    *,
    done_count: int = 0,
    failed_count: int = 0,
    expected_count: int = 0,
) -> JobStatus:
    queued = queued_text.upper()
    if any(state in queued for state in ["RUNNING", "COMPLETING"]):
        return JobStatus.RUNNING
    if any(state in queued for state in ["PENDING", "CONFIGURING"]):
        return JobStatus.PENDING

    states = [
        line.split("|", 1)[0].split()[0]
        for line in accounting_text.upper().splitlines()
        if line.strip()
    ]
    failed = any(
        state.startswith(
            ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
        )
        for state in states
    )
    completed = any(state.startswith("COMPLETED") for state in states)
    if states and all(state.startswith("COMPLETED") for state in states):
        return JobStatus.COMPLETED
    if failed and completed:
        return JobStatus.PARTIAL
    if failed:
        return JobStatus.FAILED
    if failed_count and done_count:
        return JobStatus.PARTIAL
    if failed_count:
        return JobStatus.FAILED
    if expected_count > 0 and done_count >= expected_count:
        return JobStatus.COMPLETED
    return JobStatus.UNKNOWN


class SSHSlurmBackend:
    def __init__(self, config: HPCConfig) -> None:
        if not config.host or not config.user:
            raise ValueError("Set LALF_SSH_HOST and LALF_SSH_USER for SSH execution")
        self.config = config

    def __repr__(self) -> str:
        return f"SSHSlurmBackend(port={self.config.port})"

    def _connect(self) -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(
            hostname=self.config.host,
            port=int(self.config.port),
            username=self.config.user,
            password=self.config.password or None,
            key_filename=str(self.config.key_file) if self.config.key_file else None,
            allow_agent=True,
            look_for_keys=True,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
        return client

    def _retry(self, operation: Callable[[], _T]) -> _T:
        last_error: Exception | None = None
        for attempt in range(max(1, int(self.config.max_retries))):
            try:
                return operation()
            except (EOFError, OSError, paramiko.SSHException) as exc:
                last_error = exc
                if attempt + 1 >= max(1, int(self.config.max_retries)):
                    break
                time.sleep(min(2**attempt, 8))
        if last_error is not None:
            raise last_error
        raise RuntimeError("SSH operation did not execute")

    def _remote_root(self, sftp: paramiko.SFTPClient) -> str:
        raw = self.config.remote_root.strip() or "lalf-runs"
        home = sftp.normalize(".")
        if raw == "~":
            return home
        if raw.startswith("~/"):
            return posixpath.join(home, raw[2:])
        if raw.startswith("/"):
            return posixpath.normpath(raw)
        return posixpath.join(home, raw)

    @staticmethod
    def _mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
        current = "/" if remote_dir.startswith("/") else ""
        for part in [item for item in remote_dir.split("/") if item]:
            current = posixpath.join(current, part)
            try:
                sftp.stat(current)
            except OSError:
                sftp.mkdir(current)

    @staticmethod
    def _run(client: paramiko.SSHClient, command: str) -> str:
        _, stdout, stderr = client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        output = stdout.read().decode("utf-8", errors="replace").strip()
        error = stderr.read().decode("utf-8", errors="replace").strip()
        if exit_code != 0:
            raise RuntimeError(f"Remote command failed with exit code {exit_code}: {error}")
        return output

    def upload_submit(self, bundle_dir: Path, run_id: str) -> JobHandle:
        local_bundle = Path(bundle_dir).resolve()
        if not local_bundle.is_dir():
            raise FileNotFoundError(local_bundle)
        safe_run_id = Path(run_id).name
        if safe_run_id != run_id or safe_run_id in {"", ".", ".."}:
            raise ValueError("run_id must be a single safe directory name")
        remote_leaf = f"{safe_run_id}-{bundle_digest(local_bundle)[:12]}"

        def operation() -> JobHandle:
            client = self._connect()
            try:
                sftp = client.open_sftp()
                try:
                    remote_dir = posixpath.join(self._remote_root(sftp), remote_leaf)
                    self._mkdirs(sftp, remote_dir)
                    for local_file in sorted(local_bundle.iterdir()):
                        if local_file.is_file() and not local_file.is_symlink():
                            sftp.put(str(local_file), posixpath.join(remote_dir, local_file.name))
                finally:
                    sftp.close()

                options = ["--parsable", f"--time={self.config.walltime}"]
                if self.config.partition:
                    options.append(f"--partition={self.config.partition}")
                if self.config.account:
                    options.append(f"--account={self.config.account}")
                exports = "ALL,ORCA_BIN=" + self.config.orca_bin
                if self.config.scratch_root:
                    exports += ",LALF_SCRATCH_ROOT=" + self.config.scratch_root
                options.append(f"--export={exports}")
                submit = (
                    "sbatch "
                    + " ".join(shlex.quote(option) for option in options)
                    + " run_array.sh"
                )
                command = (
                    f"cd {shlex.quote(remote_dir)} && "
                    "if [ -s .lalf_job_id ]; then cat .lalf_job_id; else "
                    f"job_id=$({submit}); "
                    "tmp=.lalf_job_id.tmp.$$; printf '%s\\n' \"$job_id\" > \"$tmp\"; "
                    "mv \"$tmp\" .lalf_job_id; cat .lalf_job_id; fi"
                )
                output = self._run(client, command)
                job_id = output.split(";", 1)[0].strip()
                if not job_id:
                    raise RuntimeError("sbatch returned no job id")
                return JobHandle(job_id=job_id, remote_dir=remote_leaf)
            finally:
                client.close()

        return self._retry(operation)

    def _full_remote_dir(self, sftp: paramiko.SFTPClient, handle: JobHandle) -> str:
        return posixpath.join(self._remote_root(sftp), Path(handle.remote_dir).name)

    def poll(self, handle: JobHandle) -> JobStatus:
        def operation() -> JobStatus:
            client = self._connect()
            try:
                queued = self._run(
                    client,
                    f"squeue -h -j {shlex.quote(handle.job_id)} -o %T || true",
                )
                accounting = self._run(
                    client,
                    f"sacct -n -X -j {shlex.quote(handle.job_id)} --format=State -P || true",
                )
                preliminary = classify_job_status(queued, accounting)
                if preliminary != JobStatus.UNKNOWN:
                    return preliminary
                sftp = client.open_sftp()
                try:
                    remote_dir = self._full_remote_dir(sftp, handle)
                    names = sftp.listdir(remote_dir)
                    done_count = sum(name.endswith(".done") for name in names)
                    failed_count = sum(name.endswith(".failed") for name in names)
                    with sftp.open(posixpath.join(remote_dir, "ids.txt"), "r") as stream:
                        raw_ids = stream.read()
                        ids_text = (
                            raw_ids.decode("utf-8", errors="replace")
                            if isinstance(raw_ids, bytes)
                            else str(raw_ids)
                        )
                        expected_count = sum(
                            bool(line.strip())
                            for line in ids_text.splitlines()
                        )
                except OSError:
                    done_count = failed_count = expected_count = 0
                finally:
                    sftp.close()
                return classify_job_status(
                    queued,
                    accounting,
                    done_count=done_count,
                    failed_count=failed_count,
                    expected_count=expected_count,
                )
            finally:
                client.close()

        return self._retry(operation)

    def download(self, handle: JobHandle, local_dir: Path) -> None:
        destination = Path(local_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        allowed_names = {"mapping.csv", "ids.txt"}
        allowed_suffixes = {".out", ".xyz", ".done", ".failed"}

        def operation() -> None:
            client = self._connect()
            try:
                sftp = client.open_sftp()
                try:
                    remote_dir = self._full_remote_dir(sftp, handle)
                    for entry in sftp.listdir_attr(remote_dir):
                        name = entry.filename
                        if name not in allowed_names and Path(name).suffix not in allowed_suffixes:
                            continue
                        sftp.get(posixpath.join(remote_dir, name), str(destination / name))
                finally:
                    sftp.close()
            finally:
                client.close()

        self._retry(operation)
