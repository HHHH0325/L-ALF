from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_FORBIDDEN_CONFIG_KEYS = {
    "api_key",
    "base_url",
    "host",
    "key_file",
    "password",
    "private_key",
    "ssh_host",
    "ssh_user",
    "user",
    "username",
}


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a YAML mapping")
    return value


def _reject_secret_fields(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_CONFIG_KEYS:
                raise ValueError(
                    f"{path}.{key} must be provided through environment variables"
                )
            _reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{path}[{index}]")


def _resolve(base: Path, raw: Any, default: str) -> Path:
    path = Path(str(raw if raw not in (None, "") else default)).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class PathsConfig:
    pool_csv: Path
    history_csv: Path
    runs_dir: Path


@dataclass(frozen=True)
class WorkflowConfig:
    rounds: int = 1
    poll_interval_seconds: float = 30.0
    wait_timeout_seconds: float = 604800.0


@dataclass(frozen=True)
class ALConfig:
    candidates_per_profile: int = 50
    select_top_k: int = 10
    llm_trials: int = 3
    beta: float = 1.0
    mmr_lambda: float = 0.6
    similarity_threshold: float = 0.2
    max_target_retries: int = 5
    s1_low: float = 1.3
    s1_high: float = 3.2
    est_max: float = 0.3


@dataclass(frozen=True)
class DFTConfig:
    nprocs: int = 32
    maxcore_mb: int = 2000
    ground_method: str = "PBE0 D3 def2-SVP OPT FREQ RIJCOSX"
    excited_method: str = "wB97X-D3 ZORA RIJCOSX TightSCF ZORA-def2-TZVP"
    solvent: str = "CPCM(Water)"
    nroots: int = 6
    est_min: float = 0.01
    est_max: float = 0.30
    random_seed: int = 42


@dataclass(frozen=True)
class HPCConfig:
    host: str = field(default="", repr=False)
    port: int = 22
    user: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    key_file: Path | None = field(default=None, repr=False)
    remote_root: str = "~/lalf-runs"
    partition: str = ""
    account: str = ""
    walltime: str = "24:00:00"
    orca_bin: str = "orca"
    scratch_root: str = ""
    max_retries: int = 3


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.4"
    temperature: float = 0.2


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    active_learning: ALConfig = field(default_factory=ALConfig)
    dft: DFTConfig = field(default_factory=DFTConfig)
    hpc: HPCConfig = field(default_factory=HPCConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def _section_values(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {unknown}")
    return dict(raw)


def _validate_config(config: AppConfig) -> AppConfig:
    if config.workflow.rounds < 1:
        raise ValueError("workflow.rounds must be at least 1")
    if config.workflow.poll_interval_seconds <= 0:
        raise ValueError("workflow.poll_interval_seconds must be positive")
    if config.workflow.wait_timeout_seconds <= 0:
        raise ValueError("workflow.wait_timeout_seconds must be positive")
    al = config.active_learning
    for name in ["candidates_per_profile", "select_top_k", "llm_trials", "max_target_retries"]:
        if int(getattr(al, name)) < 1:
            raise ValueError(f"active_learning.{name} must be at least 1")
    if not 0.0 <= al.mmr_lambda <= 1.0:
        raise ValueError("active_learning.mmr_lambda must be between 0 and 1")
    if not 0.0 <= al.similarity_threshold <= 1.0:
        raise ValueError("active_learning.similarity_threshold must be between 0 and 1")
    if al.s1_low >= al.s1_high:
        raise ValueError("active_learning.s1_low must be less than s1_high")
    if al.est_max <= 0:
        raise ValueError("active_learning.est_max must be positive")
    if config.dft.nprocs < 1:
        raise ValueError("dft.nprocs must be at least 1")
    if config.dft.maxcore_mb < 1:
        raise ValueError("dft.maxcore_mb must be at least 1")
    if config.dft.nroots < 1:
        raise ValueError("dft.nroots must be at least 1")
    if not 0 < config.dft.est_min <= config.dft.est_max:
        raise ValueError("dft.est_min must be positive and no greater than dft.est_max")
    if not 1 <= config.hpc.port <= 65535:
        raise ValueError("hpc.port must be between 1 and 65535")
    if config.hpc.max_retries < 1:
        raise ValueError("hpc.max_retries must be at least 1")
    if not 0.0 <= config.llm.temperature <= 2.0:
        raise ValueError("llm.temperature must be between 0 and 2")
    return config


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a YAML mapping")
    _reject_secret_fields(raw)

    base = config_path.parent
    paths_raw = _mapping(raw.get("paths"), "paths")
    paths = PathsConfig(
        pool_csv=_resolve(base, paths_raw.get("pool_csv"), "data/pool.csv"),
        history_csv=_resolve(
            base, paths_raw.get("history_csv"), "data/history.csv"
        ),
        runs_dir=_resolve(base, paths_raw.get("runs_dir"), "runs"),
    )

    workflow = WorkflowConfig(
        **_section_values(WorkflowConfig, _mapping(raw.get("workflow"), "workflow"))
    )
    active_learning = ALConfig(
        **_section_values(
            ALConfig, _mapping(raw.get("active_learning"), "active_learning")
        )
    )
    dft = DFTConfig(**_section_values(DFTConfig, _mapping(raw.get("dft"), "dft")))
    hpc_values = _section_values(HPCConfig, _mapping(raw.get("hpc"), "hpc"))
    configured_port = hpc_values.pop("port", 22)
    key_raw = os.getenv("LALF_SSH_KEY_FILE", "").strip()
    hpc = HPCConfig(
        **hpc_values,
        host=os.getenv("LALF_SSH_HOST", "").strip(),
        port=int(os.getenv("LALF_SSH_PORT", configured_port)),
        user=os.getenv("LALF_SSH_USER", "").strip(),
        password=os.getenv("LALF_SSH_PASSWORD", ""),
        key_file=Path(key_raw).expanduser().resolve() if key_raw else None,
    )
    llm_values = _section_values(LLMConfig, _mapping(raw.get("llm"), "llm"))
    configured_base_url = llm_values.pop("base_url", LLMConfig.base_url)
    configured_model = llm_values.pop("model", LLMConfig.model)
    llm = LLMConfig(
        **llm_values,
        api_key=os.getenv("LALF_LLM_API_KEY", ""),
        base_url=os.getenv("LALF_LLM_BASE_URL", configured_base_url),
        model=os.getenv("LALF_LLM_MODEL", configured_model),
    )
    return _validate_config(
        AppConfig(
            paths=paths,
            workflow=workflow,
            active_learning=active_learning,
            dft=dft,
            hpc=hpc,
            llm=llm,
        )
    )
