from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Thresholds:
    s1_low: float = 1.3
    s1_high: float = 3.2
    est_max: float = 0.3


@dataclass(frozen=True)
class TargetProfile:
    name: str
    focus: str
    anchors: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PropertyPrediction:
    s1: float
    est: float
    oscillator_strength: float
    ratio: float
    score: float


@dataclass(frozen=True)
class PredictionStats:
    mean: PropertyPrediction
    std: PropertyPrediction
    samples: int


@dataclass(frozen=True)
class BundleManifest:
    bundle_dir: Path
    mapping_path: Path
    valid_count: int
    invalid_count: int


@dataclass(frozen=True)
class ParsedDFTResult:
    status: str
    normal_termination: bool
    s1: float | None = None
    est: float | None = None
    oscillator_strength: float | None = None
    socme: float | None = None
    ratio: float | None = None
    t_state: int | None = None
