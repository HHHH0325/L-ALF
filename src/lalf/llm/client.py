"""LLM boundary for directional design and repeated property evaluation.

Repeated calls are aggregated into a mean prediction and a consistency signal.
The latter is used by UCB as an exploration term, not as a Bayesian posterior.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

import numpy as np
from openai import OpenAI

from lalf.config import LLMConfig
from lalf.models import PredictionStats, PropertyPrediction


class LLMBackend(Protocol):
    def complete(self, prompt: str, temperature: float, max_tokens: int) -> str: ...


class OpenAIChatBackend:
    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise ValueError("Set LALF_LLM_API_KEY before using the production LLM backend")
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self._model = config.model
        self._token_parameter = "max_completion_tokens"

    def complete(self, prompt: str, temperature: float, max_tokens: int) -> str:
        kwargs = dict(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature),
        )
        token_parameter = getattr(self, "_token_parameter", "max_completion_tokens")
        kwargs[token_parameter] = int(max_tokens)
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if token_parameter != "max_completion_tokens" or not (
                "unsupported" in message and "max_completion_tokens" in message
            ):
                raise
            kwargs.pop("max_completion_tokens")
            kwargs["max_tokens"] = int(max_tokens)
            response = self._client.chat.completions.create(**kwargs)
            self._token_parameter = "max_tokens"
        return (response.choices[0].message.content or "").strip()

    def __repr__(self) -> str:
        return f"OpenAIChatBackend(model={self._model!r})"


def parse_prediction_vector(text: str) -> PropertyPrediction | None:
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or "")
    if len(numbers) != 5:
        return None
    try:
        s1, est, oscillator_strength, ratio, score = map(float, numbers)
    except ValueError:
        return None
    if not all(np.isfinite([s1, est, oscillator_strength, ratio, score])):
        return None
    if not 0.0 <= score <= 10.0:
        return None
    return PropertyPrediction(s1, est, oscillator_strength, ratio, score)


def extract_smiles(text: str) -> str:
    candidates = re.findall(
        r"[A-Za-z0-9@+\-\\/()=\[\]#%.]+(?:\.[A-Za-z0-9@+\-\\/()=\[\]#%]+)*",
        text or "",
    )
    return candidates[0].strip() if candidates else ""


def generate_target_smiles(
    backend: LLMBackend,
    prompt: str,
    max_retries: int,
    validator: Callable[[str], bool],
    temperature: float = 0.2,
) -> str:
    """Generate one valid search target, feeding invalid attempts back on retry."""
    failed: set[str] = set()
    for attempt in range(max(1, int(max_retries))):
        retry_note = ""
        if failed:
            retry_note = f"\nDo not repeat these invalid outputs: {sorted(failed)}"
        raw = backend.complete(
            prompt + retry_note,
            temperature=min(float(temperature) + attempt * 0.05, 0.6),
            max_tokens=150,
        )
        smiles = extract_smiles(raw)
        if smiles and smiles not in failed and validator(smiles):
            return smiles
        if smiles:
            failed.add(smiles)
    return ""


def predict_properties(
    backend: LLMBackend,
    prompt: str,
    trials: int = 3,
    temperature: float = 0.15,
) -> PredictionStats:
    """Aggregate repeated five-property predictions into mean and deviation."""
    rows = []
    for trial in range(max(1, int(trials))):
        raw = backend.complete(
            prompt,
            temperature=min(float(temperature) + trial * 0.05, 0.5),
            max_tokens=80,
        )
        prediction = parse_prediction_vector(raw)
        if prediction is not None:
            rows.append(
                [
                    prediction.s1,
                    prediction.est,
                    prediction.oscillator_strength,
                    prediction.ratio,
                    prediction.score,
                ]
            )
    if not rows:
        raise ValueError("LLM returned no parseable property predictions")
    values = np.asarray(rows, dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    return PredictionStats(
        mean=PropertyPrediction(*mean.tolist()),
        std=PropertyPrediction(*std.tolist()),
        samples=len(rows),
    )
