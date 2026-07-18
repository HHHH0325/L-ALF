"""Extract the physically consistent DFT labels used by the active learner.

For every triplet below S1, the parser computes the positive singlet-triplet
gap.  Within the configured window it selects the state pair maximizing
SOCME/DeltaEST; if the window is empty, it falls back to the nearest positive
gap with available SOC data.
"""

from __future__ import annotations

import math
import re

from lalf.models import ParsedDFTResult


_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_STATE_PATTERN = re.compile(
    rf"STATE\s+(\d+):\s*E=\s*{_NUMBER}\s+au\s+({_NUMBER})\s*eV",
    re.IGNORECASE,
)
_SIMPLE_SOC_PATTERN = re.compile(
    rf"SOCME\s+T=(\d+)\s+S=(\d+)\s+X=\(({_NUMBER}),({_NUMBER})\)\s+"
    rf"Y=\(({_NUMBER}),({_NUMBER})\)\s+Z=\(({_NUMBER}),({_NUMBER})\)",
    re.IGNORECASE,
)
_ORCA_SOC_PATTERN = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+"
    rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)\s+"
    rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)\s+"
    rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)",
    re.IGNORECASE,
)


def _state_energies(text: str) -> tuple[float | None, dict[int, float]]:
    s1: float | None = None
    triplets: dict[int, float] = {}
    manifold = ""
    for line in text.splitlines():
        upper = line.upper()
        if "EXCITED STATES" in upper and "SINGLET" in upper:
            manifold = "singlet"
            continue
        if "EXCITED STATES" in upper and "TRIPLET" in upper:
            manifold = "triplet"
            continue
        match = _STATE_PATTERN.search(line)
        if match is None:
            continue
        state = int(match.group(1))
        energy = float(match.group(2))
        if manifold == "triplet" and 1 <= state <= 6:
            triplets[state] = energy
        elif manifold == "singlet" and state == 1 and s1 is None:
            s1 = energy
        elif state == 1 and s1 is None:
            s1 = energy
        elif 7 <= state <= 12:
            triplets[state - 6] = energy
    return s1, triplets


def _soc_magnitudes(text: str) -> dict[int, float]:
    magnitudes: dict[int, float] = {}
    for match in _SIMPLE_SOC_PATTERN.finditer(text):
        t_state = int(match.group(1))
        s_state = int(match.group(2))
        if s_state != 1:
            continue
        components = [float(match.group(index)) for index in range(3, 9)]
        magnitudes[t_state] = float(math.sqrt(sum(value * value for value in components)))

    for line in text.splitlines():
        match = _ORCA_SOC_PATTERN.match(line)
        if match is None:
            continue
        t_state = int(match.group(1))
        s_state = int(match.group(2))
        if s_state != 1 or not 1 <= t_state <= 6:
            continue
        components = [float(match.group(index)) for index in range(3, 9)]
        magnitudes.setdefault(
            t_state, float(math.sqrt(sum(value * value for value in components)))
        )

    for line in text.splitlines():
        if "(" not in line or "Root" in line:
            continue
        parts = line.split()
        if len(parts) < 16:
            continue
        try:
            t_state = int(parts[0])
            s_state = int(parts[1])
            if s_state != 1 or not 1 <= t_state <= 6:
                continue
            components = [
                float(parts[3]),
                float(parts[5]),
                float(parts[8]),
                float(parts[10]),
                float(parts[13]),
                float(parts[15]),
            ]
        except (ValueError, IndexError):
            continue
        magnitudes.setdefault(
            t_state, float(math.sqrt(sum(value * value for value in components)))
        )
    return magnitudes


def _oscillator_strength(text: str) -> float | None:
    in_absorption = False
    for line in text.splitlines():
        if "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
            in_absorption = True
            continue
        if in_absorption and "0-1A" in line and "1-1A" in line:
            parts = line.split()
            if len(parts) >= 7:
                try:
                    return float(parts[6])
                except ValueError:
                    pass
            values = re.findall(_NUMBER, line)
            if values:
                try:
                    return float(values[-1])
                except ValueError:
                    return None
        if in_absorption and "ORCA TERMINATED" in line:
            break
    return None


def parse_td_output(text: str, est_min: float = 0.01, est_max: float = 0.30) -> ParsedDFTResult:
    """Parse S1, f, the dominant triplet channel, SOCME, and their ratio."""
    normal = "ORCA TERMINATED NORMALLY" in str(text)
    if not normal:
        return ParsedDFTResult(status="incomplete", normal_termination=False)

    s1, triplets = _state_energies(text)
    if s1 is None:
        return ParsedDFTResult(status="missing_s1", normal_termination=True)
    if not triplets:
        return ParsedDFTResult(status="missing_triplets", normal_termination=True, s1=s1)

    oscillator_strength = _oscillator_strength(text)
    if oscillator_strength is None:
        return ParsedDFTResult(
            status="missing_oscillator_strength", normal_termination=True, s1=s1
        )
    soc = _soc_magnitudes(text)
    positive_gaps = {
        state: float(s1 - energy)
        for state, energy in triplets.items()
        if float(s1 - energy) > 0.0
    }
    if not positive_gaps:
        return ParsedDFTResult(
            status="no_positive_triplet_gap",
            normal_termination=True,
            s1=s1,
            oscillator_strength=oscillator_strength,
        )

    window_states = {
        state: gap
        for state, gap in positive_gaps.items()
        if float(est_min) <= gap <= float(est_max) and state in soc
    }
    if window_states:
        best_state = max(window_states, key=lambda state: soc[state] / window_states[state])
    else:
        states_with_soc = {state: gap for state, gap in positive_gaps.items() if state in soc}
        if not states_with_soc:
            return ParsedDFTResult(
                status="missing_soc",
                normal_termination=True,
                s1=s1,
                oscillator_strength=oscillator_strength,
            )
        best_state = min(states_with_soc, key=states_with_soc.get)

    est = positive_gaps[best_state]
    socme = soc[best_state]
    return ParsedDFTResult(
        status="complete",
        normal_termination=True,
        s1=s1,
        est=est,
        oscillator_strength=oscillator_strength,
        socme=socme,
        ratio=socme / est,
        t_state=best_state,
    )
