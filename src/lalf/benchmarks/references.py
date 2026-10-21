from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from lalf.active_learning.selection import pareto_layers
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .settings import EST_MAX, F_MIN, F_STRICT_MIN, RATIO_MIN, RATIO_STRICT_MIN, S1_HIGH, S1_LOW


_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def calculate_morgan_fingerprint(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return _FP_GENERATOR.GetFingerprint(mol)
    except Exception:
        return None


def find_most_similar_examples(
    target_smiles: str, examples: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    if not examples or k <= 0:
        return []

    target_fp = calculate_morgan_fingerprint(target_smiles)
    if target_fp is None:
        return examples[:k]

    ranked = []
    for ex in examples:
        smiles = str(ex.get("SMILES", "")).strip()
        fp = calculate_morgan_fingerprint(smiles)
        if fp is None:
            continue
        ranked.append((float(DataStructs.TanimotoSimilarity(target_fp, fp)), ex))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [ex for _, ex in ranked[:k]]


class SuccessfulExamplesManager:
    def __init__(self):
        self.pareto_weights = {"SOCME/EST": 0.50, "f": 0.50}

    @staticmethod
    def _to_records(samples_df: pd.DataFrame) -> List[Dict[str, Any]]:
        if samples_df is None or samples_df.empty:
            return []
        return samples_df.to_dict("records")

    @staticmethod
    def _safe_float(mol: Dict[str, Any], key: str, default: float = np.nan) -> float:
        try:
            return float(mol.get(key, default))
        except Exception:
            return default

    def _is_valid(self, mol: Dict[str, Any]) -> bool:
        est = self._safe_float(mol, "EST", np.inf)
        s1 = self._safe_float(mol, "S1", -np.inf)
        return (est < EST_MAX) and (S1_LOW <= s1 <= S1_HIGH)

    def _is_strict_practical(self, mol: Dict[str, Any]) -> bool:
        if not self._is_valid(mol):
            return False
        f_val = self._safe_float(mol, "f", 0.0)
        ratio = self._safe_float(mol, "SOCME/EST", 0.0)
        return (f_val > F_STRICT_MIN) and (ratio > RATIO_STRICT_MIN)

    def _constraint_gap(self, mol: Dict[str, Any]) -> float:
        est = self._safe_float(mol, "EST", np.inf)
        s1 = self._safe_float(mol, "S1", -np.inf)

        est_gap = max(0.0, est - EST_MAX)
        if s1 < S1_LOW:
            s1_gap = S1_LOW - s1
        elif s1 > S1_HIGH:
            s1_gap = s1 - S1_HIGH
        else:
            s1_gap = 0.0
        return float(est_gap + 0.5 * s1_gap)

    def _objective_gap(self, mol: Dict[str, Any]) -> float:
        f_val = self._safe_float(mol, "f", 0.0)
        ratio = self._safe_float(mol, "SOCME/EST", 0.0)
        f_gap = max(0.0, F_MIN - f_val) / max(F_MIN, 1e-6)
        ratio_gap = max(0.0, RATIO_MIN - ratio) / max(RATIO_MIN, 1e-6)
        return float(f_gap + ratio_gap)

    def _objective_strength(self, mol: Dict[str, Any]) -> float:
        f_val = max(0.0, self._safe_float(mol, "f", 0.0))
        ratio = max(0.0, self._safe_float(mol, "SOCME/EST", 0.0))
        return float(f_val / max(F_MIN, 1e-6) + ratio / max(RATIO_MIN, 1e-6))

    def _pareto_front(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not molecules:
            return []

        objectives = np.array(
            [
                [-self._safe_float(m, "SOCME/EST", 0.0), -self._safe_float(m, "f", 0.0)]
                for m in molecules
            ]
        )
        front_idx = np.flatnonzero(
            pareto_layers(pd.DataFrame(-objectives, columns=["mean_ratio", "mean_f"])).to_numpy()
            == 0
        )
        return [molecules[i] for i in front_idx]

    def identify_pareto_front(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = [m for m in molecules if self._is_valid(m)]
        return self._pareto_front(valid)

    def calculate_weighted_crowding_distance(
        self, molecules: List[Dict[str, Any]]
    ) -> Dict[int, float]:
        if len(molecules) <= 2:
            return {i: 1.0 for i in range(len(molecules))}

        ranges = {}
        for obj in ("f", "SOCME/EST"):
            values = [self._safe_float(m, obj, 0.0) for m in molecules]
            span = max(values) - min(values)
            ranges[obj] = span if span > 1e-6 else 1.0

        crowding: Dict[int, float] = {}
        for idx in range(len(molecules)):
            score = 0.0
            for obj, weight in self.pareto_weights.items():
                ordered = sorted(
                    ((j, self._safe_float(m, obj, 0.0)) for j, m in enumerate(molecules)),
                    key=lambda item: item[1],
                    reverse=True,
                )
                positions = [pos for pos, (j, _) in enumerate(ordered) if j == idx]
                if not positions:
                    continue
                pos = positions[0]
                left_val = ordered[pos - 1][1] if pos > 0 else ordered[pos][1]
                right_val = ordered[pos + 1][1] if pos < len(ordered) - 1 else ordered[pos][1]
                score += abs(right_val - left_val) / ranges[obj] * weight
            crowding[idx] = score
        return crowding

    def select_diverse_frontier_examples(
        self, molecules: List[Dict[str, Any]], n: int
    ) -> List[Dict[str, Any]]:
        frontier = self.identify_pareto_front(molecules)
        if not frontier or n <= 0:
            return []
        if len(frontier) <= n:
            return frontier

        crowding = self.calculate_weighted_crowding_distance(frontier)
        order = sorted(
            crowding.keys(),
            key=lambda idx: (crowding[idx], self._objective_strength(frontier[idx])),
            reverse=True,
        )
        return [frontier[idx] for idx in order[:n]]

    def get_discovered_frontier_archive(self, samples_df: pd.DataFrame) -> List[Dict[str, Any]]:
        molecules = self._to_records(samples_df)
        return self.identify_pareto_front(molecules)

    def get_archive_stats(self, frontier_archive: List[Dict[str, Any]]) -> Dict[str, float] | None:
        if not frontier_archive:
            return None

        f_values = np.array([self._safe_float(m, "f", 0.0) for m in frontier_archive], dtype=float)
        ratio_values = np.array(
            [self._safe_float(m, "SOCME/EST", 0.0) for m in frontier_archive], dtype=float
        )
        return {
            "best_f": float(np.max(f_values)),
            "best_ratio": float(np.max(ratio_values)),
            "q75_f": float(np.quantile(f_values, 0.75)),
            "q75_ratio": float(np.quantile(ratio_values, 0.75)),
            "median_f": float(np.median(f_values)),
            "median_ratio": float(np.median(ratio_values)),
        }

    def get_generation_memory(
        self,
        samples_df: pd.DataFrame,
        anchor_history: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, float] | None, Dict[str, Dict[str, Any]]]:
        molecules = self._to_records(samples_df)
        frontier_archive = self.get_discovered_frontier_archive(samples_df)
        archive_stats = self.get_archive_stats(frontier_archive)
        guidance = self.get_frontier_guidance(
            frontier_archive,
            molecules=molecules,
            anchor_history=anchor_history,
        )
        return frontier_archive, archive_stats, guidance

    def _sorted_focus_candidates(
        self, molecules: List[Dict[str, Any]], profile_name: str
    ) -> List[Dict[str, Any]]:
        pool = [mol for mol in molecules if self._is_strict_practical(mol)]
        if not pool:
            return []

        if profile_name == "f_focus":
            return sorted(
                pool,
                key=lambda mol: (
                    self._safe_float(mol, "f", 0.0),
                    self._safe_float(mol, "SOCME/EST", 0.0),
                ),
                reverse=True,
            )

        return sorted(
            pool,
            key=lambda mol: (
                self._safe_float(mol, "SOCME/EST", 0.0),
                self._safe_float(mol, "f", 0.0),
            ),
            reverse=True,
        )

    def _anchor_history_counts(
        self,
        profile_name: str,
        anchor_history: Dict[str, List[Dict[str, Any]]] | None,
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if not anchor_history:
            return counts

        for row in anchor_history.get(profile_name, []):
            smiles = str(row.get("ref_smiles", "")).strip()
            if not smiles:
                continue
            counts[smiles] = counts.get(smiles, 0) + 1
        return counts

    def _select_focus_anchor(
        self,
        molecules: List[Dict[str, Any]],
        profile_name: str,
        anchor_history: Dict[str, List[Dict[str, Any]]] | None,
    ) -> List[Dict[str, Any]]:
        ranked = self._sorted_focus_candidates(molecules, profile_name)
        if not ranked:
            return []

        counts = self._anchor_history_counts(profile_name, anchor_history)
        for candidate in ranked:
            if counts.get(self._anchor_signature(candidate), 0) < 3:
                return [candidate]
        return [ranked[-1]]

    def _legacy_balance_guidance(self, frontier_archive: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not frontier_archive:
            return {
                "anchor_examples": [],
                "region_hint": "The discovered frontier is still sparse; search for a balanced middle trade-off.",
                "gap_priority": 1.0,
                "target_f_hint": F_MIN * 1.15,
                "target_ratio_hint": RATIO_MIN * 1.10,
            }

        f_values = np.array([self._safe_float(m, "f", 0.0) for m in frontier_archive], dtype=float)
        ratio_values = np.array(
            [self._safe_float(m, "SOCME/EST", 0.0) for m in frontier_archive], dtype=float
        )
        f_scale = float(np.nanmax(f_values)) if np.nanmax(f_values) > 0 else 1.0
        ratio_scale = float(np.nanmax(ratio_values)) if np.nanmax(ratio_values) > 0 else 1.0

        highest_f = max(frontier_archive, key=lambda mol: self._safe_float(mol, "f", 0.0))
        highest_ratio = max(
            frontier_archive, key=lambda mol: self._safe_float(mol, "SOCME/EST", 0.0)
        )

        gap_examples = [highest_f, highest_ratio]
        gap_hint = "Current frontier suggests a balanced middle trade-off is still under-explored."
        gap_priority = 1.0
        gap_target_f = float(np.median(f_values))
        gap_target_ratio = float(np.median(ratio_values))

        if len(frontier_archive) >= 2:
            enriched = []
            for mol in frontier_archive:
                enriched.append(
                    {
                        "mol": mol,
                        "f_norm": self._safe_float(mol, "f", 0.0) / f_scale,
                        "ratio_norm": self._safe_float(mol, "SOCME/EST", 0.0) / ratio_scale,
                    }
                )

            enriched.sort(key=lambda item: item["f_norm"])
            best_gap = None
            for left, right in zip(enriched[:-1], enriched[1:]):
                dist = float(
                    np.hypot(
                        right["f_norm"] - left["f_norm"],
                        right["ratio_norm"] - left["ratio_norm"],
                    )
                )
                if best_gap is None or dist > best_gap["dist"]:
                    best_gap = {
                        "left": left["mol"],
                        "right": right["mol"],
                        "dist": dist,
                        "mid_f": 0.5
                        * (
                            self._safe_float(left["mol"], "f", 0.0)
                            + self._safe_float(right["mol"], "f", 0.0)
                        ),
                        "mid_ratio": 0.5
                        * (
                            self._safe_float(left["mol"], "SOCME/EST", 0.0)
                            + self._safe_float(right["mol"], "SOCME/EST", 0.0)
                        ),
                        "mid_f_norm": 0.5 * (left["f_norm"] + right["f_norm"]),
                        "mid_ratio_norm": 0.5 * (left["ratio_norm"] + right["ratio_norm"]),
                    }

            if best_gap is not None:
                gap_examples = [best_gap["left"], best_gap["right"]]
                gap_priority = 1.0 + best_gap["dist"]
                gap_target_f = float(best_gap["mid_f"])
                gap_target_ratio = float(best_gap["mid_ratio"])

                if best_gap["mid_f_norm"] >= 0.65 and best_gap["mid_ratio_norm"] < 0.45:
                    gap_hint = (
                        "There is an under-covered high-f / moderate-ratio region between the current frontier anchors. "
                        "Try to extend that trade-off pocket."
                    )
                elif best_gap["mid_ratio_norm"] >= 0.65 and best_gap["mid_f_norm"] < 0.45:
                    gap_hint = (
                        "There is an under-covered high-SOCME/EST / moderate-f region between the current frontier anchors. "
                        "Try to extend that trade-off pocket."
                    )
                else:
                    gap_hint = (
                        "There is a sparse middle section between two frontier anchors. "
                        "Search for a balanced non-dominated molecule in that gap."
                    )

        return {
            "anchor_examples": gap_examples,
            "region_hint": gap_hint,
            "gap_priority": gap_priority,
            "target_f_hint": gap_target_f,
            "target_ratio_hint": gap_target_ratio,
        }

    def _anchor_signature(self, mol: Dict[str, Any]) -> str:
        smiles = str(mol.get("SMILES", "")).strip()
        if smiles:
            return smiles
        return str(mol.get("molecule_id", "")).strip()

    def _balance_pair_signature(
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> tuple[str, str]:
        return tuple(sorted([self._anchor_signature(left), self._anchor_signature(right)]))

    def _history_pair_signature(self, rows: List[Dict[str, Any]]) -> tuple[str, str] | None:
        signatures = []
        for row in sorted(rows, key=lambda item: int(item.get("ref_rank", 0) or 0)):
            smiles = str(row.get("ref_smiles", "")).strip()
            if smiles:
                signatures.append(smiles)
        if len(signatures) < 2:
            return None
        return tuple(sorted(signatures[:2]))

    def _rank_balance_pairs(
        self,
        molecules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        strict_front = self._pareto_front(
            [mol for mol in molecules if self._is_strict_practical(mol)]
        )
        if len(strict_front) < 2:
            return []

        strict_front = sorted(strict_front, key=lambda mol: self._safe_float(mol, "f", 0.0))
        f_values = np.array([self._safe_float(m, "f", 0.0) for m in strict_front], dtype=float)
        ratio_values = np.array(
            [self._safe_float(m, "SOCME/EST", 0.0) for m in strict_front], dtype=float
        )
        f_scale = float(np.nanmax(f_values)) if np.nanmax(f_values) > 0 else 1.0
        ratio_scale = float(np.nanmax(ratio_values)) if np.nanmax(ratio_values) > 0 else 1.0

        candidate_pairs: List[tuple[int, int]] = [
            (idx, idx + 1) for idx in range(len(strict_front) - 1)
        ]
        if len(strict_front) >= 4:
            interior_pairs = [
                (left, right)
                for left, right in candidate_pairs
                if left > 0 and right < len(strict_front) - 1
            ]
            if interior_pairs:
                candidate_pairs = interior_pairs

        ranked_pairs = []
        for left_idx, right_idx in candidate_pairs:
            left = strict_front[left_idx]
            right = strict_front[right_idx]
            left_f_norm = self._safe_float(left, "f", 0.0) / f_scale
            right_f_norm = self._safe_float(right, "f", 0.0) / f_scale
            left_ratio_norm = self._safe_float(left, "SOCME/EST", 0.0) / ratio_scale
            right_ratio_norm = self._safe_float(right, "SOCME/EST", 0.0) / ratio_scale

            gap = float(np.hypot(right_f_norm - left_f_norm, right_ratio_norm - left_ratio_norm))
            mid_f_norm = 0.5 * (left_f_norm + right_f_norm)
            mid_ratio_norm = 0.5 * (left_ratio_norm + right_ratio_norm)
            center_penalty = abs(mid_f_norm - 0.58) + abs(mid_ratio_norm - 0.58)
            position_penalty = abs(
                ((left_idx + right_idx) / 2.0) / max(len(strict_front) - 1, 1) - 0.5
            )
            score = gap - 0.35 * center_penalty - 0.20 * position_penalty

            ranked_pairs.append(
                {
                    "left": left,
                    "right": right,
                    "gap": gap,
                    "score": score,
                    "mid_f": 0.5
                    * (self._safe_float(left, "f", 0.0) + self._safe_float(right, "f", 0.0)),
                    "mid_ratio": 0.5
                    * (
                        self._safe_float(left, "SOCME/EST", 0.0)
                        + self._safe_float(right, "SOCME/EST", 0.0)
                    ),
                    "signature": self._balance_pair_signature(left, right),
                }
            )

        ranked_pairs.sort(key=lambda item: item["score"], reverse=True)
        return ranked_pairs

    def _balance_pair_history_counts(
        self,
        anchor_history: Dict[str, List[Dict[str, Any]]] | None,
    ) -> Dict[tuple[str, str], int]:
        counts: Dict[tuple[str, str], int] = {}
        if not anchor_history:
            return counts

        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for row in anchor_history.get("balance_focus", []):
            round_idx = int(row.get("round", 0) or 0)
            if round_idx <= 0:
                continue
            grouped.setdefault(round_idx, []).append(row)

        for round_idx in sorted(grouped):
            signature = self._history_pair_signature(grouped[round_idx])
            if signature is not None:
                counts[signature] = counts.get(signature, 0) + 1
        return counts

    def _select_balance_guidance(
        self,
        frontier_archive: List[Dict[str, Any]],
        molecules: List[Dict[str, Any]],
        anchor_history: Dict[str, List[Dict[str, Any]]] | None,
    ) -> Dict[str, Any]:
        ranked_pairs = self._rank_balance_pairs(molecules)
        if not ranked_pairs:
            return self._legacy_balance_guidance(frontier_archive)

        pair_counts = self._balance_pair_history_counts(anchor_history)
        best_pair = ranked_pairs[-1]
        for pair in ranked_pairs:
            if pair_counts.get(pair["signature"], 0) < 3:
                best_pair = pair
                break
        return {
            "anchor_examples": [best_pair["left"], best_pair["right"]],
            "region_hint": (
                "Use two strict-practical Pareto anchors from the balanced middle section, skipping any pair already used three times."
            ),
            "gap_priority": 1.0 + best_pair["gap"],
            "target_f_hint": float(best_pair["mid_f"]),
            "target_ratio_hint": float(best_pair["mid_ratio"]),
        }

    def get_frontier_guidance(
        self,
        frontier_archive: List[Dict[str, Any]],
        molecules: List[Dict[str, Any]] | None = None,
        anchor_history: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        molecules = molecules or list(frontier_archive)
        if not frontier_archive:
            return {
                "f_focus": {
                    "anchor_examples": [],
                    "region_hint": "The discovered frontier is still sparse; expand the high-f side cautiously.",
                    "gap_priority": 1.0,
                },
                "ratio_focus": {
                    "anchor_examples": [],
                    "region_hint": "The discovered frontier is still sparse; expand the high-SOCME/EST side cautiously.",
                    "gap_priority": 1.0,
                },
                "balance_focus": {
                    "anchor_examples": [],
                    "region_hint": "The discovered frontier is still sparse; search for a balanced middle trade-off.",
                    "gap_priority": 1.0,
                    "target_f_hint": F_MIN * 1.15,
                    "target_ratio_hint": RATIO_MIN * 1.10,
                },
            }
        f_anchor = self._select_focus_anchor(molecules, "f_focus", anchor_history)
        ratio_anchor = self._select_focus_anchor(molecules, "ratio_focus", anchor_history)
        balance_guidance = self._select_balance_guidance(
            frontier_archive, molecules, anchor_history
        )

        return {
            "f_focus": {
                "anchor_examples": f_anchor,
                "region_hint": (
                    "Use strict-practical high-f anchors, skipping any candidate that has already appeared as this profile's anchor three times."
                ),
                "gap_priority": 1.0,
            },
            "ratio_focus": {
                "anchor_examples": ratio_anchor,
                "region_hint": (
                    "Use strict-practical high-SOCME/EST anchors, skipping any candidate that has already appeared as this profile's anchor three times."
                ),
                "gap_priority": 1.0,
            },
            "balance_focus": balance_guidance,
        }

    def get_design_reference_examples(
        self,
        samples_df: pd.DataFrame,
        best_k: int = 6,
        anchor_history: Dict[str, List[Dict[str, Any]]] | None = None,
    ) -> tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        Dict[str, float] | None,
        Dict[str, Dict[str, Any]],
    ]:
        frontier_archive = self.get_discovered_frontier_archive(samples_df)
        guidance = self.get_frontier_guidance(
            frontier_archive,
            molecules=self._to_records(samples_df),
            anchor_history=anchor_history,
        )

        priority_refs: List[Dict[str, Any]] = []
        seen_smiles: set[str] = set()
        for key in ("balance_focus", "f_focus", "ratio_focus"):
            for ref in guidance.get(key, {}).get("anchor_examples", []):
                smiles = str(ref.get("SMILES", "")).strip()
                if not smiles or smiles in seen_smiles:
                    continue
                priority_refs.append(ref)
                seen_smiles.add(smiles)

        diverse_refs = self.select_diverse_frontier_examples(frontier_archive, best_k)
        selected_refs = list(priority_refs)
        for ref in diverse_refs:
            smiles = str(ref.get("SMILES", "")).strip()
            if not smiles or smiles in seen_smiles:
                continue
            selected_refs.append(ref)
            seen_smiles.add(smiles)
            if len(selected_refs) >= best_k:
                break

        archive_stats = self.get_archive_stats(frontier_archive)
        return frontier_archive, selected_refs[:best_k], archive_stats, guidance

    def _rank_by_similarity(
        self,
        target_smiles: str,
        examples: Iterable[Dict[str, Any]],
        k: int,
        exclude_smiles: set[str] | None = None,
        tie_breaker=None,
    ) -> List[Dict[str, Any]]:
        if k <= 0:
            return []

        exclude_smiles = exclude_smiles or set()
        target_fp = calculate_morgan_fingerprint(target_smiles)
        ranked = []

        for ex in examples:
            smiles = str(ex.get("SMILES", "")).strip()
            if not smiles or smiles in exclude_smiles:
                continue

            fp = calculate_morgan_fingerprint(smiles)
            sim = (
                float(DataStructs.TanimotoSimilarity(target_fp, fp))
                if (target_fp is not None and fp is not None)
                else 0.0
            )
            tie_value = tie_breaker(ex) if tie_breaker is not None else 0.0
            ranked.append((sim, tie_value, ex))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [ex for _, _, ex in ranked[:k]]

    def _negative_priority(self, mol: Dict[str, Any]) -> float:
        gap = self._constraint_gap(mol) + 0.35 * self._objective_gap(mol)
        return -gap

    def get_scoring_references(
        self,
        samples_df: pd.DataFrame,
        target_smiles: str,
        global_k: int,
        local_positive_k: int,
        local_negative_k: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        records = self._to_records(samples_df)
        if not records:
            return {
                "global_frontier": [],
                "local_positive": [],
                "local_negative": [],
            }

        frontier_archive = self.get_discovered_frontier_archive(samples_df)
        global_frontier = self.select_diverse_frontier_examples(frontier_archive, global_k)
        frontier_smiles = {str(m.get("SMILES", "")).strip() for m in frontier_archive}
        used_smiles = {str(m.get("SMILES", "")).strip() for m in global_frontier}

        positive_pool = [m for m in records if self._is_valid(m)]
        local_positive = self._rank_by_similarity(
            target_smiles=target_smiles,
            examples=positive_pool,
            k=local_positive_k,
            exclude_smiles=used_smiles,
            tie_breaker=self._objective_strength,
        )
        used_smiles.update(str(m.get("SMILES", "")).strip() for m in local_positive)

        negative_pool = []
        for mol in records:
            smiles = str(mol.get("SMILES", "")).strip()
            if not smiles or smiles in frontier_smiles:
                continue
            if not self._is_valid(mol):
                negative_pool.append(mol)
                continue

            f_val = self._safe_float(mol, "f", 0.0)
            ratio = self._safe_float(mol, "SOCME/EST", 0.0)
            if f_val < F_MIN * 1.25 or ratio < RATIO_MIN * 1.25:
                negative_pool.append(mol)

        local_negative = self._rank_by_similarity(
            target_smiles=target_smiles,
            examples=negative_pool,
            k=local_negative_k,
            exclude_smiles=used_smiles,
            tie_breaker=self._negative_priority,
        )

        return {
            "global_frontier": global_frontier,
            "local_positive": local_positive,
            "local_negative": local_negative,
        }
