from __future__ import annotations

from typing import Iterable

import numpy as np

from .settings import EST_MAX, F_MIN, F_STRICT_MIN, RATIO_MIN, RATIO_STRICT_MIN, S1_HIGH, S1_LOW


# 为三类 target 提供不会收缩得过低的固定下限，避免 archive 偏弱时目标越来越保守。
F_FOCUS_F_MIN_FLOOR = max(1.00, F_STRICT_MIN * 3.0)
F_FOCUS_RATIO_FLOOR = max(40.0, RATIO_MIN * 2.0)
RATIO_FOCUS_F_FLOOR = max(0.45, F_STRICT_MIN * 1.5)
RATIO_FOCUS_RATIO_FLOOR = max(140.0, RATIO_STRICT_MIN * 2.0)
BALANCE_FOCUS_F_FLOOR = max(0.60, F_STRICT_MIN * 2.0)
BALANCE_FOCUS_RATIO_FLOOR = max(84.0, RATIO_STRICT_MIN * 1.2)


from lalf.active_learning.prompts import EST_DEFINITION


DESIGN_EXPERT_PROMPT_OPTIMIZED = f"""
You are an expert in photosensitizer molecular design.
{EST_DEFINITION}
Design a novel photosensitizer that satisfies both hard constraints:
- EST < {EST_MAX:.2f} eV
- {S1_LOW:.2f} <= S1 <= {S1_HIGH:.2f} eV

Optimization goals:
- Maximize oscillator strength f
- Maximize SOCME/EST

Practical reference tiers:
- Relaxed practical tier: f > {F_MIN:.2f} and SOCME/EST > {RATIO_MIN:.0f}
- Strict practical tier: f > {F_STRICT_MIN:.2f} and SOCME/EST > {RATIO_STRICT_MIN:.0f}

Design principles:
- Favor rigid, conjugated, photosensitizer-like backbones.
- Use heavy-atom or heteroatom patterns only when they still produce chemically valid structures.
- Keep the output chemically plausible and parsable by RDKit.
- Return only one SMILES string, with no explanation.
"""


PARETO_PHOTOSENSITIZER_EVALUATOR = f"""
You are an expert evaluator for photosensitizer molecules.
{EST_DEFINITION}

First estimate the following values for the target molecule:
- S1
- EST
- f
- SOCME/EST

Then assign one overall score in [0, 10] using this rule:
- 8.0-10.0: satisfies hard constraints, is likely strict practical (f > {F_STRICT_MIN:.2f}, SOCME/EST > {RATIO_STRICT_MIN:.0f}), and appears likely to extend or improve the current Pareto frontier
- 6.0-7.9: satisfies hard constraints, is at least likely relaxed practical (f > {F_MIN:.2f}, SOCME/EST > {RATIO_MIN:.0f}), and appears competitive on the frontier
- 4.0-5.9: satisfies hard constraints but is likely dominated by better candidates or falls below relaxed practical quality
- 0.0-3.9: violates hard constraints or is clearly poor

Use the reference molecules as examples of strong, local-positive, and local-negative patterns.
Output only:
[S1, EST, f, SOCME/EST, score]
Use only numbers with commas, no extra text.
"""


def _safe_stats(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.max(arr)), float(np.quantile(arr, 0.75)), float(np.median(arr))


def calculate_target_profiles(
    frontier_archive: list[dict], rnd_state: int | None = None
) -> list[dict]:
    rng = np.random.default_rng(rnd_state)

    if not frontier_archive:
        return [
            {
                "name": "f_focus",
                "focus": "Prioritize oscillator strength while keeping SOCME/EST clearly useful.",
                "target_f": round(max(0.45, F_FOCUS_F_MIN_FLOOR), 2),
                "target_socme_est": round(max(90.0, F_FOCUS_RATIO_FLOOR), 2),
            },
            {
                "name": "ratio_focus",
                "focus": "Prioritize SOCME/EST while keeping oscillator strength acceptable.",
                "target_f": round(max(0.25, RATIO_FOCUS_F_FLOOR), 2),
                "target_socme_est": round(max(160.0, RATIO_FOCUS_RATIO_FLOOR), 2),
            },
            {
                "name": "balance_focus",
                "focus": "Seek a balanced trade-off between strong f and strong SOCME/EST.",
                "target_f": round(max(0.35, BALANCE_FOCUS_F_FLOOR), 2),
                "target_socme_est": round(max(120.0, BALANCE_FOCUS_RATIO_FLOOR), 2),
            },
        ]

    best_f, q75_f, median_f = _safe_stats(float(m.get("f", 0.0)) for m in frontier_archive)
    best_ratio, q75_ratio, median_ratio = _safe_stats(
        float(m.get("SOCME/EST", 0.0)) for m in frontier_archive
    )

    return [
        {
            "name": "f_focus",
            "focus": "Push f upward while keeping SOCME/EST competitive and constraints valid.",
            "target_f": round(
                max(best_f * rng.uniform(1.05, 1.15), F_MIN * 1.25, F_FOCUS_F_MIN_FLOOR), 2
            ),
            "target_socme_est": round(
                max(q75_ratio, median_ratio, RATIO_MIN * 0.9, F_FOCUS_RATIO_FLOOR), 2
            ),
            "region_hint": "Expand the high-f edge of the discovered frontier.",
            "anchor_examples": [],
            "priority_bias": 1.0,
        },
        {
            "name": "ratio_focus",
            "focus": "Push SOCME/EST upward while keeping f acceptable and constraints valid.",
            "target_f": round(max(q75_f, median_f, F_MIN * 0.95, RATIO_FOCUS_F_FLOOR), 2),
            "target_socme_est": round(
                max(best_ratio * rng.uniform(1.05, 1.15), RATIO_MIN * 1.2, RATIO_FOCUS_RATIO_FLOOR),
                2,
            ),
            "region_hint": "Expand the high-SOCME/EST edge of the discovered frontier.",
            "anchor_examples": [],
            "priority_bias": 1.0,
        },
        {
            "name": "balance_focus",
            "focus": "Look for a non-dominated balance instead of optimizing only one extreme.",
            "target_f": round(
                max(q75_f * rng.uniform(1.02, 1.08), F_MIN * 1.10, BALANCE_FOCUS_F_FLOOR), 2
            ),
            "target_socme_est": round(
                max(
                    q75_ratio * rng.uniform(1.02, 1.08), RATIO_MIN * 1.05, BALANCE_FOCUS_RATIO_FLOOR
                ),
                2,
            ),
            "region_hint": "Fill the sparse middle trade-off region of the frontier.",
            "anchor_examples": [],
            "priority_bias": 1.0,
        },
    ]


def enrich_target_profiles(
    base_profiles: list[dict],
    frontier_guidance: dict[str, dict],
    rnd_state: int | None = None,
) -> list[dict]:
    rng = np.random.default_rng(rnd_state)
    enriched = []

    for profile in base_profiles:
        guidance = frontier_guidance.get(profile["name"], {})
        merged = dict(profile)
        merged["region_hint"] = guidance.get("region_hint", profile.get("region_hint", ""))
        merged["anchor_examples"] = guidance.get(
            "anchor_examples", profile.get("anchor_examples", [])
        )
        merged["priority_bias"] = float(
            guidance.get("gap_priority", profile.get("priority_bias", 1.0))
        )

        if profile["name"] == "balance_focus":
            f_hint = float(guidance.get("target_f_hint", merged["target_f"]))
            ratio_hint = float(guidance.get("target_ratio_hint", merged["target_socme_est"]))
            merged["target_f"] = round(
                max(f_hint * rng.uniform(1.03, 1.10), merged["target_f"], BALANCE_FOCUS_F_FLOOR),
                2,
            )
            merged["target_socme_est"] = round(
                max(
                    ratio_hint * rng.uniform(1.03, 1.10),
                    merged["target_socme_est"],
                    BALANCE_FOCUS_RATIO_FLOOR,
                ),
                2,
            )
        elif profile["name"] == "f_focus" and merged["anchor_examples"]:
            anchor = merged["anchor_examples"][0]
            anchor_ratio = float(anchor.get("SOCME/EST", 0.0))
            merged["target_socme_est"] = round(
                max(anchor_ratio, merged["target_socme_est"], F_FOCUS_RATIO_FLOOR),
                2,
            )
        elif profile["name"] == "ratio_focus" and merged["anchor_examples"]:
            anchor = merged["anchor_examples"][0]
            anchor_f = float(anchor.get("f", 0.0))
            merged["target_f"] = round(
                max(anchor_f, merged["target_f"], RATIO_FOCUS_F_FLOOR),
                2,
            )

        enriched.append(merged)
    return enriched


def inverse_design_prompt_optimized(
    target_profile: dict,
    successful_examples: list[dict],
    invalid_smiles: list[str] | None = None,
) -> str:
    prompt = DESIGN_EXPERT_PROMPT_OPTIMIZED.strip() + "\n\n"
    prompt += f"Trade-off focus: {target_profile['name']}\n"
    prompt += f"Focus guidance: {target_profile['focus']}\n"
    if target_profile.get("region_hint"):
        prompt += f"Frontier-gap guidance: {target_profile['region_hint']}\n"
    prompt += (
        "Target thresholds for this design attempt:\n"
        f"- f > {target_profile['target_f']}\n"
        f"- SOCME/EST > {target_profile['target_socme_est']}\n\n"
    )

    if successful_examples:
        prompt += "Reference frontier molecules:\n"
        for idx, example in enumerate(successful_examples, start=1):
            prompt += (
                f"{idx}. SMILES={example['SMILES']} | "
                f"S1={example['S1']:.2f} | EST={example['EST']:.2f} | "
                f"f={example['f']:.2f} | SOCME/EST={example['SOCME/EST']:.2f}\n"
            )
        prompt += "\n"

    anchor_examples = target_profile.get("anchor_examples") or []
    if anchor_examples:
        prompt += "Frontier anchor molecules near the target region:\n"
        for idx, example in enumerate(anchor_examples, start=1):
            prompt += (
                f"{idx}. SMILES={example['SMILES']} | "
                f"S1={example['S1']:.2f} | EST={example['EST']:.2f} | "
                f"f={example['f']:.2f} | SOCME/EST={example['SOCME/EST']:.2f}\n"
            )
        prompt += (
            "Use anchor-based local optimization instead of free-form generation.\n"
            "Allowed edit types:\n"
            "- aromatic substituent replacement\n"
            "- donor end-group replacement\n"
            "- acceptor end-group replacement\n"
            "- bridge replacement between conjugated fragments\n"
            "- moderate heavy-atom or heteroatom swap\n"
            "- small electronic tuning edits\n"
            "Editing rules:\n"
            "- keep the main conjugated scaffold family close to the anchor molecules\n"
            "- do not invent a completely unrelated scaffold\n"
            "- make only 1-3 meaningful edits\n"
            "- do not copy an anchor unchanged\n"
            "- if two anchors are provided, move beyond or between them while keeping the scaffold style plausible\n\n"
        )
    else:
        prompt += (
            "No discovered frontier anchors are available yet.\n"
            "Use a conservative domain-prior startup scaffold instead of an overly complex structure.\n\n"
        )

    if invalid_smiles:
        prompt += "Previously proposed invalid or repeated SMILES. Do not repeat them:\n"
        for idx, smiles in enumerate(invalid_smiles[-5:], start=1):
            prompt += f"{idx}. {smiles}\n"
        prompt += (
            "Generate a chemically different scaffold from the invalid attempts above. "
            "Change the ring-closure pattern or substituent layout if needed.\n\n"
        )

    prompt += (
        "Chemical validity emphasis:\n"
        "- Favor conservative edits around plausible photosensitizer scaffolds.\n"
        "- Avoid overloaded halogenation or too many heavy atoms on one fused core.\n"
        "- Avoid unusual valence states, strained ring closures, and unrealistic atom patterns.\n"
        "- If using Br, I, Se, or other heavy atoms, keep their placement moderate and chemically standard.\n\n"
    )

    prompt += (
        "Return exactly one new SMILES string that follows the requested trade-off focus, "
        "satisfies both hard constraints, and is syntactically valid for RDKit.\n"
        "Do not return any explanation.\n"
    )
    return prompt


def comprehensive_scoring_prompt(
    smiles_string: str,
    archive_examples: list[dict],
    positive_references: list[dict],
    negative_references: list[dict],
) -> str:
    prompt = PARETO_PHOTOSENSITIZER_EVALUATOR.strip() + "\n\n"

    if archive_examples:
        prompt += "Global frontier references:\n"
        for idx, example in enumerate(archive_examples, start=1):
            prompt += (
                f"{idx}. SMILES={example['SMILES']} | "
                f"S1={example['S1']:.2f} | EST={example['EST']:.2f} | "
                f"f={example['f']:.2f} | SOCME/EST={example['SOCME/EST']:.2f}\n"
            )
        prompt += "\n"

    if positive_references:
        prompt += "Local positive references with related structures:\n"
        for idx, example in enumerate(positive_references, start=1):
            prompt += (
                f"{idx}. SMILES={example['SMILES']} | "
                f"S1={example['S1']:.2f} | EST={example['EST']:.2f} | "
                f"f={example['f']:.2f} | SOCME/EST={example['SOCME/EST']:.2f}\n"
            )
        prompt += "\n"

    if negative_references:
        prompt += "Local negative or boundary references with related structures:\n"
        for idx, example in enumerate(negative_references, start=1):
            prompt += (
                f"{idx}. SMILES={example['SMILES']} | "
                f"S1={example['S1']:.2f} | EST={example['EST']:.2f} | "
                f"f={example['f']:.2f} | SOCME/EST={example['SOCME/EST']:.2f}\n"
            )
        prompt += (
            "Penalize the target if it appears to repeat these local failure or boundary patterns "
            "without clearly improving them.\n\n"
        )

    prompt += (
        "Scoring policy refinement:\n"
        f"- If EST is likely >= {EST_MAX:.2f}, the score should usually be <= 3.0.\n"
        f"- If S1 is likely outside [{S1_LOW:.2f}, {S1_HIGH:.2f}], the score should usually be <= 3.0.\n"
        f"- Scores above 6.0 should usually require hard constraints plus at least relaxed practical quality "
        f"(f > {F_MIN:.2f}, SOCME/EST > {RATIO_MIN:.0f}).\n"
        f"- Scores above 8.0 should usually require hard constraints plus likely strict practical quality "
        f"(f > {F_STRICT_MIN:.2f}, SOCME/EST > {RATIO_STRICT_MIN:.0f}) or clear frontier-extension potential.\n"
        "- Prefer molecules that clearly improve a frontier anchor over molecules that merely resemble local positives.\n\n"
    )

    prompt += f"Target molecule: {smiles_string}\n"
    prompt += "Return only [S1, EST, f, SOCME/EST, score]."
    return prompt
