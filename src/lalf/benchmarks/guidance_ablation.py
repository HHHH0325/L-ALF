"""Remove explicit chemical heuristics while preserving references and targets.

This does not suppress the model's internal reasoning or pretrained knowledge.
"""

import re
from .prompts import inverse_design_prompt_optimized, comprehensive_scoring_prompt


def generation_prompt(profile):
    return strip_generation_guidance(profile)


def evaluation_prompt(smiles, refs):
    return strip_evaluation_guidance(smiles, refs)


def strip_generation_guidance(profile):
    prompt = inverse_design_prompt_optimized(profile, [])
    prompt = prompt.replace(
        "You are an expert in photosensitizer molecular design.",
        "Propose a molecular SMILES for the supplied numerical targets.",
    )
    prompt = re.sub(r"Design principles:\n.*?(?=Trade-off focus:)", "", prompt, flags=re.S)
    prompt = re.sub(
        r"Use anchor-based local optimization.*?(?=Chemical validity emphasis:)",
        "",
        prompt,
        flags=re.S,
    )
    prompt = prompt.replace(
        "Use a conservative domain-prior startup scaffold instead of an overly complex structure.\n",
        "",
    )
    prompt = re.sub(
        r"Chemical validity emphasis:\n.*?(?=Return exactly one new SMILES)", "", prompt, flags=re.S
    )
    return prompt


def strip_evaluation_guidance(smiles, refs):
    prompt = comprehensive_scoring_prompt(
        smiles, refs["global_frontier"], refs["local_positive"], refs["local_negative"]
    )
    prompt = prompt.replace(
        "You are an expert evaluator for photosensitizer molecules.",
        "Estimate molecular properties from the supplied inputs.",
    )
    prompt = prompt.replace(
        "Use the reference molecules as examples of strong, local-positive, and local-negative patterns.",
        "",
    )
    prompt = prompt.replace(
        "Penalize the target if it appears to repeat these local failure or boundary patterns without clearly improving them.",
        "",
    )
    return prompt
