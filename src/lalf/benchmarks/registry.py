"""Only methods retained in the final manuscript comparison."""

METHODS = {
    "llm_gen_rf": ("R1-7", "LLM evaluator replaced by RF", ["breiman2001rf"]),
    "llm_gen_xgb": ("R1-7", "LLM evaluator replaced by XGBoost", ["chen2016xgboost"]),
    "llm_gen_gp": ("R1-7", "LLM evaluator replaced by fingerprint GP", ["srinivas2010gpucb"]),
    "llm_gen_graph": ("R1-7", "LLM evaluator replaced by GCN ensemble", ["kipf2017gcn"]),
    "similarity_gen_llm": ("R1-7", "Observed-anchor generator", ["rogers2010ecfp"]),
    "rule_gen_llm": ("R1-7", "Valence-checked local mutation generator", []),
    "evolution_gen_llm": ("R1-7", "Local mutation plus observed-fitness generator", []),
    "llm_no_context": ("R1-7", "No evaluation references and simplified generation prompt", []),
    "random": ("R1-9", "One uniform sample without replacement, then batch slicing", []),
    "similarity_search": ("R1-9", "Observed-anchor Tanimoto search", ["rogers2010ecfp"]),
    "evolutionary_search": ("R1-9", "Finite-library Morgan-bit genetic search", []),
    "rf_search": ("R1-9", "RF surrogate whole-library search", ["breiman2001rf"]),
    "xgb_search": ("R1-9", "XGBoost surrogate whole-library search", ["chen2016xgboost"]),
    "gp_search": ("R1-9", "Fingerprint GP whole-library search", ["srinivas2010gpucb"]),
    "graph_search": ("R1-9", "GCN ensemble whole-library search", ["kipf2017gcn"]),
    "graph_gp_search": (
        "R1-9",
        "Bond-labeled WL kernel GP",
        ["shervashidze2011wl", "griffiths2023gauche"],
    ),
    "graphga_search": ("R1-9", "Graph-GA finite-library adaptation", ["jensen2019graphga"]),
    "molleo_search": (
        "R1-9",
        "MolLEO paper-driven finite-library adaptation",
        ["wang2025molleo", "jensen2019graphga"],
    ),
    "qpots": ("R1-9", "Historical qPOTS discrete-pool adaptation", ["renganathan2025qpots"]),
    "qlognparego": (
        "R1-9",
        "Historical BoTorch qLogNParEGO adaptation",
        ["knowles2006parego", "daulton2020qehvi", "ament2023logei", "botorchqlognparego"],
    ),
}


def needs_llm(method: str) -> bool:
    return method in METHODS and (METHODS[method][0] == "R1-7" or method == "molleo_search")
