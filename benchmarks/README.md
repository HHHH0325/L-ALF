# Manuscript comparison code

This directory documents the 21 retained comparison methods: nine module
comparisons (R1-7) and twelve whole-search baselines (R1-9). Implementations live
in `src/lalf/benchmarks/`. The production L-ALF algorithm remains available as
`lalf run`; this addition supplies the comparison code, configuration and
references. Research datasets, run tables, figures, API responses and experimental
results are not distributed.

## Install and inspect without running experiments

From the repository root, in a Python virtual environment:

```sh
python -m pip install -e ".[dev,benchmarks]"
lalf-benchmark --list-methods
lalf-benchmark --config configs/benchmarks.example.yaml --check-config
python -m pytest -q
```

The checks and tests use artificial structures, invented labels, and fake LLM
responses. They make no network calls or ORCA submissions. Optional scientific
dependencies are isolated in the `benchmarks` extra; installing the production
package alone does not require PyTorch, BoTorch or XGBoost. Tested comparison
stack: SciPy 1.15.2, scikit-learn 1.6.1, XGBoost 3.1.1, PyTorch 2.9.1,
BoTorch 0.17.2 and GPyTorch 1.15.2. Public dependency ranges retain these minor
versions. CPU execution is supported; the GCN retains its historical automatic
CUDA selection, while the configuration `device` controls MOBO.

## Run with your own labeled library

Supply a CSV outside the repository using the same public field names as the
main algorithm:

```text
molecule_id,smiles,s1,est,f,ratio
```

`socme` and `t_state` may be included and are carried through. `s1` and `est`
are in eV; `ratio` means SOCME divided by the processed EST. Use the label
definition in the main README. Existing ORCA outputs can be parsed using
`lalf.dft.parser.parse_td_output`; benchmark code does not perform new quantum
calculations. It cannot reconstruct state energies or change the EST definition
from an aggregate label table alone.

Molecule IDs must be unique. Labels must be finite, and all structures must be
RDKit-readable. Duplicate SMILES strings retain the first row, preserving input
row order. For legacy uppercase tables, explicitly rename `SMILES` to `smiles`,
`S1` to `s1`, `EST` to `est`, and `SOCME/EST` to `ratio`; provide a stable
`molecule_id` column. Internal aliases preserve the paper's prediction order,
but all public input and output uses the main package schema.

Copy `configs/benchmarks.example.yaml` to an ignored `*.local.yaml` file, set
`labels_csv` to your external CSV, and select the methods to run:

```sh
lalf-benchmark --config configs/my-benchmarks.local.yaml
```

Paths resolve relative to the YAML file. R1-7 methods and `molleo_search` use the
same `LALF_LLM_API_KEY`, `LALF_LLM_BASE_URL`, `LALF_LLM_MODEL` variables and LLM
backend as the main algorithm. These commands make paid LLM requests when such
methods are selected; there is no endpoint, credential or historical provider
embedded in this distribution. Transport or JSON failures stop the method;
they do not become a different baseline.

The manuscript configuration fixes a batch size of 20, 20 rounds, a 400-label
budget, 50 retrieved candidates, and seeds 0–4. The portable runner allows 1–20
rounds for smoke checks. It requires 400 unique structures for a full random or
MOBO run and 430 for a full run requiring a final 50-candidate panel. These
minimums are checked before acquiring labels. Changing model constants or
the packaged `paper_protocol.json` defines a different experiment.

Local output is one `selected_verified_all.csv` per method and seed, plus the
configuration and immutable pre-query batch selections. A full run CSV has
400 rows, with `round` and `selection_rank` columns. Existing run directories
are rejected to avoid overwriting earlier work; the portable runner has no
resume mode. Runtime files belong under ignored `runs/` or outside the
repository and are not part of this code release.

## Alignment and interpretation

- Scientific defaults share the main package's `Thresholds` and parser EST
  definition. Hard feasibility is `1.3 <= s1 <= 3.2` and `est < 0.30`, without
  a separate screening lower bound. Strict descriptor hits additionally require
  `f > 0.3` and `ratio > 70`; relaxed hits require `f > 0.1` and `ratio > 20`.
- Predictors and generators only receive structures and their own queried
  history. Library fingerprints and PCA may use all unlabeled structures.
  Held-out labels stay behind `Oracle.query` until batch selection is fixed.
- Retained comparisons use the original detailed paper prompts, history
  reference selection, deterministic property-derived score and first-front
  ranking. The public production workflow has shorter prompts, LLM-reported
  scores and successive Pareto layers. Those differences are explicit; this
  addition does not change the production algorithm to fit the baselines.
- All non-random comparisons start with a shared, budgeted 20-label random
  first batch for a given seed. Random follows the original figure protocol:
  one uniform sample of the whole budget, sliced into batches. Historical
  L-ALF chose its first batch through its own full workflow. Zero initial
  labels do not imply identical first batches or paired runs.
- The original five L-ALF histories, exact historical backend versions and
  feature caches are not distributed. This is a portable refactoring of
  retained source, not a claim that a new invocation recreates the historical
  manuscript trajectories byte for byte. Third-party model names are requested
  identifiers, not independent proof of the provider's backend identity.

`metrics.evaluate(selected, reference)` supplies post-hoc descriptor hit counts
and two-objective HV, using log1p objectives normalized by the feasible reference
library maxima and reference point `(0,0)`. Reference labels enter evaluation
after acquisition only. First-hit budgets are rounded to the end of the hit
batch; absence of a hit is retained as censored. Response variation and ensemble
dispersion are ranking signals, not calibrated posterior uncertainties.

See [METHODS.md](METHODS.md) for the per-method changes and parameters,
[references.bib](references.bib) for citations, and
[PROVENANCE.json](PROVENANCE.json) for source provenance and release adaptations.
