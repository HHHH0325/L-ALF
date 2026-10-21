# L-ALF: LLM-Guided Multi-Objective Active Learning with DFT Feedback

L-ALF connects large-language-model reasoning, molecular-library retrieval,
active sampling, and quantum-chemical validation in a closed optimization loop.
Its purpose is to allocate a limited DFT budget to candidates most likely to
expand a feasible multi-objective Pareto frontier.

## Installation and quick start

Use Python 3.10 or newer. Clone the repository and install the package in a
virtual environment:

```sh
git clone https://github.com/HHHH0325/L-ALF.git
cd L-ALF
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
lalf check-config --config configs/public.example.yaml
python scripts/check_public_release.py .
```

Configuration validation works offline without a molecular dataset and makes
no LLM requests or ORCA submissions. The public release contains algorithm
code, configuration templates, documentation and references. Validation tests,
fixtures, demonstration datasets and test outputs are kept locally and are
excluded from the current source tree and distribution packages.

For a production run, provide a molecular pool CSV with `molecule_id,smiles`
columns, and set its path in a copy of `configs/public.example.yaml`. History
uses `molecule_id,smiles,s1,est,f,socme,ratio,t_state`; an absent history file or
a file containing only this header starts with zero task labels. Configuration
paths are resolved relative to the YAML file. The default paths are placeholders
for user-supplied files; no molecular pool or history CSV is bundled.

Set connection details through environment variables in your local session:

| Variable | Purpose |
|---|---|
| `LALF_LLM_API_KEY` | API credential for the LLM backend |
| `LALF_LLM_BASE_URL` | Optional compatible API endpoint |
| `LALF_LLM_MODEL` | Model identifier supported by your provider |
| `LALF_SSH_HOST`, `LALF_SSH_USER` | Compute-cluster connection |
| `LALF_SSH_PORT` | Optional SSH port; defaults to 22 |
| `LALF_SSH_KEY_FILE` | Optional SSH private-key file path |
| `LALF_SSH_PASSWORD` | Optional password authentication |

The compute cluster must provide ORCA and Slurm commands (`sbatch`, `squeue`,
and `sacct`). Install ORCA separately according to its license, configure its
executable and Slurm settings in YAML, and verify the SSH host key in your
known-hosts file before connecting. Then run or inspect the workflow with:

```sh
lalf run --config configs/my.local.yaml
lalf status --config configs/my.local.yaml
```

`run` calls the configured LLM service and submits quantum-chemical jobs. The
persisted state allows the same command to resume an interrupted run. The
example configuration uses six excited-state roots; users supply the candidate
library, credentials, model, and computing resources for their own experiments.
Both public example configurations specify 20 rounds with 20 selected molecules
per round, for a planned total of 400 selections per run.

The optimization targets are computational screening descriptors. Their values
alone do not establish an experimental ISC rate, ROS yield, or photosensitizer
performance. This repository publishes the packaged algorithm and configuration
templates; it does not include the labeled research dataset, original run
trajectories, or manuscript experimental results.

## Manuscript comparison algorithms

The [benchmark code and guide](benchmarks/README.md) provide the 20 retained
module controls and whole-search baselines used for R1-7 and R1-9, including
configurations, [method descriptions](benchmarks/METHODS.md), and
[literature references](benchmarks/references.bib). They use the same public
molecular/label schema and scientific thresholds as L-ALF. Each method retains
its actual ranking and acquisition rules; finite-library adaptations and
differences from historical L-ALF are documented explicitly.

```sh
python -m pip install -e ".[benchmarks]"
lalf-benchmark --list-methods
lalf-benchmark --config configs/benchmarks.example.yaml --check-config
```

Benchmark execution requires a user-supplied labeled library. Published files
contain code, configuration and citations; research results are not included.

## Algorithm overview

```text
User molecular library
  -> feasible history and current Pareto frontier
  -> f-focus, ratio-focus, and balance-focus target profiles
  -> LLM-guided directional molecular design
  -> fingerprint retrieval and dynamic MMR expansion
  -> context-enhanced repeated LLM property prediction
  -> hard constraints, Pareto layers, and UCB ranking
  -> ORCA DFT/TD-DFT validation
  -> property parsing and history/frontier update
  -> next active-learning round
```

## 1. Three frontier-guided search directions

At every round, L-ALF extracts molecules satisfying the photophysical working
window and computes a Pareto frontier in oscillator strength `f` and the ISC
descriptor `R = SOCME / DeltaEST`.

Three complementary profiles guide the next search:

- `f-focus` advances the high-oscillator-strength boundary.
- `ratio-focus` advances the high-`SOCME / DeltaEST` boundary.
- `balance-focus` explores trade-offs between both objectives.

This prevents the search from collapsing onto a single local direction. In a
zero-start run, the first targets are generated from task constraints; once DFT
labels become available, later rounds are guided by verified frontier anchors.

## 2. Fingerprint retrieval and diversified expansion

An LLM-generated SMILES describes a useful structural direction, but is not
treated as the final DFT candidate. Instead, it becomes a retrieval anchor for
the user-provided molecular library.

Morgan fingerprints and Tanimoto similarity measure target relevance. Greedy
MMR then balances relevance and structural diversity:

```text
MMR(x) = lambda * Similarity(x, target)
         - (1 - lambda) * max Similarity(x, selected)
```

The resulting neighborhood remains related to the target while avoiding a
batch dominated by redundant analogues.

## 3. Context-enhanced LLM property prediction

Each candidate is evaluated together with:

- representative global Pareto-frontier molecules;
- structurally local feasible positive references;
- structurally local infeasible or low-quality negative references.

The LLM predicts:

```text
[S1, DeltaEST, f, SOCME/DeltaEST, overall_score]
```

Repeated independent predictions provide a mean performance estimate and a
standard deviation that measures inference consistency. The deviation is used
as an exploration signal rather than interpreted as a strict Bayesian
posterior uncertainty.

## 4. Pareto-UCB active sampling

The default feasibility window is:

```text
1.3 eV <= S1 <= 3.2 eV
DeltaEST < 0.30 eV
```

Predicted feasible candidates are non-dominated-sorted in `f` and
`SOCME/DeltaEST`. Within the same Pareto layer, candidates are ranked by:

```text
UCB(x) = mean_score(x) + beta * std_score(x)
```

The final DFT batch is selected by predicted feasibility, Pareto layer, and UCB
rank. This combines exploitation of high-potential candidates with exploration
of candidates whose evaluations remain less stable.

## 5. DFT/TD-DFT oracle

Selected SMILES are converted into three-dimensional structures with RDKit
ETKDGv3. MMFF is attempted first and UFF is used as a fallback. A mapping table
preserves the relationship between `molecule_id`, SMILES, charge,
multiplicity, and ORCA file number.

Default calculations use:

- Ground-state optimization and frequencies: PBE0-D3/def2-SVP, RIJCOSX,
  CPCM(Water).
- Excited states and SOC: wB97X-D3, ZORA-def2-TZVP, RIJCOSX, TightSCF,
  CPCM(Water).
- SARC-ZORA-TZVP for Sb, I, and Te.
- Six low-energy roots with singlet, triplet, and singlet-triplet SOC output.

## 6. Dominant ISC channel

For each low-energy triplet state `Tn`, L-ALF computes the absolute raw gap:

```text
Delta_n = |E(S1) - E(Tn)|
R(n) = SOCME(n) / Delta_n
```

Within the inclusive configured window (default `0.01 <= Delta_n <= 0.30 eV`),
the channel with available SOC and the largest `R(n)` is selected. If none exists,
the parser uses the nearest triplet channel with available SOC and floors its
gap at `est_min` (default `0.01 eV`). `DeltaEST` is the resulting processed label,
and `SOCME/DeltaEST` uses that same denominator. No valid SOC channel means a
missing label. Triplets above S1 and zero raw gaps are handled by the same rule.

Final screening uses `DeltaEST < 0.30 eV` by default, with no additional lower
bound. A channel at `0.30 eV` may be parsed but fails screening. A label floored
to `0.01 eV` does not imply a raw gap of at least `0.01 eV`.

Parsed labels include `S1`, `DeltaEST`, `f`, `SOCME`, `SOCME/DeltaEST`, and the
dominant triplet-state index.

## 7. Closed-loop update

New DFT labels are merged into the history by stable `molecule_id`. Conflicting
ID-to-SMILES mappings are rejected. The feasible set, Pareto frontier, target
anchors, and local reference neighborhoods are recomputed before the next
round.

The complete loop is therefore:

```text
design -> retrieve -> predict -> select -> validate -> update
```

## Source layout

```text
src/lalf/
├── workflow.py
├── active_learning/   # profiles, prompts, Pareto-UCB
├── chemistry/         # fingerprints, MMR, conformers
├── dft/               # ORCA bundle generation and parsing
├── hpc/               # SSH/SFTP and Slurm orchestration
├── llm/               # LLM interface and repeated prediction
└── storage/           # history and resumable state
```

## Algorithm outputs

Round outputs include expanded candidates, repeated LLM
predictions, Pareto-UCB selections, ORCA bundles, parsed DFT labels, updated
Pareto fronts, and resumable workflow state.

The molecular candidate library and DFT-labeled research dataset used in the
study are not distributed with this repository.
