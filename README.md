# L-ALF: LLM-Guided Multi-Objective Active Learning with DFT Feedback

L-ALF connects large-language-model reasoning, molecular-library retrieval,
active sampling, and quantum-chemical validation in a closed optimization loop.
Its purpose is to allocate a limited DFT budget to candidates most likely to
expand a feasible multi-objective Pareto frontier.

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

For each low-energy triplet state `Tj`, L-ALF computes the positive gap:

```text
DeltaEST(j) = E(S1) - E(Tj)
R(j) = SOCME(j) / DeltaEST(j)
```

Within the configured energy window, the state pair with the largest `R(j)` is
selected as the dominant ISC channel. If the window contains no valid state,
the parser falls back to the nearest positive triplet gap with available SOC
data.

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
