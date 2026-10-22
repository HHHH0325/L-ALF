# DFT/TD-DFT labeled benchmark dataset

[mol_label_total.csv](mol_label_total.csv) is the existing labeled molecular
library used for the finite-library manuscript comparisons. Molecules are
identified by PubChem CID and SMILES. The property columns are computational
screening labels, not experimental measurements. The original CSV is published
byte for byte, without recalculation, label correction, reordering or deduplication.

## Files and provenance

| File | Contents |
|---|---|
| `mol_label_total.csv` | Original 7-column label table, 12,181 records |
| `manifest.json` | Byte size, SHA256, schema and integrity observations |
| `README.md` | Field definitions, limitations and usage |

Release date: 2026-10-22. The table is the source library referenced by the
research comparison protocol as `mol_label_total.csv`. CIDs refer to
[PubChem](https://pubchem.ncbi.nlm.nih.gov/); the labels were supplied by the
study's DFT/TD-DFT workflow. The public calculation templates and processed
EST/channel definition are described in the [main README](../README.md).
The workflow uses six excited-state roots. This aggregate table does not
contain per-molecule inputs, termination flags, state-resolved energies or
calculation-to-file mappings, so it does not independently establish the
calculation settings or convergence status for every record.

CSV size: **1,909,558 bytes**. SHA256:

```text
03fbb79b3dc8ffbfff56cfd86326a983eb8cada2a8529ada3aca81967de36d55
```

Clone this repository or download the CSV from GitHub. The dataset is not
embedded in the Python wheel/source distributions.

## Column dictionary

| Original column | Meaning | Unit / representation | Runner column |
|---|---|---|---|
| `CID` | PubChem compound identifier | Integer identifier; treat as a stable ID | `molecule_id` |
| `SMILES` | Molecular structure string | SMILES | `smiles` |
| `S1` | Recorded lowest singlet excitation energy | eV | `s1` |
| `EST` | Processed absolute S1-to-selected-triplet gap | eV | `est` |
| `T_state` | Selected triplet channel index | 1-6, stored as a numeric value | `t_state` |
| `SOCME/EST` | SOC matrix-element magnitude / processed EST | cm^-1 eV^-1 | `ratio` |
| `f` | Recorded oscillator strength | Dimensionless | `f` |

`EST` is a selected-channel descriptor, not necessarily the S1-T1 gap.
The declared protocol selects the maximum SOCME/gap among available channels
with absolute gaps in the inclusive 0.01-0.30 eV window. If none exists, it
uses the nearest valid channel and floors the gap at 0.01 eV. Final screening
separately requires `1.3 <= S1 <= 3.2` and `EST < 0.30`. The aggregate CSV
does not contain the underlying channel data needed to recompute that choice.
SOCME is not provided as a separate column. The ratio is a screening descriptor,
not an experimentally measured ISC rate or ROS yield.

## Integrity and quality observations

The release audit found no blank fields or nonfinite property labels and
12,181 unique CIDs. All SMILES parse with RDKit 2026.03.1; parser acceptance
alone does not establish chemical or quantum-calculation validity.

There are **12,180 distinct SMILES strings**. CIDs `92026874` and `71333398`
share the same SMILES but have different recorded property labels. Both rows
remain in the published CSV. The existing comparison loader removes duplicate
SMILES strings by retaining the first row in file order (CID `92026874` here),
giving a 12,180-entry benchmark library. This is a literal-string rule, not
an additional structure-standardization or canonicalization step.

| Property | Minimum | Maximum | Negative values (original rows) |
|---|---:|---:|---:|
| `S1` | -1.265 | 12.736 | 167 |
| `EST` | 0.01 | 2.506 | 0 |
| `T_state` | 1 | 6 | 0 |
| `SOCME/EST` | 0 | 181643.970648629 | 0 |
| `f` | -0.000860249 | 5.373309755 | 162 |

Negative singlet excitation energies and oscillator strengths require checking
the original calculations/parsing before physical interpretation. All 162
negative-f records also have negative S1 and fall outside the stated S1
screening window. They remain in the library and are not removed from model
training automatically. This release preserves the historical inputs; it is
not a newly curated dataset with those issues resolved. The table alone cannot
show whether an anomalous value arose from a calculation or extraction error.

## Use with the comparison runner

After installing `.[benchmarks]`, run the following Python code from the
repository root. It changes column names only and writes a local adapter file
outside the repository, preserving record order and the original public CSV:

```python
from pathlib import Path
import pandas as pd

# Keep the recorded label text unchanged during column conversion.
frame = pd.read_csv("data/mol_label_total.csv", dtype=str)
frame = frame.rename(columns={
    "CID": "molecule_id", "SMILES": "smiles", "S1": "s1", "EST": "est",
    "T_state": "t_state", "SOCME/EST": "ratio",
})
output = Path("../user-data/labels.csv")
output.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(output, index=False)
```

The resulting path matches `configs/benchmarks.example.yaml`. That configuration
uses **20 rounds x 20 molecules**, with seeds 0-4. See the
[comparison guide](../benchmarks/README.md) for method selection and execution.
The benchmark queries existing labels after each batch is fixed and does not
run new ORCA calculations. This table is the hidden-label oracle for that
validation; loading all its labels into the production history would not be
a zero-start experiment.

## Availability scope

The public files include this labeled library, algorithm code, configurations
and citations. Historical acquisition logs, original five full-workflow run
tables, separate final screened-candidate tables, raw ORCA inputs/outputs,
API caches, figures and local validation files are not included. The labeled
subset is not a release of the complete PubChem retrieval universe.
Use the availability statement in the main README when describing this release.
