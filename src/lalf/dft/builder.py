"""Build the high-cost oracle batch selected by Pareto-UCB.

Every molecule receives a stable file number and mapping record before the
ground-state and excited-state ORCA inputs are rendered.  This mapping is the
identity bridge used when calculated labels return to the active-learning
history.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from lalf.chemistry.conformers import generate_conformer, write_xyz
from lalf.config import DFTConfig
from lalf.models import BundleManifest


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _render(template_name: str, replacements: dict[str, object]) -> str:
    content = (_TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    for key, value in replacements.items():
        content = content.replace("{{" + key + "}}", str(value))
    if re.search(r"\{\{[A-Z0-9_]+\}\}", content):
        raise ValueError(f"Unresolved placeholder in {template_name}")
    return content


def build_orca_bundle(
    selected: pd.DataFrame,
    bundle_dir: str | Path,
    config: DFTConfig,
) -> BundleManifest:
    """Create structures, two-stage ORCA inputs, mapping, and array-job script."""
    required = {"molecule_id", "smiles"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected table is missing columns: {missing}")

    directory = Path(bundle_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    mapping_rows: list[dict[str, object]] = []
    valid_numbers: list[int] = []

    for file_number, (_, row) in enumerate(selected.iterrows(), start=1):
        molecule_id = str(row["molecule_id"]).strip()
        smiles = str(row["smiles"]).strip()
        conformer = generate_conformer(smiles, random_seed=config.random_seed + file_number)
        mapping = {
            "file_number": file_number,
            "molecule_id": molecule_id,
            "smiles": smiles,
            "charge": conformer.charge,
            "multiplicity": conformer.multiplicity,
            "is_valid": bool(conformer.success),
            "structure_status": conformer.status,
        }
        mapping_rows.append(mapping)
        if not conformer.success:
            continue

        valid_numbers.append(file_number)
        write_xyz(directory / f"a{file_number}.xyz", conformer, molecule_id)
        replacements = {
            "FILE_NUMBER": file_number,
            "CHARGE": conformer.charge,
            "MULTIPLICITY": conformer.multiplicity,
            "NPROCS": config.nprocs,
            "MAXCORE_MB": config.maxcore_mb,
            "GROUND_METHOD": config.ground_method,
            "EXCITED_METHOD": config.excited_method,
            "SOLVENT": config.solvent,
            "NROOTS": config.nroots,
        }
        (directory / f"{file_number}.inp").write_text(
            _render("ground_state.inp.tpl", replacements),
            encoding="utf-8",
            newline="\n",
        )
        (directory / f"{file_number}_td.inp").write_text(
            _render("excited_state.inp.tpl", replacements),
            encoding="utf-8",
            newline="\n",
        )

    mapping_path = directory / "mapping.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False, encoding="utf-8")
    (directory / "ids.txt").write_text(
        "".join(f"{number}\n" for number in valid_numbers),
        encoding="utf-8",
        newline="\n",
    )
    if valid_numbers:
        run_script = _render(
            "run_array.sh",
            {"VALID_COUNT": len(valid_numbers), "NPROCS": config.nprocs},
        )
    else:
        run_script = "#!/usr/bin/env bash\necho 'No valid molecules in this bundle.'\n"
    (directory / "run_array.sh").write_text(
        run_script, encoding="utf-8", newline="\n"
    )
    return BundleManifest(
        bundle_dir=directory,
        mapping_path=mapping_path,
        valid_count=len(valid_numbers),
        invalid_count=len(mapping_rows) - len(valid_numbers),
    )
