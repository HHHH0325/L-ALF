"""Convert selected SMILES into traceable three-dimensional ORCA starting points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class ConformerResult:
    success: bool
    status: str
    charge: int = 0
    multiplicity: int = 1
    symbols: tuple[str, ...] = ()
    coordinates: tuple[tuple[float, float, float], ...] = ()


def _charge_and_multiplicity(molecule: Chem.Mol) -> tuple[int, int]:
    charge = int(Chem.GetFormalCharge(molecule))
    electrons = sum(atom.GetAtomicNum() for atom in molecule.GetAtoms()) - charge
    return charge, 1 if electrons % 2 == 0 else 2


def _coordinates_valid(coordinates: np.ndarray, minimum_distance: float = 0.55) -> bool:
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        return False
    if not np.isfinite(coordinates).all():
        return False
    for left in range(len(coordinates)):
        for right in range(left + 1, len(coordinates)):
            if float(np.linalg.norm(coordinates[left] - coordinates[right])) < minimum_distance:
                return False
    return True


def generate_conformer(smiles: str, random_seed: int = 42) -> ConformerResult:
    """Generate an ETKDGv3 conformer and pre-optimize it with MMFF or UFF."""
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return ConformerResult(False, "invalid_smiles")
    charge, multiplicity = _charge_and_multiplicity(molecule)
    molecule = Chem.AddHs(molecule)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = False
    embed_code = AllChem.EmbedMolecule(molecule, params)
    if embed_code != 0:
        params.useRandomCoords = True
        embed_code = AllChem.EmbedMolecule(molecule, params)
    if embed_code != 0:
        return ConformerResult(False, "embedding_failed", charge, multiplicity)

    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=500)
            status = "mmff"
        else:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
            status = "uff"
    except Exception:
        status = "embedded_unoptimized"

    conformer = molecule.GetConformer()
    coordinates = np.asarray(conformer.GetPositions(), dtype=float)
    if not _coordinates_valid(coordinates):
        return ConformerResult(False, "invalid_coordinates", charge, multiplicity)
    symbols = tuple(atom.GetSymbol() for atom in molecule.GetAtoms())
    coordinate_rows = tuple(tuple(map(float, row)) for row in coordinates)
    return ConformerResult(
        True,
        status,
        charge,
        multiplicity,
        symbols,
        coordinate_rows,
    )


def write_xyz(path: Path, result: ConformerResult, molecule_id: str) -> None:
    if not result.success:
        raise ValueError("Cannot write an unsuccessful conformer")
    lines = [str(len(result.symbols)), f"molecule_id={molecule_id}; source=RDKit"]
    for symbol, (x, y, z) in zip(result.symbols, result.coordinates):
        lines.append(f"{symbol:<2} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
