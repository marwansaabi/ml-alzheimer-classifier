"""Shared paths and chemistry helpers."""
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent
RAW_DATA = ROOT / "data" / "chembl_ache_ic50.csv"
CLEAN_DATA = ROOT / "data" / "ache_dataset_clean.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ACTIVE_NM = 1_000      # IC50 <= 1 µM  -> active
INACTIVE_NM = 10_000   # IC50 >= 10 µM -> inactive
N_BITS = 1024
RADIUS = 2
SEED = 42

_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)


def standardize(smiles):
    """Canonical SMILES of the parent molecule (largest fragment, neutralised), or None."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = _uncharger.uncharge(_largest_fragment.choose(mol))
    return Chem.MolToSmiles(mol)


def fingerprints(smiles_list):
    """Boolean matrix of Morgan fingerprints (radius 2, 1024 bits)."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return np.array([_morgan.GetFingerprintAsNumPy(m) for m in mols], dtype=bool)


def bit_vectors(smiles_list):
    """RDKit bit vectors, used for Tanimoto similarity."""
    return [_morgan.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_list]


def scaffold(smiles):
    """Bemis-Murcko scaffold; acyclic molecules share the empty scaffold."""
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles)
