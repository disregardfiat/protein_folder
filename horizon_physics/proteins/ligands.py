"""
Ligand support: parse ligand input (PDB HETATM, InChI, SMILES) and represent each
ligand as a 6-DOF rigid body (LigandAgent) that interacts with the protein via horizon forces.

Tunnel cone/lip plane apply ONLY to the protein chain; ligands are free 6-DOF agents.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# Default z_shell for ligand atoms (carbon-like for horizon)
DEFAULT_LIGAND_Z = 6


def _euler_to_rotation(euler: np.ndarray) -> np.ndarray:
    """ZYX euler (3,) in radians -> 3x3 rotation matrix."""
    a, b, c = float(euler[0]), float(euler[1]), float(euler[2])
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    return np.array([
        [ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc],
        [sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc],
        [-sb, cb * sc, cb * cc],
    ], dtype=np.float64)


class LigandAgent:
    """
    One ligand as a 6-DOF rigid body. Local coordinates are stored relative to centroid;
    world positions = t + R @ local, with R from euler angles (ZYX).
    Interacts with the protein via horizon + clash in e_tot.
    """

    def __init__(
        self,
        res_name: str = "LIG",
        atom_names: Optional[List[str]] = None,
        local_positions: Optional[np.ndarray] = None,
        elements: Optional[List[str]] = None,
        z_list: Optional[np.ndarray] = None,
    ):
        self.res_name = res_name.strip()[:3] or "LIG"
        self.atom_names = list(atom_names) if atom_names else []
        pos = np.asarray(local_positions, dtype=np.float64) if local_positions is not None else np.zeros((0, 3))
        if pos.ndim == 1:
            pos = pos.reshape(-1, 3)
        self.local_positions = pos  # (N, 3) relative to centroid
        self.elements = list(elements) if elements else ["C"] * len(self.atom_names)
        n = len(self.atom_names)
        if n and len(self.elements) != n:
            self.elements = (self.elements + ["C"] * n)[:n]
        self.z_list = np.asarray(z_list, dtype=np.int32) if z_list is not None else np.full(pos.shape[0], DEFAULT_LIGAND_Z)
        if self.z_list.size != pos.shape[0]:
            self.z_list = np.full(pos.shape[0], DEFAULT_LIGAND_Z)
        # 6-DOF: translation (Å) and euler ZYX (radians)
        self._t = np.zeros(3, dtype=np.float64)
        self._euler = np.zeros(3, dtype=np.float64)

    def set_6dof(self, t: np.ndarray, euler: np.ndarray) -> None:
        """Set pose: t (3,) in Å, euler (3,) ZYX in radians."""
        self._t = np.asarray(t, dtype=np.float64).ravel()[:3]
        self._euler = np.asarray(euler, dtype=np.float64).ravel()[:3]

    def get_6dof(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (t (3,), euler (3,))."""
        return self._t.copy(), self._euler.copy()

    def get_6dof_flat(self) -> np.ndarray:
        """Return (6,) for L-BFGS state."""
        return np.concatenate([self._t, self._euler])

    def set_6dof_flat(self, x: np.ndarray) -> None:
        """Set from (6,) state."""
        x = np.asarray(x, dtype=np.float64).ravel()
        self._t = x[:3].copy()
        self._euler = x[3:6].copy()

    def get_world_positions(self) -> np.ndarray:
        """World positions (N, 3) = t + R @ local."""
        if self.local_positions.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        R = _euler_to_rotation(self._euler)
        return self._t + (self.local_positions @ R.T)

    def n_atoms(self) -> int:
        return len(self.atom_names)


def _parse_pdb_hetatm_block(text: str) -> List[Tuple[str, int, str, np.ndarray, str]]:
    """Parse PDB ATOM/HETATM lines. Returns list of (res_name, res_seq, atom_name, xyz, element)."""
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip() or "LIG"
            res_seq = int(line[22:26].strip() or 1)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            elem = line[76:78].strip() if len(line) >= 78 else "C"
            out.append((res_name, res_seq, atom_name, np.array([x, y, z], dtype=np.float64), elem or "C"))
        except (ValueError, IndexError):
            continue
    return out


def _parse_smiles_or_inchi_line(line: str) -> Optional[Tuple[str, List[str], np.ndarray, List[str]]]:
    """If line looks like SMILES or InChI, try to generate 3D. Returns (res_name, atom_names, xyz, elements) or None."""
    line = line.strip()
    if not line or len(line) > 2000:
        return None
    # Optional RDKit: generate 3D from SMILES/InChI
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = None
        if line.startswith("InChI="):
            mol = Chem.MolFromInchi(line)
        else:
            mol = Chem.MolFromSmiles(line)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
        conf = mol.GetConformer()
        atom_names = []
        xyz = []
        elements = []
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            sym = atom.GetSymbol()
            elements.append(sym)
            atom_names.append(f"{sym}{i+1}")
            pt = conf.GetAtomPosition(i)
            xyz.append([pt.x, pt.y, pt.z])
        xyz = np.array(xyz, dtype=np.float64)
        return ("LIG", atom_names, xyz, elements)
    except Exception:
        return None


def parse_ligands(ligand_str: str) -> List[LigandAgent]:
    """
    Parse ligand_str into a list of LigandAgent.
    Supports: (1) PDB block (ATOM/HETATM lines), (2) one SMILES or InChI per line.
    Returns [] if empty or parse fails.
    """
    if not ligand_str or not ligand_str.strip():
        return []
    text = ligand_str.strip()
    agents: List[LigandAgent] = []

    # If it looks like PDB (has ATOM or HETATM), parse as PDB; one agent per residue (res_name, res_seq)
    if "ATOM  " in text or "HETATM" in text:
        records = _parse_pdb_hetatm_block(text)
        if not records:
            return []
        by_res: Dict[Tuple[str, int], List[Tuple[str, np.ndarray, str]]] = defaultdict(list)
        for res_name, res_seq, atom_name, xyz, elem in records:
            by_res[(res_name, res_seq)].append((atom_name, xyz, elem))
        for (res_name, _res_seq), atoms in by_res.items():
            positions = np.array([a[1] for a in atoms])
            centroid = np.mean(positions, axis=0)
            local = positions - centroid
            atom_names = [a[0] for a in atoms]
            elements = [a[2] for a in atoms]
            ag = LigandAgent(res_name=res_name, atom_names=atom_names, local_positions=local, elements=elements)
            ag.set_6dof(centroid, np.zeros(3))
            agents.append(ag)
        return agents

    # Line-by-line: SMILES or InChI
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = _parse_smiles_or_inchi_line(line)
        if parsed is None:
            continue
        res_name, atom_names, xyz, elements = parsed
        centroid = np.mean(xyz, axis=0)
        local = xyz - centroid
        ag = LigandAgent(res_name=res_name, atom_names=atom_names, local_positions=local, elements=elements)
        ag.set_6dof(centroid, np.zeros(3))
        agents.append(ag)

    return agents
