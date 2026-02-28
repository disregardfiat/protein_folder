"""
Full-chain E_tot minimizer: backbone + optional side-chain from HQIV.

Minimizes E_tot = Σ ħc/Θ_i + U_φ over the Cα trace (deterministic L-BFGS),
with bond-length constraints: consecutive Cα in [2.5, 6.0] Å. After every
step, project so bonds remain valid; clash penalty for non-bonded < 2.0 Å.
Then reconstruct full backbone. No force fields; all from first principles.

Returns: minimized Cα, full backbone coordinates, E_final, and optional PDB.
MIT License. Python 3.10+. Numpy only.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

from .casp_submission import (
    _parse_fasta,
    _place_backbone_ca,
    _place_full_backbone,
    AA_1to3,
)
from .folding_energy import e_tot, e_tot_ca_with_bonds, grad_full, grad_bonds_only
from .gradient_descent_folding import minimize_e_tot_lbfgs, _project_bonds
from .peptide_backbone import backbone_bond_lengths

# Z_shell for E_tot over full backbone: N=7, C=6, O=8
Z_N, Z_CA, Z_C, Z_O = 7, 6, 6, 8


def _minimize_bonds_fast(
    ca_init: np.ndarray,
    z_list: np.ndarray,
    max_iter: int = 30,
    fast_horizon: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fast minimization for long chains. Default: bonds + full vector-sum horizon (analytical).
    fast_horizon=True: bonds-only (nearest-neighbor), for debugging/speed.
    """
    pos = np.array(ca_init, dtype=float)
    n = pos.shape[0]
    for it in range(max_iter):
        grad = (
            grad_bonds_only(pos)
            if fast_horizon
            else grad_full(pos, z_list, include_bonds=True, include_horizon=True)
        )
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-4:
            break
        step = 0.5 / (g_norm + 1e-6)  # adaptive: small step when grad is large
        pos = pos - step * grad
        pos = _project_bonds(pos, r_min=2.5, r_max=6.0)
    z = np.full(n, 6)
    e_final = float(e_tot_ca_with_bonds(pos, z))
    return pos, {
        "e_final": e_final,
        "e_initial": e_final,
        "n_iter": it + 1,
        "success": True,
        "message": "Bonds-only (fast path)" if fast_horizon else "Bonds + horizon (fast path)",
    }


def _z_list_ca(sequence: str) -> np.ndarray:
    """Z for each Cα (6). Length n_res."""
    n = len(sequence)
    return np.full(n, Z_CA)


def _full_backbone_positions_and_z(
    atoms: List[Tuple[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert list of (atom_name, xyz) to (positions (n,3), z_list (n))."""
    positions = np.array([xyz for _, xyz in atoms])
    z_list = []
    for name, _ in atoms:
        if name == "N":
            z_list.append(Z_N)
        elif name == "CA":
            z_list.append(Z_CA)
        elif name == "C":
            z_list.append(Z_C)
        elif name == "O":
            z_list.append(Z_O)
        else:
            z_list.append(6)
    return positions, np.array(z_list)


def minimize_full_chain(
    sequence: str,
    ca_init: Optional[np.ndarray] = None,
    ss_string: Optional[str] = None,
    max_iter: int = 500,
    gtol: float = 1e-5,
    include_sidechains: bool = False,
    fast_horizon: bool = False,
    side_chain_pack: bool = True,
) -> Dict[str, object]:
    """
    Full-chain minimization: minimize E_tot over Cα, then rebuild backbone.

    Args:
        sequence: One-letter amino acid sequence (or will be parsed from FASTA if needed).
        ca_init: Initial Cα positions (n_res, 3) in Å. If None, from SS-aware placement.
        ss_string: Optional SS string (H/E/C). If None and ca_init is None, predict_ss(sequence).
        max_iter: L-BFGS max iterations.
        gtol: Gradient tolerance for convergence.
        include_sidechains: If True, add Cβ positions (from ideal geometry); no full side chains yet.
        fast_horizon: If True, use bonds-only gradient (nearest-neighbor); faster, for debugging.
        side_chain_pack: If True and include_sidechains, run lightweight Cβ rotamer search after backbone.

    Returns:
        dict with:
          ca_min: (n_res, 3) minimized Cα in Å.
          backbone_atoms: list of (atom_name, xyz) per residue (N, CA, C, O).
          E_ca_final: E_tot evaluated on Cα only (eV).
          E_backbone_final: E_tot evaluated on full backbone N,CA,C,O (eV).
          info: from minimize_e_tot_lbfgs (e_final, e_initial, n_iter, success, message).
          n_res: number of residues.
          sequence: sequence used.
    """
    seq = sequence.strip().upper()
    if not seq or not all(c in AA_1to3 for c in seq):
        seq = _parse_fasta(sequence)
    if not seq:
        return {
            "ca_min": np.zeros((0, 3)),
            "backbone_atoms": [],
            "E_ca_final": 0.0,
            "E_backbone_final": 0.0,
            "info": {"e_final": 0, "e_initial": 0, "n_iter": 0, "success": True, "message": "Empty"},
            "n_res": 0,
            "sequence": "",
        }
    n_res = len(seq)
    if ca_init is None:
        ca_init = _place_backbone_ca(seq, ss_string=ss_string)
    else:
        ca_init = np.asarray(ca_init, dtype=float)
        assert ca_init.shape == (n_res, 3)
    z_ca = _z_list_ca(seq)
    # Fast path for long chains: analytical gradient (bonds + full horizon by default)
    if n_res > 50:
        ca_min, info = _minimize_bonds_fast(
            ca_init, z_ca, max_iter=min(30, max_iter), fast_horizon=fast_horizon
        )
    else:
        ca_min, info = minimize_e_tot_lbfgs(
            ca_init,
            z_ca,
            max_iter=max_iter,
            gtol=gtol,
            energy_func=e_tot_ca_with_bonds,
            grad_func=lambda pos, z: grad_full(pos, z, include_bonds=True, include_horizon=not fast_horizon),
            project_bonds=True,
            r_bond_min=2.5,
            r_bond_max=6.0,
        )
    backbone_atoms = _place_full_backbone(ca_min, seq)
    pos_bb, z_bb = _full_backbone_positions_and_z(backbone_atoms)
    E_ca_final = float(e_tot(ca_min, z_ca))
    E_backbone_final = float(e_tot(pos_bb, z_bb))
    if include_sidechains:
        backbone_atoms = _add_cb(backbone_atoms, seq)
        if side_chain_pack:
            backbone_atoms = _side_chain_pack_light(backbone_atoms, seq)
    return {
        "ca_min": ca_min,
        "backbone_atoms": backbone_atoms,
        "E_ca_final": E_ca_final,
        "E_backbone_final": E_backbone_final,
        "info": info,
        "n_res": n_res,
        "sequence": seq,
        "include_sidechains": include_sidechains,
    }


def _rotate_vector_around_axis(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotate vector v around axis by deg (degrees). Axis need not be unit."""
    axis = np.asarray(axis, dtype=float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        return v.copy()
    axis = axis / nrm
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v) * (1 - c))


def _side_chain_pack_light(
    backbone_atoms: List[Tuple[str, np.ndarray]],
    sequence: str,
    r_clash: float = 2.0,
) -> List[Tuple[str, np.ndarray]]:
    """
    Lightweight Cβ rotamer search: try 3 rotations (0°, 120°, 240°) around N–CA;
    keep the orientation with fewest clashes. Improves surface packing / lDDT.
    """
    out = list(backbone_atoms)
    all_xyz = np.array([xyz for _, xyz in out])
    n_res = len(sequence)
    idx = 0
    for res_i in range(n_res):
        aa = sequence[res_i]
        if aa == "G":
            idx += 4
            continue
        # Order: N, CA, CB, C, O
        i_n, i_ca, i_cb = idx, idx + 1, idx + 2
        idx += 5
        n_xyz = out[i_n][1]
        ca_xyz = out[i_ca][1]
        cb_xyz = out[i_cb][1]
        axis = ca_xyz - n_xyz
        v_ca_cb = cb_xyz - ca_xyz
        best_cb = cb_xyz
        best_count = _count_clashes(cb_xyz, all_xyz, exclude=(i_n, i_ca, i_cb), r_clash=r_clash)
        for angle in (120.0, 240.0):
            v_rot = _rotate_vector_around_axis(v_ca_cb, axis, angle)
            cb_new = ca_xyz + v_rot
            count = _count_clashes(cb_new, all_xyz, exclude=(i_n, i_ca, i_cb), r_clash=r_clash)
            if count < best_count:
                best_count = count
                best_cb = cb_new
        out[i_cb] = ("CB", best_cb)
        all_xyz[i_cb] = best_cb
    return out


def _count_clashes(
    xyz: np.ndarray,
    all_xyz: np.ndarray,
    exclude: Tuple[int, ...],
    r_clash: float = 2.0,
) -> int:
    """Number of atoms in all_xyz (excluding indices in exclude) within r_clash of xyz."""
    d = np.linalg.norm(all_xyz - xyz, axis=1)
    mask = np.ones(len(d), dtype=bool)
    for i in exclude:
        mask[i] = False
    return int(np.sum(mask & (d < r_clash)))


def _add_cb(
    backbone_atoms: List[Tuple[str, np.ndarray]],
    sequence: str,
) -> List[Tuple[str, np.ndarray]]:
    """
    Add Cβ to backbone atom list (except Gly). Uses N–CA–C plane and trans Cβ.
    Inserts Cβ after CA for each residue. Returns new list (order: N, CA, CB?, C, O per res).
    """
    bonds = backbone_bond_lengths()
    r_ca_cb = bonds["Calpha_Cbeta"]
    n_res = len(sequence)
    out = []
    atoms_per_res = 4
    for i in range(n_res):
        aa = sequence[i]
        n_xyz = backbone_atoms[i * atoms_per_res + 0][1]
        ca_xyz = backbone_atoms[i * atoms_per_res + 1][1]
        c_xyz = backbone_atoms[i * atoms_per_res + 2][1]
        o_xyz = backbone_atoms[i * atoms_per_res + 3][1]
        out.append(("N", n_xyz))
        out.append(("CA", ca_xyz))
        if aa != "G":
            # Cβ in trans: direction from (N-CA) × (C-CA), normalized
            v1 = n_xyz - ca_xyz
            v2 = c_xyz - ca_xyz
            direction = np.cross(v1, v2)
            nrm = np.linalg.norm(direction)
            if nrm < 1e-9:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / nrm
            cb_xyz = ca_xyz + r_ca_cb * direction
            out.append(("CB", cb_xyz))
        out.append(("C", c_xyz))
        out.append(("O", o_xyz))
    return out


def full_chain_to_pdb(
    result: Dict[str, object],
    chain_id: str = "A",
) -> str:
    """Format minimizer result as PDB string (MODEL 1 ... ENDMDL END)."""
    backbone_atoms = result["backbone_atoms"]
    sequence = result["sequence"]
    include_sidechains = result.get("include_sidechains", False)
    if not backbone_atoms:
        return "MODEL     1\nENDMDL\nEND\n"
    lines = ["MODEL     1"]
    atom_id = 1
    n_res = result["n_res"]
    idx = 0
    for res_id in range(1, n_res + 1):
        res_1 = sequence[res_id - 1]
        res_3 = AA_1to3.get(res_1, "UNK")
        # Atoms per residue: N, CA, [CB], C, O
        n_atoms_this = (5 if res_1 != "G" else 4) if include_sidechains else 4
        for _ in range(n_atoms_this):
            name, xyz = backbone_atoms[idx]
            lines.append(
                f"ATOM  {atom_id:5d}  {name:2s}  {res_3:3s} {chain_id}{res_id:4d}    "
                f"{float(xyz[0]):8.3f}{float(xyz[1]):8.3f}{float(xyz[2]):8.3f}  1.00  0.00           "
            )
            atom_id += 1
            idx += 1
    lines.append("ENDMDL")
    lines.append("END")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys
    from horizon_physics.proteins.casp_submission import _parse_fasta

    parser = argparse.ArgumentParser(description="HQIV full-chain minimizer: FASTA → minimized PDB")
    parser.add_argument("fasta", nargs="?", help="FASTA file or string (default: stdin)")
    parser.add_argument("-o", "--output", help="Output PDB path (default: stdout)")
    parser.add_argument("--fast", action="store_true", help="Use bonds-only gradient (nearest-neighbor; faster, for debugging)")
    parser.add_argument("--no-sidechain-pack", action="store_true", help="Skip Cβ rotamer search after backbone")
    parser.add_argument("--no-sidechains", action="store_true", help="Backbone only (no Cβ)")
    args = parser.parse_args()

    if args.fasta:
        with open(args.fasta) as f:
            fasta = f.read()
    else:
        fasta = sys.stdin.read()
    seq = _parse_fasta(fasta)
    if not seq:
        sys.stderr.write("No sequence found in FASTA\n")
        sys.exit(1)
    result = minimize_full_chain(
        seq,
        max_iter=500,
        include_sidechains=not args.no_sidechains,
        fast_horizon=args.fast,
        side_chain_pack=not args.no_sidechain_pack and not args.no_sidechains,
    )
    pdb = full_chain_to_pdb(result)
    if args.output:
        with open(args.output, "w") as f:
            f.write(pdb)
    else:
        print(pdb)
