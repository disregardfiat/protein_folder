"""
Insulin fragment example: B-chain N-terminal (e.g. first 30 residues) from HQIV.

FASTA → PDB via hqiv_predict_structure. MIT License, Python 3.10+, numpy only.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from horizon_physics.proteins import hqiv_predict_structure, backbone_geometry

# Insulin B-chain residues 1–30 (fragment)
INSULIN_B_FRAGMENT_FASTA = """>insulin B 1-30
FVNQHLCGSHLVEALYLVCGERGFFYTPK

"""

def main():
    geom = backbone_geometry()
    print("Insulin B fragment (30 residues) — HQIV")
    print(f"  Cα–C: {geom['Calpha_C']:.4f} Å, ω: {geom['omega_deg']:.1f}°")
    pdb = hqiv_predict_structure(INSULIN_B_FRAGMENT_FASTA)
    n_atoms = pdb.count("ATOM")
    print(f"  PDB: {n_atoms} ATOM lines")
    out_path = os.path.join(os.path.dirname(__file__), "insulin_b_fragment_hqiv.pdb")
    with open(out_path, "w") as f:
        f.write(pdb)
    print(f"  Written: {out_path}")
    return pdb


if __name__ == "__main__":
    main()
