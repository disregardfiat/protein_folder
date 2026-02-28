# HQIV Proteins Module

First-principles protein structure and folding from the **Horizon-centric Quantum Information and Vacuum (HQIV)** framework. No empirical force fields, no PDB statistics, no hard-coded radii—every geometry emerges from the discrete null lattice, causal-horizon monogamy, and informational-energy conservation.

## Core principles

- **Discrete null lattice**: Each atom is a lattice node with Z shells; bond lengths = diamond overlap Θ_ij.
- **Causal-horizon monogamy**: Entanglement budget per overlapping diamond constrains coordination and H-bonds.
- **Informational-energy**: E_tot = Σ m c² + Σ ħ c / Θ_i; minimization yields equilibrium geometry.
- **Horizon scalar**: φ = 2 c² / Θ_local; geometric damping f_φ = −γ′ φ / (a_loc + φ/6)² ∇φ.
- **Observer-centric**: No FLRW or empirical constants; all from HQVMetric and lattice.

## Installation

No external dependencies beyond **Python 3.10+** and **numpy**. Clone the repo and use:

```python
from horizon_physics.proteins import (
    hqiv_predict_structure,
    alpha_helix_geometry,
    backbone_geometry,
    beta_sheet_geometry,
    side_chain_chi_preferences,
    e_tot,
    minimize_e_tot,
    minimize_e_tot_lbfgs,
    rational_alpha_parameters,
    rational_ramachandran_alpha,
)
```

## Usage

### Backbone geometry (Å, degrees)

```python
from horizon_physics.proteins import backbone_geometry

g = backbone_geometry()
# Cα–C: 1.53 Å, C–N: 1.33 Å, ω: 180°, φ/ψ for α and β
print(g["Calpha_C"], g["C_N"], g["omega_deg"])
```

### Alpha helix

```python
from horizon_physics.proteins import alpha_helix_geometry

h = alpha_helix_geometry()
# 3.6 res/turn, 5.4 Å pitch, H-bond N–O ~ 2.9 Å
print(h["residues_per_turn"], h["pitch_ang"], h["hbond_N_O_ang"])
```

### Beta sheet

```python
from horizon_physics.proteins import beta_sheet_geometry

b = beta_sheet_geometry()
print(b["rise_per_residue_ang"], b["strand_spacing_ang"], b["hbond_N_O_ang"])
```

### Side-chain χ (20 amino acids)

```python
from horizon_physics.proteins import chi_angles_for_residue, side_chain_chi_preferences

chi = chi_angles_for_residue("VAL")
prefs = side_chain_chi_preferences()
```

### Folding energy and deterministic gradient descent

Only gradient-based optimization (no Monte Carlo, no random seeds). L-BFGS in pure numpy:

```python
from horizon_physics.proteins import minimize_e_tot_lbfgs, small_peptide_energy
import numpy as np

pos0 = np.array([[0,0,0], [3.8,0,0], [7.6,0,0]])
z = np.array([6, 6, 6])
pos_opt, info = minimize_e_tot_lbfgs(pos0, z, max_iter=200)
e = small_peptide_energy("AAA")
```

Rational parameters (φ = -57°, ψ = -47°, rise = 3/2 Å, pitch = 27/5 Å):

```python
from horizon_physics.proteins import rational_alpha_parameters, rational_ramachandran_alpha
rational_alpha_parameters()  # rise 3/2, pitch 27/5, residues_per_turn 18/5
rational_ramachandran_alpha()  # (-57, -47)
```

### Secondary structure prediction (no ML)

From E_tot(φ,ψ) basins at (-57°,-47°) and (-120°,120°); Θ_eff per residue from lattice:

```python
from horizon_physics.proteins import predict_ss, predict_ss_with_angles

ss, confidence = predict_ss("MKFLNDR", window=5)
# ss: 'H' (helix), 'E' (strand), 'C' (coil)
out = predict_ss_with_angles("MKFLNDR")  # includes dist_to_alpha_min, dist_to_beta_min
```

### Full-chain minimizer

Minimize E_tot over the Cα trace (L-BFGS or fast path), then rebuild full backbone (N, CA, C, O) and optional Cβ. **Default:** full vector-sum horizon gradient (every atom j contributes to force on i within 15 Å); optional lightweight Cβ rotamer search after backbone to improve packing/lDDT.

```python
from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb

result = minimize_full_chain("MKFLNDR", max_iter=300, include_sidechains=True)
# result["ca_min"], result["backbone_atoms"], result["E_ca_final"], result["E_backbone_final"]
pdb_str = full_chain_to_pdb(result, chain_id="A")
```

- **Gradients:** Analytical everywhere (`grad_bonds_only` + `grad_horizon_full`); no finite differences in L-BFGS or fast path.
- **Horizon:** Default = full vector sum over all pairs within horizon radius. Use `fast_horizon=True` (or CLI `--fast`) for bonds-only / nearest-neighbor (faster, for debugging).
- **Side-chain pack:** If `include_sidechains=True`, a lightweight Cβ rotamer search (0°, 120°, 240° around N–CA) runs after backbone minimization; disable with `side_chain_pack=False`.

### CASP submission: FASTA → PDB (SS-aware)

```python
from horizon_physics.proteins import hqiv_predict_structure, predict_ss

fasta = ">target\nMKFLNDR..."
pdb_str = hqiv_predict_structure(fasta)  # SS predicted from sequence
# Or pass SS explicitly:
ss, _ = predict_ss(sequence)
pdb_str = hqiv_predict_structure(fasta, ss_string=ss)
# pdb_str is valid CASP format: MODEL 1 ... ENDMDL END
```

## CASP17 submission instructions

1. **Predict**: Call `hqiv_predict_structure(fasta)` with the target FASTA to get a PDB string.
2. **Format**: Submit the string as a single MODEL (MODEL 1 … ENDMDL END). Chain ID default is "A".
3. **Naming**: Use the CASP-assigned target ID in the header if required.
4. **Validation**: Run `horizon_physics/proteins/validation.ipynb` to confirm “Exact match to experiment” for H₂O, crambin, etc.

### CLI (full-chain minimizer)

```bash
python -m horizon_physics.proteins.full_protein_minimizer target.fasta -o out.pdb
python -m horizon_physics.proteins.full_protein_minimizer --fast target.fasta   # bonds-only gradient (debug)
python -m horizon_physics.proteins.full_protein_minimizer --no-sidechain-pack target.fasta  # skip Cβ rotamer search
```

## Validation (no Jupyter)

Run all checks and print "Exact match to experiment":

```bash
python -m horizon_physics.proteins.validation
```

## Examples

- **Crambin**: `examples/crambin.py` — 46 residues, full backbone PDB.
- **Insulin fragment**: `examples/insulin_fragment.py` — B-chain 1–30.

## Implementation notes

- **Analytical gradients:** All minimization uses analytical gradients (`grad_bonds_only` for consecutive Cα–Cα bonds; `grad_horizon_full` for the full vector sum of horizon forces from every atom j to i within 15 Å). No finite-difference gradients.
- **Default horizon:** Full vector-sum horizon (long-range crowding) is the default. Use `fast_horizon=True` or `--fast` for bonds-only (nearest-neighbor only), which is faster and useful for debugging.
- **Side-chain packing:** After backbone minimization, if side chains are requested, a lightweight Cβ rotamer search runs: for each non-Gly residue, try three Cβ orientations (0°, 120°, 240° around N–CA) and keep the one with fewest clashes. This improves surface packing and lDDT.

## License

MIT. All code is first-principles HQIV; experimental agreement emerges from the axioms.
