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
- **Side-chain pack:** If `include_sidechains=True`, rotamer packing runs after backbone: χ1 from `side_chain_chi_preferences()` (grid pref±120°, ±60°, 0°) then 5 L-BFGS steps on χ1 to minimize clash energy. Disable with `side_chain_pack=False`. Optional standalone: `pack_sidechains(result)`.

### Current algorithm (full-chain pipeline)

The pipeline used by `minimize_full_chain` is:

1. **Sequence** — Parse FASTA or use the one-letter sequence; validate against the 20 standard amino acids.

2. **Secondary structure** — Predict per-residue SS (H/E/C) from Θ_eff and E_tot(φ,ψ) basins (no ML). Optional window majority vote. If no `ss_string` is provided, `predict_ss(sequence)` is used.

3. **Initial Cα placement** — Place Cα segment-by-segment from N→C:
   - **H**: helix geometry (rise 3/2 Å, pitch 27/5 Å from HQIV).
   - **E/C**: extended/coil (rise 3.2 Å / 3.0 Å).
   - Segments are stitched by aligning the segment start direction to the previous segment end; no bond-length sweep at this stage.

4. **Cα minimization** — Minimize an effective energy (bond penalty + horizon forces) over the Cα trace:
   - **Bonds:** Consecutive Cα–Cα penalized outside [2.5, 6.0] Å; soft minimum near 3.8 Å. Gradient from `build_bond_poles` (analytical).
   - **Horizon:** Full vector sum of repulsive forces from every atom j to i within horizon radius (12–15 Å). Each pair contributes a **bond potential vector (pole)** (i→j); gradient from `build_horizon_poles` or `grad_horizon_full`. A **neighbor list** (12 Å) limits pairs to nearby atoms for speed.
   - **Combined gradient:** Analytical only (`grad_bonds_only` + `grad_horizon_full`); no finite differences. Optional `fast_horizon`: use bonds-only (nearest-neighbor) for debugging.
   - **Step:** Gradient descent with adaptive step; after each step, **project** consecutive bonds into [2.5, 6.0] Å.
   - **Paths:** For n_res > 50, a fast path (bonds + horizon, ~30 iter) is used; for shorter chains, L-BFGS with the same analytical gradient and bond projection.

5. **Backbone rebuild** — From minimized Cα, place N, CA, C, O per residue using local tangent and HQIV bond lengths (N–CA, CA–C, C–O).

6. **Side chains (optional)** — If `include_sidechains=True`: add Cβ from N–CA–C plane (trans); then **side-chain packing**: χ1 from `side_chain_chi_preferences()`, grid (pref ±120°, ±60°, 0°), then a short L-BFGS on the χ1 vector to minimize Cβ clash energy.

7. **Output** — Result dict (e.g. `ca_min`, `backbone_atoms`, energies, `info`); `full_chain_to_pdb(result)` writes MODEL 1 … ENDMDL END.

Poles (bond potential vectors) can be stored per minimization step: `build_horizon_poles`, `build_bond_poles`, and `grad_from_poles` let you keep and reuse the ± vectors per (i, j).

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

### Bond potential vectors (poles)

Each horizon or chain bond is represented as a **pole**: a bond potential vector with a ± convention. A pole is `(i, j, vec)` where `vec` points from atom `i` toward atom `j`; the force on `i` is −`vec`, on `j` is +`vec`. You can build and store poles for reuse (e.g. per minimization step):

```python
from horizon_physics.proteins import build_horizon_poles, build_bond_poles, grad_from_poles

# Horizon poles (all pairs within cutoff)
poles_h = build_horizon_poles(positions, z_list)
# Chain bonds only (consecutive Cα–Cα)
poles_b = build_bond_poles(positions)
# Reconstruct gradient from any list of poles
grad = grad_from_poles(poles_b + poles_h, n)
```

`grad_horizon_full(..., return_poles=True)` returns `(grad, poles)` so you can keep the horizon poles for the current step.

## Implementation notes

- **Analytical gradients:** All minimization uses analytical gradients (`grad_bonds_only` for consecutive Cα–Cα bonds; `grad_horizon_full` for the full vector sum of horizon forces from every atom j to i within 15 Å). No finite-difference gradients.
- **Default horizon:** Full vector-sum horizon (long-range crowding) is the default. Use `fast_horizon=True` or `--fast` for bonds-only (nearest-neighbor only), which is faster and useful for debugging.
- **Side-chain packing:** After backbone minimization, χ1 is optimized using `side_chain_chi_preferences()`: grid search (pref ±120°, ±60°, 0°) then L-BFGS on the χ1 vector to minimize clash energy. Uses existing geometry (N–CA–Cβ). Improves lDDT and all-atom RMSD; run standalone with `pack_sidechains(result)` on an existing result.

## License

MIT. All code is first-principles HQIV; experimental agreement emerges from the axioms.
