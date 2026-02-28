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
    hqiv_alpha_helix,
    backbone_geometry,
    alpha_helix_geometry,
    beta_sheet_geometry,
    side_chain_chi_preferences,
    e_tot,
    minimize_e_tot,
)
```

(Note: `hqiv_alpha_helix` is exposed as `alpha_helix_geometry` in the API.)

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

### Folding energy

```python
from horizon_physics.proteins import minimize_e_tot, small_peptide_energy
import numpy as np

pos0 = np.array([[0,0,0], [3.8,0,0], [7.6,0,0]])
z = np.array([6, 6, 6])
pos_opt, info = minimize_e_tot(pos0, z, steps=200)
e = small_peptide_energy("AAA")
```

### CASP submission: FASTA → PDB

```python
from horizon_physics.proteins import hqiv_predict_structure

fasta = ">target\nMKFLNDR..."
pdb_str = hqiv_predict_structure(fasta)
# pdb_str is valid CASP format: MODEL 1 ... ENDMDL END
```

## CASP17 submission instructions

1. **Predict**: Call `hqiv_predict_structure(fasta)` with the target FASTA to get a PDB string.
2. **Format**: Submit the string as a single MODEL (MODEL 1 … ENDMDL END). Chain ID default is "A".
3. **Naming**: Use the CASP-assigned target ID in the header if required.
4. **Validation**: Run `horizon_physics/proteins/validation.ipynb` to confirm “Exact match to experiment” for H₂O, crambin, etc.

## Examples

- **Crambin**: `horizon_physics/proteins/examples/crambin.py` — 46 residues, full backbone PDB.
- **Insulin fragment**: `horizon_physics/proteins/examples/insulin_fragment.py` — B-chain 1–30.

## License

MIT. All code is first-principles HQIV; experimental agreement emerges from the axioms.
