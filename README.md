# PROtien — HQIV Protein Folding

First-principles protein structure prediction from the **Horizon-centric Quantum Information and Vacuum (HQIV)** framework. No empirical force fields, no PDB statistics.
HQIV => https://zenodo.org/records/18794890

## Quick start

```bash
pip install numpy  # Python 3.10+
python -c "
from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb
result = minimize_full_chain('MKFLNDR', include_sidechains=True)
print(full_chain_to_pdb(result))
"
```

## Package

- **`horizon_physics/proteins/`** — Full HQIV protein folding pipeline
- See [horizon_physics/proteins/README.md](horizon_physics/proteins/README.md) for details

## License

MIT
