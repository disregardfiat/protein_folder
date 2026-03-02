# PROtein — HQIV Protein Folding

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

### Optional: JAX and TPU

For the hierarchical minimizer and TPU/GPU acceleration:

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-tpu.txt   # jax, jaxlib
```

On a **Google Cloud TPU VM**, install TPU support so JAX sees the device:

```bash
pip install "jax[tpu]"
# or if libtpu download fails: pip install libtpu -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Co-translational ribosome tunnel (optional)

Simulate the ribosome exit tunnel: null search cone, plane at the lip, fast-pass spaghetti (rigid group + bell-end translations), and a short HKE min pass on each connection. With `post_extrusion_refine=True` (default), the full HKE collapse/refine is run repeatedly after extrusion (no cone/plane) until two consecutive runs produce the same structure.

```bash
python -c "
from horizon_physics.proteins import minimize_full_chain, full_chain_to_pdb
result = minimize_full_chain(
    'MKFLNDR',
    simulate_ribosome_tunnel=True,
    tunnel_length=25.0,
    cone_half_angle_deg=12.0,
)
print(full_chain_to_pdb(result))
"
```

## Package

- **`horizon_physics/proteins/`** — Full HQIV protein folding pipeline
- See [horizon_physics/proteins/README.md](horizon_physics/proteins/README.md) for details

## License

MIT
