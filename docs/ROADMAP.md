## Roadmap

- **Temperature-aware folding / “body → cryo” schedule**  
  - Today the pipelines (single-chain, tunnel, assembly, ligands) are pure deterministic minimizers on \(E_\text{tot}\) (gradient descent / L‑BFGS style), without any explicit temperature parameter or annealing schedule. There is no distinction between “body temperature” folding and a final “cryo” quench; all stages are effectively \(T \to 0\) searches of the same energy landscape.  
  - Roadmap: introduce a temperature-like schedule to better mimic biology + crystallography:
    - Early “folding” phase that behaves more like body temperature (e.g. stochastic perturbations, simulated annealing, or temperature-scaled clash/horizon terms).
    - Late “freeze-out” phase tuned for cryo / crystallographic conditions (stronger clash, reduced noise, tighter convergence) before final PDB export.
    - Expose knobs in the API (e.g. `fold_temperature`, `cryo_refine=True`, or a small preset enum) and document in `/help` and the README.

