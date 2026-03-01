# Funnel and ribosome surface as null search space

The folding funnel and ribosome exit tunnel are approximated as a **null search space**: the most natural bound on where the chain can explore. This gives a three-phase progression that matches physical emergence:

1. **Helical structures form** — Within the funnel/tunnel bound, local geometry (Ramachandran minima, horizon, bonds) favors alpha helices and secondary structure. The null search space confines the chain so that helical turns are the natural first outcome.

2. **Larger translations by HKE group** — Once secondary structure is present, the hierarchical kinematic engine operates at the **group** level: 6-DOF (translation + rotation) per residue or per segment. The funnel still bounds the search; group-level moves explore compact arrangements within that bound.

3. **At exit: complex wraps into itself** — When the chain exits the tunnel (or the bound is relaxed), the search space allows **tertiary** folding: the chain can wrap back on itself, form domain contacts, and reach the native compact state. Stage 3 (flat Cartesian refinement) in HKE corresponds to this exit regime, where full positional freedom permits complex self-wrapping.

## Implementation mapping

| Phase              | Null search space      | PROtein / HKE analogue                          |
|--------------------|------------------------|--------------------------------------------------|
| Helices form       | Funnel/tunnel interior | Stage 1 init (compact start), φ/ψ in alpha basin |
| Group translations | Same bound             | Stage 1–2: DOF minimization (6-DOF + φ, ψ)      |
| Exit / wrap        | Bound relaxed or exit | Stage 3: Cartesian refinement, full horizon      |

**Implementation:** A soft **cone** funnel is available: axis = line from first to last group COM; the allowed radius grows linearly along the axis from `funnel_radius` (narrow end) to `funnel_radius_exit` (exit). Each group COM is penalized if its distance to the axis exceeds the local cone radius (penalty = `funnel_stiffness * (d - R(t))^2` when d > R(t)). Default: `funnel_radius_exit = 2 * funnel_radius` (cone); set `funnel_radius_exit=funnel_radius` for a cylinder. Used in stages 1–2 only; stage 3 has no funnel (exit). Pass `funnel_radius=10.0` (Å) and optional `funnel_stiffness=1.0`, `funnel_radius_exit=20.0` to `minimize_full_chain_hierarchical(...)` or `protein.minimize_hierarchical(...)`.

## References

- Ribosome exit tunnel: confined volume in which the nascent chain is sterically bounded.
- Folding funnel: energy landscape metaphor; here we treat the *accessible volume* as the null search space (natural bound) so that helices → group moves → exit → wrap is the natural order.
