# pyhqiv integration for PROtein / molecular HQIV

PROtein uses HQIV for protein folding. When **pyhqiv** is installed, `_hqiv_base` delegates all metric math to pyhqiv:

- **φ = 2/Θ:** `pyhqiv.utils.phi_from_theta_local(theta, c=1)`
- **γ:** `pyhqiv.constants.GAMMA`
- **Θ(Z, coord):** `pyhqiv.utils.theta_local`, `theta_for_atom`
- **r_eq(Θ_i, Θ_j):** `pyhqiv.utils.bond_length_from_theta`
- **|f_φ|:** `pyhqiv.utils.damping_force_magnitude`
- **Constants:** `pyhqiv.constants.HBAR_C_EV_ANG`, `A_LOC_ANG`

If pyhqiv is not installed, local fallbacks are used so PROtein still runs. The following section describes the features that were recommended and are now provided by pyhqiv.

---

## 1. Lattice Θ from Z and coordination (molecular / diamond lattice)

**In pyhqiv:** `pyhqiv.utils.theta_local`, `theta_for_atom`. PROtein uses them when available.

*Original recommendation:*

- **Θ_local(Z, coordination)** — diamond size in Å from nuclear charge (Z shell) and coordination (monogamy):  
  `Θ ∝ Z^{-α} / coordination^{1/3}` with α ≈ 0.91, reference set so that Θ_C(coord=2) ≈ 1.53 Å, Θ_N(coord=2) ≈ 1.33 Å.

**Recommendation:** Add to pyhqiv (e.g. `pyhqiv.utils` or a small `pyhqiv.molecular` / `pyhqiv.lattice_theta`):

```python
def theta_local(z_shell: int, coordination: int = 1, alpha: float = 0.91, theta_ref_ang: float = 1.53) -> float:
    """
    Diamond size Θ_local (Å) at a lattice node from shell number and monogamy.
    Θ = theta_ref * (6^alpha * 2^(1/3)) * Z^{-alpha} / coordination^{1/3}.
    """
```

- **Units:** length in Å; `z_shell` = nuclear charge (e.g. C=6, N=7, O=8).
- **Optional:** `theta_for_atom(symbol: str, coordination: int = 1) -> float` using a small Z map (H=1, C=6, N=7, O=8, S=16) so callers can pass element symbol.

This keeps the paper’s lattice/monogamy derivation in one place and lets PROtein (and others) drop local copies.

---

## 2. Equilibrium bond length from Θ (diamond overlap)

**In pyhqiv:** `pyhqiv.utils.bond_length_from_theta`. PROtein uses it when available.

*Original recommendation:*

- **r_eq(Θ_i, Θ_j, monogamy_factor)** — equilibrium separation between two nodes:  
  `r_eq = min(Θ_i, Θ_j) * monogamy_factor` (causal diamond containing both atoms).

**Recommendation:** Add to the same pyhqiv module:

```python
def bond_length_from_theta(theta_i: float, theta_j: float, monogamy_factor: float = 1.0) -> float:
    """Equilibrium separation (Å) between two nodes: r_eq = min(Θ_i, Θ_j) * monogamy_factor."""
```

- **Units:** inputs and output in Å.

---

## 3. Geometric damping force magnitude

**In pyhqiv:** `pyhqiv.utils.damping_force_magnitude`. PROtein uses it when available.

*Original recommendation:*

- **f_φ magnitude:** `γ * φ * |∇φ| / (a_loc + φ/6)²` (paper’s geometric damping).

**Recommendation:** Add to pyhqiv (e.g. `utils` or `phase`), so all “φ + γ” metric machinery lives in pyhqiv:

```python
def damping_force_magnitude(phi: float, grad_phi: float, a_loc: float = 1.0, gamma: float = None) -> float:
    """|f_φ| = γ * φ * |∇φ| / (a_loc + φ/6)². Default gamma from pyhqiv.constants.GAMMA."""
```

- **Units:** φ and a_loc consistent (e.g. φ = 2/Θ with Θ in Å); grad_phi = |∇φ| in Å^{-1}; return value = force magnitude (e.g. eV/Å).

This would allow PROtein to call a single pyhqiv function instead of reimplementing the formula.

---

## 4. Constants

**In pyhqiv:** `pyhqiv.constants.HBAR_C_EV_ANG`, `A_LOC_ANG`. PROtein uses them when available.

*Original recommendation:*

- **HBAR_C_EV_ANG** ≈ 1973.27 (eV·Å) — for E = ħc/Θ when Θ is in Å (informational energy).
- **A_LOC_ANG** = 1.0 (Å) — reference scale in the damping denominator (a_loc).

Adding these (or a single “molecular” / “protein” constants dict) would allow PROtein to rely on pyhqiv for both φ/γ and for ħc and a_loc when computing energies and forces.

---

## 5. Summary table

| Feature | PROtein currently | Recommended pyhqiv addition |
|--------|--------------------|-----------------------------|
| φ = 2/Θ | Uses `phi_from_theta_local` ✓ | Already present |
| γ | Uses `GAMMA` ✓ | Already present |
| Θ(Z, coord) | Local `theta_local` | `theta_local(z_shell, coordination, ...)` |
| Θ(symbol) | Local `theta_for_atom` | Optional `theta_for_atom(symbol, coordination)` |
| r_eq(Θ_i, Θ_j) | Local `bond_length_from_theta` | `bond_length_from_theta(theta_i, theta_j, monogamy_factor)` |
| Damping \|f_φ\| | Local `damping_force_magnitude` | `damping_force_magnitude(phi, grad_phi, a_loc, gamma)` |
| ħc (eV·Å), a_loc | Local constants | Optional in `constants` or `constants.molecular` |

---

## 6. Suggested placement in pyhqiv

- **Option A:** Add `theta_local`, `bond_length_from_theta`, and (optionally) `theta_for_atom` to **`pyhqiv.utils`**; add `damping_force_magnitude` there or in **`pyhqiv.phase`**; add `HBAR_C_EV_ANG` and `A_LOC_ANG` to **`pyhqiv.constants`**.
- **Option B:** New submodule **`pyhqiv.molecular`** (or **`lattice_theta`**) for Z/coordination Θ, bond length from Θ, and element symbol → Θ, with constants in that module or in `constants`; keep `damping_force_magnitude` in `utils` or `phase` next to φ/γ.

PROtein’s `_hqiv_base` now imports these from pyhqiv when the package is installed and uses local fallbacks otherwise.
