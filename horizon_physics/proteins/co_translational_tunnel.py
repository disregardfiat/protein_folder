"""
Co-translational folding: ribosome exit tunnel cone + lip plane constraints.

Simulates the ribosome exit tunnel: (1) null search cone so residues inside the
tunnel stay within a conical volume; (2) hard plane at the tunnel lip so
rotations that would drive the chain back through the lip are nullified.
Used for fast-pass spaghetti building (rigid-group + bell-end only large trans)
and connection-triggered HKE min passes. JAX-portable logic (same ops work with
jnp; this implementation uses numpy to match full_protein_minimizer).

MIT License. Python 3.10+. Numpy.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

# Default tunnel axis is +Z (extrusion direction)
DEFAULT_TUNNEL_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    a = np.asarray(axis, dtype=float).ravel()
    n = np.linalg.norm(a)
    if n < 1e-12:
        return DEFAULT_TUNNEL_AXIS.copy()
    return a / n


def inside_tunnel_mask(
    positions: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
) -> np.ndarray:
    """
    Boolean mask: True for residues whose Cα lies inside the tunnel (distance
    along axis from PTC in [0, tunnel_length]). Positions (n, 3), axis unit.
    """
    axis = _normalize_axis(axis)
    n = positions.shape[0]
    s = (positions - ptc_origin) @ axis  # (n,)
    return (s >= -1e-9) & (s <= tunnel_length + 1e-9)


def past_lip_mask(
    positions: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    lip_distance: float,
) -> np.ndarray:
    """True for residues past the tunnel lip (s > lip_distance along axis)."""
    axis = _normalize_axis(axis)
    s = (positions - ptc_origin) @ axis
    return s > lip_distance + 1e-9


def cone_constraint_mask_gradient(
    grad: np.ndarray,
    positions: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    cone_half_angle_deg: float,
    tunnel_length: float,
) -> None:
    """
    In-place: zero the component of grad that would push a Cα outside the cone.
    Residues inside the tunnel (s <= tunnel_length): radial outward component
    beyond the cone boundary is zeroed. Cone originates at PTC; half-angle in degrees.
    """
    axis = _normalize_axis(axis)
    half_angle_rad = np.deg2rad(cone_half_angle_deg)
    tan_alpha = np.tan(half_angle_rad)
    n = positions.shape[0]
    for i in range(n):
        v = positions[i] - ptc_origin
        s = float(np.dot(v, axis))
        if s < -1e-9 or s > tunnel_length + 1e-9:
            continue
        r_parallel = s * axis
        r_perp_vec = v - r_parallel
        r_perp = np.linalg.norm(r_perp_vec)
        r_max = max(s, 1e-9) * tan_alpha
        if r_perp < 1e-12:
            continue
        radial_unit = r_perp_vec / r_perp
        if r_perp >= r_max - 1e-9:
            # Zero outward radial component of gradient
            out_comp = np.dot(grad[i], radial_unit)
            if out_comp > 0:
                grad[i] -= out_comp * radial_unit


def plane_lip_null_backward_gradient(
    grad: np.ndarray,
    positions: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    lip_distance: float,
) -> None:
    """
    In-place: nullify gradient components that would move residues back across
    the lip plane (unphysical re-entry). For any residue past the lip, zero the
    component of grad in the -axis direction (toward PTC).
    """
    axis = _normalize_axis(axis)
    n = positions.shape[0]
    s = (positions - ptc_origin) @ axis
    for i in range(n):
        if s[i] <= lip_distance + 1e-9:
            continue
        ax_comp = np.dot(grad[i], axis)
        if ax_comp < 0:
            grad[i] -= ax_comp * axis


def apply_cone_and_plane_masking(
    grad: np.ndarray,
    positions: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
    cone_half_angle_deg: float,
    lip_plane_distance: float = 0.0,
) -> None:
    """
    Apply both cone constraint (for residues inside tunnel) and plane-at-lip
    (null backward motion for residues past lip). Modifies grad in place.
    lip_plane_distance: lip is at ptc_origin + (tunnel_length + lip_plane_distance) * axis.
    """
    cone_constraint_mask_gradient(
        grad, positions, ptc_origin, axis, cone_half_angle_deg, tunnel_length
    )
    lip_distance = tunnel_length + lip_plane_distance
    plane_lip_null_backward_gradient(grad, positions, ptc_origin, axis, lip_distance)


def zero_gradient_below_tunnel_fraction(
    grad: np.ndarray,
    positions: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
    fraction: float,
) -> None:
    """
    Zero gradient for residues at or below fraction of tunnel length (s <= fraction * tunnel_length).
    Use so HKE minimization only updates the chain above this point (e.g. fraction=0.5 → HKE only above 50% of tunnel).
    """
    axis = _normalize_axis(axis)
    s = (positions - ptc_origin) @ axis
    threshold = fraction * tunnel_length
    for i in range(grad.shape[0]):
        if s[i] <= threshold + 1e-9:
            grad[i] = 0.0


def rigid_body_gradient_for_group(
    positions: np.ndarray,
    grad: np.ndarray,
    indices: List[int],
) -> None:
    """
    Replace grad[indices] with a single rigid-body gradient (6-DOF: translation + rotation)
    so that the group moves as one. Modifies grad in place for indices.
    F = sum(grad[i]), T = sum((pos[i]-com) × grad[i]), I = inertia tensor; omega = I^{-1} T;
    grad_rigid[i] = F + omega × (pos[i]-com).
    """
    if not indices:
        return
    pos_group = positions[indices]
    grad_group = grad[indices]
    com = np.mean(pos_group, axis=0)
    F = np.sum(grad_group, axis=0)
    r = pos_group - com
    T = np.sum(np.cross(r, grad_group), axis=0)
    I = np.zeros((3, 3))
    for k in range(len(indices)):
        rk = r[k]
        I += (np.dot(rk, rk) * np.eye(3) - np.outer(rk, rk))
    I += 1e-8 * np.eye(3)
    try:
        omega = np.linalg.solve(I, T)
    except np.linalg.LinAlgError:
        omega = np.zeros(3)
    for idx, i in enumerate(indices):
        grad[i] = F + np.cross(omega, r[idx])


def bell_end_indices(n_res: int, n_bell: int = 2) -> List[int]:
    """Indices of the last n_bell residues (bell end; 1 or 2 typically)."""
    if n_res <= 0:
        return []
    return list(range(max(0, n_res - n_bell), n_res))


def run_masked_lbfgs_pass(
    positions: np.ndarray,
    z_list: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
    cone_half_angle_deg: float,
    lip_plane_distance: float,
    max_iter: int,
    grad_func,
    project_bonds,
    r_bond_min: float = 2.5,
    r_bond_max: float = 6.0,
    hke_above_tunnel_fraction: Optional[float] = 0.5,
) -> Tuple[np.ndarray, int]:
    """
    Run one short HKE (L-BFGS) minimization pass with cone and plane gradient
    masking at each step. If hke_above_tunnel_fraction is set (default 0.5),
    gradient is zeroed for residues at or below that fraction of tunnel length,
    so HKE only updates the chain above that point (e.g. above 50% of tunnel).
    Returns (positions_opt, n_iter_used).
    """
    from .folding_energy import e_tot_ca_with_bonds
    from .gradient_descent_folding import _project_bonds, _lbfgs_two_loop

    pos = np.array(positions, dtype=float)
    n = pos.shape[0]
    x = pos.ravel()
    lip_distance = tunnel_length + lip_plane_distance
    use_hke_fraction = hke_above_tunnel_fraction is not None

    def _grad_raw(x_flat: np.ndarray) -> np.ndarray:
        p = x_flat.reshape(n, 3)
        g = grad_func(p, z_list)
        apply_cone_and_plane_masking(
            g, p, ptc_origin, axis, tunnel_length, cone_half_angle_deg, lip_plane_distance
        )
        if use_hke_fraction and hke_above_tunnel_fraction is not None:
            zero_gradient_below_tunnel_fraction(
                g, p, ptc_origin, axis, tunnel_length, hke_above_tunnel_fraction
            )
        return g.ravel()

    s_list: list = []
    y_list: list = []
    m = 10
    gtol = 1e-5
    grad = _grad_raw(x)
    for it in range(max_iter):
        g_norm = np.linalg.norm(grad)
        if g_norm <= gtol:
            return x.reshape(n, 3).copy(), it
        if len(s_list) >= m:
            s_list.pop(0)
            y_list.pop(0)
        if len(s_list) == 0:
            direction = -grad
        else:
            direction = _lbfgs_two_loop(grad, s_list, y_list, m)
        step = 1.0
        e_curr = e_tot_ca_with_bonds(x.reshape(n, 3), z_list)
        c1 = 1e-4
        for _ in range(40):
            x_new = x + step * direction
            pos_new = _project_bonds(
                x_new.reshape(n, 3), r_min=r_bond_min, r_max=r_bond_max
            )
            x_new = pos_new.ravel()
            e_new = e_tot_ca_with_bonds(pos_new, z_list)
            if e_new <= e_curr + c1 * step * np.dot(grad, direction):
                break
            step *= 0.5
        x_prev = x.copy()
        x = x_new
        grad_new = _grad_raw(x)
        s_list.append(x - x_prev)
        y_list.append(grad_new - grad)
        grad = grad_new
    return x.reshape(n, 3).copy(), max_iter


def align_chain_to_tunnel(
    ca_positions: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """
    Align a Cα chain so that the N-terminus is at PTC and the chain extends
    along the given tunnel axis. Uses the first bond direction to define
    rotation. For very short chains (n<2), places only at PTC.
    """
    axis = _normalize_axis(axis)
    n = ca_positions.shape[0]
    if n == 0:
        return np.zeros((0, 3))
    out = np.array(ca_positions, dtype=float)
    if n == 1:
        out[0] = ptc_origin.copy()
        return out
    # First bond direction in reference chain
    first_bond = out[1] - out[0]
    first_bond_norm = np.linalg.norm(first_bond)
    if first_bond_norm < 1e-9:
        first_bond = axis.copy()
    else:
        first_bond = first_bond / first_bond_norm
    # Translate so residue 0 at origin, then rotate first_bond -> axis
    out = out - out[0]
    cos_a = np.dot(first_bond, axis)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    if np.abs(cos_a) < 1 - 1e-6:
        rot_axis = np.cross(first_bond, axis)
        rot_axis_norm = np.linalg.norm(rot_axis)
        if rot_axis_norm > 1e-9:
            rot_axis = rot_axis / rot_axis_norm
            angle = np.arccos(cos_a)
            c, s = np.cos(angle), np.sin(angle)
            R = (
                c * np.eye(3)
                + (1 - c) * np.outer(rot_axis, rot_axis)
                + s * np.array(
                    [
                        [0, -rot_axis[2], rot_axis[1]],
                        [rot_axis[2], 0, -rot_axis[0]],
                        [-rot_axis[1], rot_axis[0], 0],
                    ]
                )
            )
            out = (R @ out.T).T
    out = out + ptc_origin
    return out


def co_translational_minimize(
    ca_init: np.ndarray,
    z_list: np.ndarray,
    ptc_origin: np.ndarray,
    axis: np.ndarray,
    tunnel_length: float,
    cone_half_angle_deg: float,
    lip_plane_distance: float,
    grad_func,
    project_bonds,
    n_bell: int = 2,
    fast_pass_steps_per_connection: int = 5,
    min_pass_iter_per_connection: int = 15,
    r_bond_min: float = 2.5,
    r_bond_max: float = 6.0,
    hke_above_tunnel_fraction: float = 0.5,
) -> Tuple[np.ndarray, dict]:
    """
    Co-translational minimization: fast method to build the chain (rigid group +
    bell-end only, no per-residue HKE inside the tunnel), then connection-triggered
    HKE min pass at each chain length. HKE is applied only above hke_above_tunnel_fraction
    of the tunnel (default 0.5 = 50%): residues below that have gradient zeroed in the
    L-BFGS pass. Chain is built along the tunnel; at each length k we run a fast-pass
    (gradient masked with cone/plane, non-bell replaced by rigid-body gradient) for a
    few steps, then one short masked L-BFGS pass (connection event). Returns (ca_min, info).
    """
    from .folding_energy import e_tot_ca_with_bonds

    pos = align_chain_to_tunnel(ca_init, ptc_origin, axis)
    n = pos.shape[0]
    lip_distance = tunnel_length + lip_plane_distance
    total_min_steps = 0

    for k in range(2, n + 1):
        pos_k = pos[:k].copy()
        z_k = z_list[:k]
        inside = inside_tunnel_mask(pos_k, ptc_origin, axis, tunnel_length)
        past_lip = past_lip_mask(pos_k, ptc_origin, axis, lip_distance)
        bell = set(bell_end_indices(k, n_bell))
        # Rigid group = all except bell-end (inside tunnel + spaghetti move as one)
        rigid_indices = [i for i in range(k) if i not in bell]

        # Fast-pass: a few gradient steps with cone/plane + rigid group
        for _ in range(fast_pass_steps_per_connection):
            grad = grad_func(pos_k, z_k)
            apply_cone_and_plane_masking(
                grad, pos_k, ptc_origin, axis, tunnel_length, cone_half_angle_deg, lip_plane_distance
            )
            rigid_body_gradient_for_group(pos_k, grad, rigid_indices)
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-6:
                break
            step = 0.3 / (g_norm + 1e-9)
            pos_k = pos_k - step * grad
            pos_k = project_bonds(pos_k, r_min=r_bond_min, r_max=r_bond_max)

        # Connection-triggered min pass (short HKE with cone/plane; HKE only above fraction of tunnel)
        pos_k, n_iter = run_masked_lbfgs_pass(
            pos_k,
            z_k,
            ptc_origin,
            axis,
            tunnel_length,
            cone_half_angle_deg,
            lip_plane_distance,
            max_iter=min_pass_iter_per_connection,
            grad_func=grad_func,
            project_bonds=project_bonds,
            r_bond_min=r_bond_min,
            r_bond_max=r_bond_max,
            hke_above_tunnel_fraction=hke_above_tunnel_fraction,
        )
        total_min_steps += n_iter
        pos[:k] = pos_k

    e_final = float(e_tot_ca_with_bonds(pos, z_list))
    info = {
        "e_final": e_final,
        "e_initial": e_final,
        "n_iter": total_min_steps,
        "success": True,
        "message": "Co-translational tunnel (fast-pass + connection-triggered HKE)",
    }
    return pos, info
