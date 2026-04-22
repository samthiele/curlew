"""
Eshelby (1957/59) exterior and interior displacement fields for an oblate
spheroidal inclusion in an infinite isotropic full-space. These can be used to 
model displacement fields associated with faults and dykes and, because the 
ellipsoids have a thickness, kink-folds and some fault-propagation folds.

PHYSICAL SETUP
---------------
An oblate spheroidal region Ω (semi-axes r > t > 0, with r = equatorial
radius and t = polar half-thickness) undergoes a uniform stress-free
eigenstrain (i.e. slip±dilation) ε*_{ij} = ½(v_i n_j + v_j n_i), where:

  n  = unit normal to the flat faces of the spheroid  ("fault orientation")
  v  = slip / Burgers vector direction (per-source magnitude only via ``EshelbyField.weights``)

Special cases:
  v ⊥ n  →  shear fault:  tr(ε*) = v·n = 0,  Lamé pressure term vanishes
  v ∥ n  →  tensile dyke: tr(ε*) = |v|,       Lamé term drives opening

FORMULA
--------
The displacement outside Ω is (Mura 1987):

    8πμ(1−ν) u_i(x)  =  σ*_{ij} · ∂φ/∂x_j

where σ*_{ij} = λ_e tr(ε*) δ_{ij} + 2μ ε*_{ij}  is the eigenstress
(Hooke's law applied to ε*), and φ(x) = ∫_Ω dV'/|x−x'| is the Newtonian
potential of the spheroid (equivalent, curiously, to the gravitational potential at x
due to a uniform-density body filling Ω).  

All geometric complexity collapses into spatial derivatives of this one scalar, 
which has a closed-form expression in terms of the exterior ellipsoidal coordinate λ(x).

COORDINATE FRAMES
------------------
All computation is done in a local frame where ê₃ is aligned with n.  In
this frame the spheroid is "canonical" (flat face perpendicular to ê₃), the
eigenstrain tensor is diagonal-like, and the λ(x) formula is standard rotation matrix R 
(shape 3×3, rows = local basis vectors expressed in
global coordinates), which maps:

    x_local  =  R  @  x_global          (global → local)
    u_global =  Rᵀ @  u_local           (local  → global)

IMPLEMENTATION NOTES
---------------------
Key precomputations done at construction (not per-call):

  _r2, _t2, _r2t2, _r2_t2 — scalar powers of r and t used in every λ solve
  _D_floor                 — tip-line ill-conditioning floor for D
  _4pi_r2t                 — numerator constant in dφ/dλ
  _int_scale               — (3,) tensor [-I₁, -I₁, -I₃] for interior ∇φ
  _sig_scaled              — eigenstress pre-divided by 8πμ(1−ν); the Mura formula hot path is then just one matmul

REFERENCES
-----------
Eshelby (1957) Proc. R. Soc. A 241
Eshelby (1959) Proc. R. Soc. A 252
Mura (1987)    Micromechanics of Defects in Solids (Martinus Nijhoff)
Ju & Sun (1999) J. Appl. Mech. 66, 570
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import curlew
from curlew import _tensor, _numpy
from curlew.fields import BaseAF

#  Module-level geometry helpers (pure numpy as these are called only once at construction)
def _build_rotation(n, stretch=0.5):
    """
    Build the 3×3 rotation matrix R (numpy) mapping global → local frame,
    where the local ê₃ axis is aligned with n_hat.

    Construction (Gram-Schmidt):
      1. Choose a reference vector not parallel to n_hat.
      2. ê₁ = ref × n_hat  (first tangent axis)
      3. ê₂ = n_hat × ê₁  (completes right-handed triad)
      4. Rows of R = [ê₁, ê₂, n_hat]  →  R @ v_global = v_local

    Parameters
    ----------
    n : (3,) numpy array — unit normal (need not be pre-normalised)
    stretch : scaling factor applied in the normal direction, to fudge (increase or decrease) the reach
              of deformation associated with an Eshelby inclusion. Larger values increase the reach.
    Returns
    -------
    R : (3, 3) numpy array
    """
    n   = n / np.linalg.norm(n)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(ref, n);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(n,   e1); e2 /= np.linalg.norm(e2)
    return np.array([e1, e2, n/stretch])

def _broadcast_param_1d(param, n: int, name: str) -> np.ndarray:
    """Scalar or length-n float array → (n,) float64."""
    a = np.asarray(param, dtype=np.float64)
    if a.shape == ():
        return np.full(n, float(a), dtype=np.float64)
    if a.shape == (n,):
        return a
    raise ValueError(f"{name} must be a scalar or shape ({n},); got {a.shape}")

def _single_eshelby_tensors(
    position,
    normal,
    slip,
    radius,
    thickness,
    *,
    stretch,
    mu,
    nu,
    far_radius_mult,
):
    """
    Per-source tensors and scalars for one oblate spheroid. Returns (R, RT, pos,
    sig_scaled, int_scale, r2, t2, r2t2, r2_t2, D_floor, _4pi_r2t, far_r2_cutoff, n_hat_g).
    Slip *direction* is encoded here; overall magnitude is applied via ``weights`` on ``EshelbyField``.
    """
    radius = float(radius)
    thickness = float(thickness)
    mu = float(mu)
    nu = float(nu)
    far_radius_mult = float(far_radius_mult)
    stretch = float(stretch)

    if not (thickness > 0.0 and radius > thickness):
        raise ValueError(f"Need 0 < thickness < radius; got thickness={thickness}, radius={radius}")
    if mu <= 0.0:
        raise ValueError(f"mu must be positive; got {mu}")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"nu must be in (-1, 0.5); got {nu}")
    if far_radius_mult <= 0:
        raise ValueError(f"far_radius_mult must be positive; got {far_radius_mult}")

    r2 = radius ** 2
    t2 = thickness ** 2
    r2t2 = r2 + t2
    r2_t2 = r2 * t2
    D_floor = (thickness / radius) ** 4 * 0.25
    _4pi_r2t = 4.0 * math.pi * r2 * thickness
    far_r2_cutoff = (far_radius_mult * radius) ** 2

    n_np = _numpy(normal)
    n_np = n_np / np.linalg.norm(n_np)
    R_np = _build_rotation(n_np, stretch)
    R = _tensor(R_np)
    RT = _tensor(R_np.T)
    pos = _tensor(np.asarray(position, np.float64))

    n_loc = _tensor([0.0, 0.0, 1.0]) # normal of fault in local frame
    v_np = _numpy(slip)
    v_np = v_np / np.linalg.norm(v_np)
    v_loc = R @ _tensor(v_np)
    tr_eps = float(np.dot(R_np @ v_np, [0.0, 0.0, 1.0]))
    eps_loc = (torch.outer(v_loc, n_loc) + torch.outer(n_loc, v_loc)) * (0.5 / (2.0 * thickness))

    delta = math.sqrt(r2 - t2)
    at = math.atan(delta / thickness)
    C = 4.0 * math.pi * r2 * thickness
    I1 = C * (at / delta**3 - thickness / (delta**2 * r2))
    I3 = C * (2.0 / delta**2) * (1.0 / thickness - at / delta)
    int_scale = _tensor([-I1, -I1, -I3])

    lame = 2.0 * mu * nu / (1.0 - 2.0 * nu)
    lame_tr = float(lame * tr_eps)
    sig_loc = _tensor(torch.eye(3)) * lame_tr + eps_loc * (2.0 * mu)
    norm_factor = 8.0 * math.pi * mu * (1.0 - nu)
    sig_scaled = sig_loc / norm_factor

    # ── Normalise so max(pole, equatorial) displacement magnitude = 1 ──────
    # Both candidates use the interior formula: grad_phi = x_loc * [-I1, -I1, -I3]
    int_scale_np = np.array([-I1, -I1, -I3])

    # Candidate 1: pole at x_loc = [0, 0, t*(1-eps)]
    eps = 1e-4
    pole_pt      = np.array([0.0, 0.0, thickness * (1 - eps)])
    pole_grad    = pole_pt * int_scale_np                          # [0, 0, -I3*t*(1-eps)]
    u_pole_loc   = _numpy(sig_scaled) @ pole_grad
    u_pole_global = R_np.T @ u_pole_loc
    pole_mag     = float(np.linalg.norm(u_pole_global))

    # Candidate 2: equatorial point at x_loc = [r*(1-eps), 0, 0]
    eq_pt        = np.array([radius * (1 - eps), 0.0, 0.0])
    eq_grad      = eq_pt * int_scale_np                            # [-I1*r*(1-eps), 0, 0]
    u_eq_loc     = _numpy(sig_scaled) @ eq_grad
    u_eq_global  = R_np.T @ u_eq_loc
    eq_mag       = float(np.linalg.norm(u_eq_global))

    norm_mag = max(pole_mag, eq_mag)
    if norm_mag > 0.0:
        # avoid inncorectly large displacements that arise from numerical nastyness near tips
        # (and ensure largest displacement is 1 so that we get magnitude info entirely from the weights)
        sig_scaled = sig_scaled / norm_mag
            
    n_hat_g = _tensor(n_np)
    return R, RT, pos, sig_scaled, int_scale, r2, t2, r2t2, r2_t2, D_floor, _4pi_r2t, far_r2_cutoff, n_hat_g


class EshelbyField( BaseAF ):
    """
    Vectorised superposition of oblate spheroidal Eshelby inclusions.

    At construction all per-source tensors are stacked into batched
    (n_sources, ...) representations.  Displacement evaluation loops only
    over *receiver* chunks (batch_size, typically 256–512) so peak memory
    is O(n_sources × batch_size) rather than O(n_sources × m).

    Pass ``positions``, ``normals``, ``slips``, ``radii``, ``thicknesses``,
    and constant ``mu`` / ``nu`` via the constructor (keyword arguments to
    ``initField``).  ``radii``, ``thicknesses``, ``stretch``, ``far_radius_mult``,
    and ``weights`` may be scalars broadcast to every source.

    Parameters
    ----------
    positions : (n, 3) array-like
        Inclusion centroids in global coordinates.
    normals : (n, 3) or (3,) array-like
        Fault/dyke normals; broadcast if shape ``(3,)``.
    slips : (n, 3) or (3,) array-like
        Slip (Burgers) *directions*; broadcast if shape ``(3,)``.  Each vector
        is normalised for the eigenstrain; slip **magnitude** (and any taper)
        is entirely in ``weights``.
    radii : float or (n,) array-like
        Equatorial semi-axes r (must satisfy r > t).
    thicknesses : float or (n,) array-like
        Polar half-thicknesses t.
    mu, nu : float
        Shear modulus and Poisson ratio (shared by all sources).
    stretch : float or (n,) array-like, optional
        Normal-direction stretch for ``_build_rotation`` (default 1).
    far_radius_mult : float or (n,) array-like, optional
        Spherical influence cutoff uses ``far_radius_mult[i] * r[i]`` unless
        overridden in ``influence_mask`` / ``evaluate``.
    weights : float, (n,) array-like, or None
        Per-source multiplier on displacement (slip magnitude, Gaussian taper,
        etc.).  ``None`` is equivalent to ``1.0``.  A scalar is broadcast to
        all ``n`` sources.  If ``learnable_weights`` is True, initial values
        are taken from this array (after broadcasting) and stored as
        ``nn.Parameter``; otherwise as a buffer.
    learnable_weights : bool, optional
        If True, ``weights`` is registered as a trainable ``nn.Parameter``.
        Default False.
    max_ram_mb : float
        Receiver chunk budget for ``displacement`` / ``evaluate``.
    n_taper : int, optional
        Number of concentric sub-ellipses used to approximate a smooth rim
        taper (radii spaced from 0.5r to r, cosine-bell weights).  1 disables
        tapering (single full-size ellipsoid).  Default 3.
    plastic_damp : bool, optional
        If True, suppress displacement components perpendicular to the slip
        direction near the ellipsoid boundary, mimicking plastic yielding.
        Default True.
    plastic_xi_lo, plastic_xi_hi : float, optional
        Smoothstep bounds (in ellipsoidal coordinate xi) over which the
        perpendicular damping is blended from 0 (fully damped) to 1 (elastic).
        Defaults 0.7 and 1.5.
    linear_decay : float, optional
        Linear decay correction factor. 0.0 disables, 1.0 is full correction
        making displacement decay as 1/r rather than 1/r**2. Essentially reduces the 
        near-field deformation relative to the far field. 
    Notes
    -----
    Stacked tensors (curlew.dtype on curlew.device):

        _R            (n, 3, 3)   global → local rotation matrices
        _RT           (n, 3, 3)   local → global
        _pos          (n, 3)      source centroids
        _sig_scaled   (n, 3, 3)   pre-normalised eigenstress / (8πμ(1−ν))
        _int_scale    (n, 3)      interior ∇φ scale  [−I₁, −I₁, −I₃]
        _r2           (n,)        equatorial radius²
        _t2           (n,)        polar half-thickness²
        _r2t2         (n,)        r²+t²
        _r2_t2        (n,)        r²·t²
        _D_floor      (n,)        tip-line ill-conditioning floor
        _4pi_r2t      (n,)        dφ/dλ numerator
        _slip_hat     (n, 3)      unit slip directions in global coordinates
        _n_hat        (n, 3)      unit mid-plane normals in global coords

        Taper precomputes — shape (n, T):
        _r2_tap, _t2_tap, _r2t2_tap, _r2_t2_tap, _Df_tap, _pi_tap

        _sig_taper    (n, 3, 3)   sig_scaled fused with taper weights and summed
                                  (only stored when n_taper > 1; replaces per-loop einsum)

        ``weights`` — (n,) ``nn.Parameter`` or buffer.
    """

    def initField (
        self,
        positions,
        normals,
        slips,
        radii,
        thicknesses,
        *,
        mu=1.0,
        nu=0.25,
        stretch=1.0,
        far_radius_mult=20.0,
        weights=None,
        learnable_weights=False,
        max_ram_mb=2048,
        n_taper=3,
        plastic_damp=True,
        plastic_xi_lo=0.7,
        plastic_xi_hi=1.5,
        linear_decay=1.0,
        surface_height=None,
    ):
        positions   = np.asarray(positions,  dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (n, 3); got {positions.shape}")
        n_real = int(positions.shape[0])
        if n_real == 0:
            raise ValueError("positions must be non-empty (n >= 1)")

        normals = np.asarray(normals, dtype=np.float64)
        if normals.shape == (3,):
            normals = np.broadcast_to(normals, (n_real, 3)).copy()
        elif normals.shape != (n_real, 3):
            raise ValueError(f"normals must be (3,) or (n, 3); got {normals.shape}")

        slips = np.asarray(slips, dtype=np.float64)
        if slips.shape == (3,):
            slips = np.broadcast_to(slips, (n_real, 3)).copy()
        elif slips.shape != (n_real, 3):
            raise ValueError(f"slips must be (3,) or (n, 3); got {slips.shape}")

        radii       = _broadcast_param_1d(radii,           n_real, "radii")
        thicknesses = _broadcast_param_1d(thicknesses,     n_real, "thicknesses")
        stretch_a   = _broadcast_param_1d(stretch,         n_real, "stretch")
        frm_a       = _broadcast_param_1d(far_radius_mult, n_real, "far_radius_mult")

        w_arr = _broadcast_param_1d(1.0 if weights is None else weights, n_real, "weights")

        # ── Free-surface mirror sources ────────────────────────────────────────
        use_mirror = surface_height is not None
        if use_mirror:
            zh = float(surface_height)

            # Depth check: remove sources above the free surface
            depths = zh - positions[:, 2] # positive = below surface
            valid  = depths > 0  # centroid must be below surface
            n_removed = int((~valid).sum())
            if n_removed > 0:
                import warnings
                warnings.warn(
                    f"{n_removed} source(s) above or at surface (z={zh}) removed "
                    f"(indices: {np.where(~valid)[0].tolist()})"
                )
                positions   = positions[valid]
                normals     = normals[valid]
                slips       = slips[valid]
                radii       = radii[valid]
                thicknesses = thicknesses[valid]
                stretch_a   = stretch_a[valid]
                frm_a       = frm_a[valid]
                w_arr       = w_arr[valid]
                n_real      = int(valid.sum())
            if n_real == 0:
                raise ValueError("No sources remain after removing above-surface sources.")

            # Mirror: reflect z through surface_height, negate normal
            mirror_positions              = positions.copy()
            mirror_positions[:, 2]        = 2.0 * zh - positions[:, 2]
            mirror_normals                = -normals          # negated → reverses eigenstress
            mirror_slips                  = slips.copy()      # slip direction unchanged
            mirror_weights                = -w_arr            # opposite sign for cancellation

            # Stack real + mirror
            positions   = np.concatenate([positions,   mirror_positions],  axis=0)
            normals     = np.concatenate([normals,      mirror_normals],    axis=0)
            slips       = np.concatenate([slips,        mirror_slips],      axis=0)
            radii       = np.concatenate([radii,        radii],             axis=0)
            thicknesses = np.concatenate([thicknesses,  thicknesses],       axis=0)
            stretch_a   = np.concatenate([stretch_a,    stretch_a],         axis=0)
            frm_a       = np.concatenate([frm_a,        frm_a],             axis=0)
            w_arr       = np.concatenate([w_arr,        mirror_weights],    axis=0)

        n = int(positions.shape[0])   # n_real or 2*n_real

        self.n_sources      = n
        self.n_real_sources = n_real
        self.surface_height = float(surface_height) if use_mirror else None
        self.max_ram_mb     = max_ram_mb
        self._N_TAPER       = max(1, int(n_taper))
        self._plastic_damp  = bool(plastic_damp)
        self._plastic_xi_lo = float(plastic_xi_lo)
        self._plastic_xi_hi = float(plastic_xi_hi)
        self._linear_decay   = float(linear_decay)
        
        R_list, RT_list, pos_list, sig_list, int_list = [], [], [], [], []
        n_hat_list, slip_hat_list = [], []
        r2_list, t2_list, r2t2_list, r2_t2_list = [], [], [], []
        Df_list, pi_list, far_list = [], [], []

        for i in range(n):
            (R, RT, pos, sig_scaled, int_scale,
            r2, t2, r2t2, r2_t2, D_floor, _4pi_r2t,
            far_c, nhg) = _single_eshelby_tensors(
                positions[i], normals[i], slips[i],
                float(radii[i]), float(thicknesses[i]),
                stretch=float(stretch_a[i]), mu=mu, nu=nu,
                far_radius_mult=float(frm_a[i]),
            )
            R_list.append(R);          RT_list.append(RT)
            pos_list.append(pos);      sig_list.append(sig_scaled)
            int_list.append(int_scale); n_hat_list.append(nhg)
            r2_list.append(r2);        t2_list.append(t2)
            r2t2_list.append(r2t2);    r2_t2_list.append(r2_t2)
            Df_list.append(D_floor);   pi_list.append(_4pi_r2t)
            far_list.append(far_c)

            v_np = np.asarray(slips[i], dtype=np.float64)
            slip_hat_list.append(_tensor(v_np / np.linalg.norm(v_np)))

        self._R          = torch.stack(R_list,        dim=0)
        self._RT         = torch.stack(RT_list,       dim=0)
        self._n_hat      = torch.stack(n_hat_list,    dim=0)
        self._slip_hat   = torch.stack(slip_hat_list, dim=0)
        self._pos        = torch.stack(pos_list,      dim=0)
        self._sig_scaled = torch.stack(sig_list,      dim=0)
        self._int_scale  = torch.stack(int_list,      dim=0)
        self._r2         = _tensor(r2_list)
        self._t2         = _tensor(t2_list)
        self._r2t2       = _tensor(r2t2_list)
        self._r2_t2      = _tensor(r2_t2_list)
        self._D_floor    = _tensor(Df_list)
        self._4pi_r2t    = _tensor(pi_list)
        self._far_r2_cutoff = _tensor(far_list)
        self._radii      = torch.sqrt(self._r2)

        # ── Taper geometry (n includes mirror sources) ─────────────────────────
        T = self._N_TAPER
        taper_r_np          = np.linspace(0.5, 1.0, T) if T > 1 else np.array([1.0])
        self._TAPER_RADII   = _tensor(taper_r_np)
        raw                 = np.cos(taper_r_np * (math.pi / 2.0))
        self._TAPER_WEIGHTS = _tensor(raw / raw.sum())

        r2_tap    = self._r2.unsqueeze(1) * self._TAPER_RADII.unsqueeze(0) ** 2
        t2_tap    = self._t2.unsqueeze(1).expand(-1, T).clone()
        self._r2_tap    = r2_tap
        self._t2_tap    = t2_tap
        self._r2t2_tap  = r2_tap + t2_tap
        self._r2_t2_tap = r2_tap * t2_tap
        r_tap           = torch.sqrt(r2_tap)
        self._Df_tap    = (torch.sqrt(t2_tap) / r_tap) ** 4 * 0.25
        self._pi_tap    = 4.0 * math.pi * r2_tap * torch.sqrt(t2_tap)

        # ── Weights (mirror weights already negated in w_arr) ─────────────────
        w_init = _tensor(w_arr)
        if learnable_weights:
            self.weights = nn.Parameter(w_init.clone())
        else:
            self.register_buffer("weights", w_init, persistent=True)

    # ──────────────────────────────────────────────────────────────────────
    def influence_mask(self, x, k=None):
        """
        True for receivers within spherical distance k*r of *any* source centroid.
        """
        x_t     = _tensor(x)
        squeeze = x_t.shape == (3,)
        if squeeze:
            x_t = x_t.unsqueeze(0)
        if x_t.shape[-1] != 3:
            raise ValueError("receivers must have last dimension 3")
        flat = x_t.reshape(-1, 3)
        dist = torch.cdist(self._pos, flat)
        if k is None:
            thr = torch.sqrt(self._far_r2_cutoff).unsqueeze(1)
        else:
            kk = float(k)
            if kk <= 0:
                raise ValueError(f"k must be positive; got {kk}")
            thr = (kk * self._radii).unsqueeze(1)
        m = (dist <= thr).any(dim=0).reshape(x_t.shape[:-1])
        if squeeze:
            m = m.squeeze(0)
        return m

    # ──────────────────────────────────────────────────────────────────────
    def evaluate(self, x, max_ram_mb=2048, k=None):
        """
        Like ``displacement`` but skips points outside ``influence_mask``.
        """
        return_numpy = not isinstance(x, torch.Tensor)
        x_t     = _tensor(x)
        mask    = self.influence_mask(x_t, k=k)
        squeeze = x_t.shape == (3,)
        if squeeze:
            x_t  = x_t.unsqueeze(0)
            if mask.ndim == 0:
                mask = mask.unsqueeze(0)
        if x_t.shape[-1] != 3:
            raise ValueError("receivers must have last dimension 3")
        batch_shape = x_t.shape[:-1]
        flat    = x_t.reshape(-1, 3)
        mflat   = mask.reshape(-1)
        u_flat  = torch.zeros_like(flat)
        if bool(mflat.any()):
            u_flat[mflat] = self.displacement(flat[mflat], max_ram_mb=max_ram_mb)
        u_out = u_flat.reshape(*batch_shape, 3)
        if squeeze:
            u_out = u_out.squeeze(0)
        return _numpy(u_out) if return_numpy else u_out

    # ──────────────────────────────────────────────────────────────────────
    def displacement(self, x, max_ram_mb=2048):
        """
        Evaluate the total displacement at m receiver positions.

        Corrections applied
        -------------------
        1. Cosine-bell taper over ``n_taper`` concentric sub-ellipses — all
           taper levels are evaluated in a single vectorised pass (no Python
           loop), with the taper dimension T fused into the batch.
        2. Plastic damping — slip-perpendicular displacement suppressed near
           the rim via a smoothstep on xi (the ellipsoidal coordinate).
        3. Equatorial crossing check — per-source displacement rescaled so no
           receiver is moved across the source mid-plane.
        """
        return_numpy = not isinstance(x, torch.Tensor)
        x       = _tensor(x)
        squeeze = x.shape == (3,)
        if squeeze:
            x = x.unsqueeze(0)
        if x.shape[-1] != 3:
            raise ValueError("receivers must have last dimension 3")

        batch_shape = x.shape[:-1]
        x_flat      = x.reshape(-1, 3)
        m           = x_flat.shape[0]

        bytes_per_element = {torch.float16: 2, torch.float32: 4,
                             torch.float64: 8}.get(x_flat.dtype, 4)
        # Memory budget accounts for the extra taper dimension in working tensors
        batch_size = int(max_ram_mb * 1024**2 /
                         (self.n_sources * self._N_TAPER * 3 * bytes_per_element * 2.0))
        batch_size = max(1, min(batch_size, m))
        u_out = torch.zeros_like(x_flat)

        # Taper scalars — all (n, T, 1) for broadcasting against (n, T, B)
        r2_k    = self._r2_tap.unsqueeze(2)       # (n, T, 1)
        t2_k    = self._t2_tap.unsqueeze(2)
        r2t2_k  = self._r2t2_tap.unsqueeze(2)
        r2_t2_k = self._r2_t2_tap.unsqueeze(2)
        Df_k    = self._Df_tap.unsqueeze(2)
        pi_k    = self._pi_tap.unsqueeze(2)

        # Taper weights — (1, T, 1, 1) for broadcasting against (n, T, B, 3)
        w_t = self._TAPER_WEIGHTS[None, :, None, None]

        # Interior scale is taper-independent (thickness fixed) — fuse weights now:
        # sum_k w_k * (x_loc * int_scale) = x_loc * int_scale  (weights sum to 1)
        # So the interior contribution is identical to the single-level case.
        int_scale_4d = self._int_scale[:, None, None, :]  # (n, 1, 1, 3) — broadcast over T & B

        for start in range(0, m, batch_size):
            end   = min(start + batch_size, m)
            xb    = x_flat[start:end]                                   # (B, 3)
            B     = end - start

            x_rel = xb.unsqueeze(0) - self._pos.unsqueeze(1)           # (n, B, 3)
            x_loc = torch.einsum('nij,nbj->nbi', self._R, x_rel)       # (n, B, 3)
            rho2  = x_loc[..., 0]**2 + x_loc[..., 1]**2               # (n, B)

            # xi for outermost ellipsoid — used by plastic damping
            xi_outer = (rho2 / self._r2.unsqueeze(1) +
                        x_loc[..., 2]**2 / self._t2.unsqueeze(1))      # (n, B)

            # ── 1. Vectorised taper ────────────────────────────────────────
            # Expand x_loc / rho2 over taper dimension
            x_loc_t = x_loc.unsqueeze(1).expand(-1, self._N_TAPER, -1, -1).contiguous()  # (n, T, B, 3)
            rho2_t  = rho2.unsqueeze(1).expand(-1, self._N_TAPER, -1).contiguous()       # (n, T, B)

            xi_k_all = rho2_t / r2_k + x_loc_t[..., 2]**2 / t2_k     # (n, T, B)
            inside   = (xi_k_all <= 1.0).unsqueeze(-1)                 # (n, T, B, 1)

            # Interior: weight-sum collapses to single int_scale (weights sum to 1)
            grad_i = x_loc_t * int_scale_4d                            # (n, T, B, 3)

            # Exterior: full batched solve across T levels
            grad_e = self._grad_phi_ext_k(
                x_loc_t, rho2_t, r2_k, t2_k, r2t2_k, r2_t2_k, Df_k, pi_k
            )                                                           # (n, T, B, 3)

            # Weighted taper sum → (n, B, 3)
            grad_phi_t = torch.where(inside, grad_i, grad_e)           # (n, T, B, 3)
            grad_phi   = (grad_phi_t * w_t).sum(dim=1)                 # (n, B, 3)

            # Single einsum with _sig_scaled (tip 3: one matmul, not T)
            u_loc_sum = torch.einsum('nbj,nij->nbi', grad_phi, self._sig_scaled)

            # Rotate to global frame
            u_global = torch.einsum('nij,nbj->nbi', self._RT, u_loc_sum)  # (n, B, 3)

            # Plastic damping
            if self._plastic_damp:
                s_hat  = self._slip_hat[:, None, :]                    # (n, 1, 3)
                u_par  = (u_global * s_hat).sum(dim=-1, keepdim=True) * s_hat
                u_perp = u_global - u_par
                xi_lo  = self._plastic_xi_lo
                xi_hi  = self._plastic_xi_hi
                a      = ((xi_outer - xi_lo) / (xi_hi - xi_lo)).clamp(0.0, 1.0)
                a      = a * a * (3.0 - 2.0 * a)                      # smoothstep (n, B)
                u_global = u_par + a.unsqueeze(-1) * u_perp

            # Equatorial crossing check
            n_hat    = self._n_hat[:, None, :]                         # (n, 1, 3)
            d_before = (x_rel * n_hat).sum(dim=-1)                    # (n, B)
            d_after  = ((x_rel + u_global) * n_hat).sum(dim=-1)       # (n, B)
            crossing = (torch.sign(d_before) != torch.sign(d_after)) & (d_before.abs() > 0.0)
            scale    = torch.where(
                crossing,
                -d_before / (d_after - d_before + 1e-30),
                torch.ones_like(d_before),
            )                                                          # (n, B)
            u_global = u_global * scale.unsqueeze(-1)

            # Linear decay correction
            # Raw Eshelby decays as 1/r². Multiply by (d / r_ref) to get 1/r decay,
            # and then blend with the original displacement according to the specified 
            # linear decay factor. The result is no longer a valid elastic solution, 
            # but can account for non-elastic behaviour like viscous relaxation and 
            # distrubution of deformation across a damage zone.
            if self._linear_decay > 0:
                d = torch.norm(x_rel, dim=-1)               # (n, B)
                r_ref = self._radii.unsqueeze(1)             # (n, 1)
                decay_w = (d / r_ref).clamp(min=1.0)         # (n, B)  — 1.0 inside/near source
                u_global = self._linear_decay *u_global * decay_w.unsqueeze(-1) + (1 - self._linear_decay)*u_global
    
            u_out[start:end] = (u_global * self.weights[:, None, None]).sum(dim=0)

        u_out = u_out.reshape(*batch_shape, 3)
        if squeeze:
            u_out = u_out.squeeze(0)
        return _numpy(u_out) if return_numpy else u_out

    def _grad_phi_ext_k(self, x_loc, rho2, r2, t2, r2t2, r2_t2, D_floor, pi_r2t):
        """
        Exterior ∇φ — fully vectorised over sources (n), taper levels (T),
        and receivers (B).  All inputs broadcast consistently:

            x_loc   (n, T, B, 3)
            rho2    (n, T, B)
            r2 etc. (n, T, 1)     ← unsqueeze(2) in displacement()
        """
        z2   = x_loc[..., 2] ** 2                                      # (n, T, B)
        b    = -(rho2 + z2 - r2t2)
        c    = -(rho2 * t2 + z2 * r2 - r2_t2)
        disc = torch.clamp(b * b - 4.0 * c, min=0.0)
        lam  = 0.5 * (-b + torch.sqrt(disc))
        al   = lam + r2
        be   = lam + t2

        dphi_dlam = (-pi_r2t) / (al * torch.sqrt(be))
        D         = torch.clamp(rho2 / (al * al) + z2 / (be * be), min=D_floor)
        scale_eq  = dphi_dlam / (al * D) * 2.0                        # (n, T, B)
        scale_pol = dphi_dlam / (be * D) * 2.0

        grad_phi = torch.empty_like(x_loc)
        grad_phi[..., 0] = scale_eq  * x_loc[..., 0]
        grad_phi[..., 1] = scale_eq  * x_loc[..., 1]
        grad_phi[..., 2] = scale_pol * x_loc[..., 2]
        return grad_phi