"""
Tests for PointKernelField (PKF): signed distance fields, RBF-style scalar interpolation,
and RBF-style vector field interpolation.
"""

import numpy as np
import torch


def _sample_circle(n_pts: int, r: float = 1.0):
    """Sample points and outward normals on circle x² + y² = r²."""
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    positions = np.stack([x, y], axis=1)
    normals = positions / r
    return positions.astype(np.float32), normals.astype(np.float32)


def _sample_sphere(n_pts: int, r: float = 1.0):
    """Sample points and outward normals on sphere x² + y² + z² = r²."""
    n_theta = max(3, int(np.sqrt(n_pts * 2)))
    n_phi = max(2, (n_pts + n_theta - 1) // n_theta)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi + 2)[1:-1]
    theta_g, phi_g = np.meshgrid(theta, phi, indexing="ij")
    theta_g = theta_g.ravel()
    phi_g = phi_g.ravel()
    x = r * np.sin(phi_g) * np.cos(theta_g)
    y = r * np.sin(phi_g) * np.sin(theta_g)
    z = r * np.cos(phi_g)
    positions = np.stack([x, y, z], axis=1)
    normals = positions / r
    return positions.astype(np.float32), normals.astype(np.float32)


def _true_sdf_circle(x: np.ndarray) -> np.ndarray:
    """True signed distance to unit circle: r - 1."""
    r = np.sqrt(np.sum(x ** 2, axis=-1))
    return (r - 1.0).astype(np.float32)


def _true_sdf_sphere(x: np.ndarray) -> np.ndarray:
    """True signed distance to unit sphere: r - 1."""
    r = np.sqrt(np.sum(x ** 2, axis=-1))
    return (r - 1.0).astype(np.float32)


def _implicit_sin_2d(x: np.ndarray) -> np.ndarray:
    """Sinusoidal implicit: sin(x)*cos(y)."""
    return (np.sin(x[..., 0]) * np.cos(x[..., 1])).astype(np.float32)


def _implicit_sin_3d(x: np.ndarray) -> np.ndarray:
    """Sinusoidal implicit: sin(x)*cos(y)*sin(z)."""
    return (
        np.sin(x[..., 0]) * np.cos(x[..., 1]) * np.sin(x[..., 2])
    ).astype(np.float32)

def _vector_field_2d(x: np.ndarray) -> np.ndarray:
    """2D vector field (vx, vy) = (cos(x), sin(y)). Shape (..., 2)."""
    out = np.empty((*x.shape[:-1], 2), dtype=np.float32)
    out[..., 0] = np.cos(x[..., 0])
    out[..., 1] = np.sin(x[..., 1])
    return out

def _vector_field_3d(x: np.ndarray) -> np.ndarray:
    """3D vector field (vx, vy, vz) = (sin(x), cos(y), sin(z)). Shape (..., 3)."""
    out = np.empty((*x.shape[:-1], 3), dtype=np.float32)
    out[..., 0] = np.sin(x[..., 0])
    out[..., 1] = np.cos(x[..., 1])
    out[..., 2] = np.sin(x[..., 2])
    return out

def test_point_kernel_distance_fields():
    """PointKernelField as SDF from points on circle/sphere: key points and vs true SDF (loops over dim)."""
    from curlew.fields.point import PointKernelField

    cases = [
        {
            "dim": 2,
            "sample": lambda n: _sample_circle(n),
            "true_sdf": _true_sdf_circle,
            "n_pts_basic": 128,
            "n_pts_grid": 256,
            "origin": np.zeros((1, 2), dtype=np.float32),
            "exterior": np.array([[2.0, 0.0]], dtype=np.float32),
            "on_surf": np.array([[1.0, 0.0]], dtype=np.float32),
            "grid_mae_max": 0.12,
            "sdf_inside_tol": 0.15,
            "sdf_outside_tol": 0.15,
            "on_surf_tol": 0.1,
            "far_q": np.array([[2.0, 0.0]], dtype=np.float32),
            "far_lo": 0.5,
            "far_hi": 1.5,
        },
        {
            "dim": 3,
            "sample": lambda n: _sample_sphere(n),
            "true_sdf": _true_sdf_sphere,
            "n_pts_basic": 256,
            "n_pts_grid": 512,
            "origin": np.zeros((1, 3), dtype=np.float32),
            "exterior": np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            "on_surf": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "grid_mae_max": 0.15,
            "sdf_inside_tol": 0.2,
            "sdf_outside_tol": 0.2,
            "on_surf_tol": 0.15,
            "far_q": np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            "far_lo": 0.5,
            "far_hi": 1.5,
        },
    ]

    for c in cases:
        dim = c["dim"]
        positions, normals = c["sample"](c["n_pts_basic"])
        field = PointKernelField(
            name="sdf",
            input_dim=dim,
            positions=positions,
            normals=normals,
            kernel="closest",
        )

        # Origin (inside): SDF negative ≈ -1
        out_origin = np.atleast_1d(
            field.forward(torch.tensor(c["origin"])).reshape(-1).numpy()
        )
        assert out_origin[0] < -0.5, f"dim={dim}: inside expected negative SDF"
        assert abs(out_origin[0] - (-1.0)) < c["sdf_inside_tol"], f"dim={dim}: inside ≈ -1"

        # Exterior: SDF positive ≈ 1
        out_ext = np.atleast_1d(
            field.forward(torch.tensor(c["exterior"])).reshape(-1).numpy()
        )
        assert out_ext[0] > 0.5, f"dim={dim}: outside expected positive SDF"
        assert abs(out_ext[0] - 1.0) < c["sdf_outside_tol"], f"dim={dim}: outside ≈ 1"

        # On surface: SDF ≈ 0
        out_surf = np.atleast_1d(
            field.forward(torch.tensor(c["on_surf"])).reshape(-1).numpy()
        )
        assert abs(out_surf[0]) < c["on_surf_tol"], f"dim={dim}: on surface ≈ 0"

        # Vs true SDF on a query set
        positions_grid, _ = c["sample"](c["n_pts_grid"])
        if dim == 2:
            xx = np.linspace(0.3, 2.5, 8)
            yy = np.linspace(0.3, 2.5, 8)
            xg, yg = np.meshgrid(xx, yy, indexing="ij")
            queries = np.stack([xg.ravel(), yg.ravel()], axis=1).astype(np.float32)
        else:
            np.random.seed(42)
            n_q = 200
            r = np.random.uniform(0.4, 2.0, size=n_q)
            theta = np.random.uniform(0, 2 * np.pi, size=n_q)
            phi = np.random.uniform(0, np.pi, size=n_q)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            queries = np.stack([x, y, z], axis=1).astype(np.float32)
        true_sdf = c["true_sdf"](queries)
        pred = field.forward(torch.tensor(queries)).reshape(-1).numpy()
        mae = np.mean(np.abs(pred - true_sdf))
        assert mae < c["grid_mae_max"], f"dim={dim}: SDF MAE {mae:.4f} < {c['grid_mae_max']}"

        # Closest kernel: far query in reasonable range
        out_far = np.atleast_1d(
            field.forward(torch.tensor(c["far_q"])).reshape(-1).numpy()
        )
        assert out_far[0] > c["far_lo"] and out_far[0] < c["far_hi"], f"dim={dim}: far query in range"

def test_rbf_scalar_interpolation():
    """RBF-style scalar interpolation with linear and gaussian kernels (loops over dim and kernel)."""
    from curlew.fields.point import PointKernelField

    cases = [
        # (dim, kernel, value_fn, positions_builder, query_builder, kernel_kwargs, atol_seed, rtol_seed, mae_max)
        (2, "linear", _true_sdf_circle, (0.4, 2.2, 10), ("grid", 0.5, 2.0, 6), {"eps": 1e-6}, 1e-3, 1e-4, 0.2),
        (2, "gaussian", _implicit_sin_2d, (-1.5, 1.5, 8), ("random", 42, 50, -1.0, 1.0), {"sigma": 0.25}, 0.08, 0.08, 0.35),
        (3, "linear", _true_sdf_sphere, (-1.5, 1.5, 6), ("shell", 43, 80, 0.6, 1.8), {"eps": 1e-6}, 1e-3, 1e-4, 0.45),
        (3, "gaussian", _implicit_sin_3d, (-1.0, 1.0, 5), ("random", 44, 40, -0.8, 0.8), {"sigma": 0.3}, 0.12, 0.12, 0.45),
    ]

    for dim, kernel, value_fn, pos_args, q_spec, kernel_kwargs, atol_seed, rtol_seed, mae_max in cases:
        lo, hi, n = pos_args
        if dim == 2:
            xx = np.linspace(lo, hi, n)
            yy = np.linspace(lo, hi, n)
            xg, yg = np.meshgrid(xx, yy, indexing="ij")
            positions = np.stack([xg.ravel(), yg.ravel()], axis=1).astype(np.float32)
        else:
            xx = np.linspace(lo, hi, n)
            yy = np.linspace(lo, hi, n)
            zz = np.linspace(lo, hi, n)
            xg, yg, zg = np.meshgrid(xx, yy, zz, indexing="ij")
            positions = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1).astype(np.float32)
        values = value_fn(positions)

        pkf = PointKernelField(
            name="rbf",
            input_dim=dim,
            positions=positions,
            values=values,
            kernel=kernel,
            kernel_kwargs=kernel_kwargs,
        )

        pred_at_seeds = pkf.forward(torch.tensor(positions)).reshape(-1).numpy()
        np.testing.assert_allclose(
            pred_at_seeds, values, rtol=rtol_seed, atol=atol_seed,
            err_msg=f"dim={dim} kernel={kernel}: at seeds",
        )

        if q_spec[0] == "grid":
            _, qlo, qhi, qn = q_spec
            qx = np.linspace(qlo, qhi, qn)
            qy = np.linspace(qlo, qhi, qn)
            qgx, qgy = np.meshgrid(qx, qy, indexing="ij")
            queries = np.stack([qgx.ravel(), qgy.ravel()], axis=1).astype(np.float32)
        elif q_spec[0] == "random":
            _, seed, n_q, qlo, qhi = q_spec
            rng = np.random.default_rng(seed)
            queries = rng.uniform(qlo, qhi, size=(n_q, dim)).astype(np.float32)
        else:  # shell
            _, seed, n_q, r_lo, r_hi = q_spec
            np.random.seed(seed)
            r = np.random.uniform(r_lo, r_hi, size=n_q)
            theta = np.random.uniform(0, 2 * np.pi, size=n_q)
            phi = np.random.uniform(0, np.pi, size=n_q)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            queries = np.stack([x, y, z], axis=1).astype(np.float32)
        true_vals = value_fn(queries)
        pred = pkf.forward(torch.tensor(queries)).reshape(-1).numpy()
        mae = np.mean(np.abs(pred - true_vals))
        assert mae < mae_max, f"dim={dim} kernel={kernel}: MAE {mae:.4f} < {mae_max}"

def test_rbf_vector_interpolation():
    """RBF-style vector field interpolation (loops over dim and kernel)."""
    from curlew.fields.point import PointKernelField

    vector_fns = {2: _vector_field_2d, 3: _vector_field_3d}
    cases = [
        (2, "linear", (-1.2, 1.2, 8), (45, 30, -0.8, 0.8), {"eps": 1e-6}, 1e-3, 1e-4, 0.25),
        (2, "gaussian", (-1.5, 1.5, 7), (46, 35, -1.0, 1.0), {"sigma": 0.35}, 0.15, 0.2, 0.4),
        (3, "linear", (-1.0, 1.0, 5), (47, 25, -0.7, 0.7), {"eps": 1e-6}, 1e-3, 1e-4, 0.35),
        (3, "gaussian", (-1.0, 1.0, 5), (48, 30, -0.7, 0.7), {"sigma": 0.35}, 0.12, 0.12, 0.5),
    ]

    for dim, kernel, pos_args, q_spec, kernel_kwargs, atol_seed, rtol_seed, mae_max in cases:
        lo, hi, n = pos_args
        seed, n_q, qlo, qhi = q_spec
        vec_fn = vector_fns[dim]
        if dim == 2:
            xx = np.linspace(lo, hi, n)
            yy = np.linspace(lo, hi, n)
            xg, yg = np.meshgrid(xx, yy, indexing="ij")
            positions = np.stack([xg.ravel(), yg.ravel()], axis=1).astype(np.float32)
        else:
            xx = np.linspace(lo, hi, n)
            yy = np.linspace(lo, hi, n)
            zz = np.linspace(lo, hi, n)
            xg, yg, zg = np.meshgrid(xx, yy, zz, indexing="ij")
            positions = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1).astype(np.float32)
        values = vec_fn(positions)

        pkf = PointKernelField(
            name="rbf_vec",
            input_dim=dim,
            positions=positions,
            values=values,
            kernel=kernel,
            kernel_kwargs=kernel_kwargs,
        )

        pred_at_seeds = pkf.forward(torch.tensor(positions)).numpy()
        np.testing.assert_allclose(
            pred_at_seeds, values, rtol=rtol_seed, atol=atol_seed,
            err_msg=f"dim={dim} kernel={kernel}: at seeds",
        )

        rng = np.random.default_rng(seed)
        queries = rng.uniform(qlo, qhi, size=(n_q, dim)).astype(np.float32)
        true_vals = vec_fn(queries)
        pred = pkf.forward(torch.tensor(queries)).numpy()
        mae = np.mean(np.abs(pred - true_vals))
        assert mae < mae_max, f"dim={dim} kernel={kernel}: MAE {mae:.4f} < {mae_max}"


# -----------------------------------------------------------------------------
# Eshelby / moment-tensor displacement tests: compare to Mogi (isotropic) and
# double-couple analytical solutions.
# -----------------------------------------------------------------------------

def _displacement_isotropic_mogi_reference(
    r: torch.Tensor,
    M0: float,
    mu: float,
    nu: float,
    r_min: float = 1e-8,
) -> torch.Tensor:
    """
    Displacement from a pure isotropic (Mogi-type) point source in 3D full space.
    M = M0 * I  =>  u = (M0 / (8*pi*mu*(1-nu)*r^2)) * n  (radial).
    r shape (..., 3); returns (..., 3).
    """
    import math
    r_norm = torch.linalg.norm(r, dim=-1, keepdim=True).clamp(min=r_min)
    n = r / r_norm
    r_sq = (r_norm.squeeze(-1) ** 2).clamp(min=r_min ** 2).unsqueeze(-1)
    coeff = M0 / (8.0 * math.pi * mu * (1.0 - nu) * r_sq)
    return coeff * n


def _displacement_double_couple_reference(
    r: torch.Tensor,
    n_fault: torch.Tensor,
    d_slip: torch.Tensor,
    M0: float,
    mu: float,
    nu: float,
    r_min: float = 1e-8,
) -> torch.Tensor:
    """
    Displacement from a double-couple point source in 3D full space.
    M = M0 * (n⊗d + d⊗n). Uses same formula as _displacementPointMomentTensor.
    n_fault, d_slip: unit vectors (3,); r shape (..., 3); returns (..., 3).
    """
    import math
    M = M0 * (
        torch.outer(n_fault, d_slip) + torch.outer(d_slip, n_fault)
    )
    r_norm = torch.linalg.norm(r, dim=-1, keepdim=True).clamp(min=r_min)
    n = r / r_norm
    n_M_n = (n * (M @ n.unsqueeze(-1)).squeeze(-1)).sum(dim=-1, keepdim=True)
    trM = (M.trace() * torch.ones((*n.shape[:-1], 1), device=n.device, dtype=n.dtype))
    M_n = (M @ n.unsqueeze(-1)).squeeze(-1)
    r_sq = (r_norm.squeeze(-1) ** 2).clamp(min=r_min ** 2).unsqueeze(-1)
    coeff = 1.0 / (16.0 * math.pi * mu * (1.0 - nu) * r_sq)
    u = coeff * (2.0 * M_n - trM * n + 3.0 * n_M_n * n)
    return u


def test_eshelby_displacement_isotropic_mogi():
    """Displacement from isotropic (Mogi-type) moment tensor matches analytical formula."""
    from curlew.fields.point import _displacementPointMomentTensor

    mu = 1.0
    nu = 0.25
    M0 = 1.0
    r_min = 0.1

    # 3D: observer at (3, 0, 0)
    r = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)
    M_iso = M0 * torch.eye(3, dtype=torch.float64).unsqueeze(0)

    u_ref = _displacement_isotropic_mogi_reference(r, M0, mu, nu, r_min)
    u_pred = _displacementPointMomentTensor(r, M_iso, mu=mu, nu=nu, r_min=r_min)

    np.testing.assert_allclose(
        u_pred.numpy(), u_ref.numpy(), rtol=1e-5, atol=1e-8,
        err_msg="Isotropic (Mogi) displacement vs reference",
    )

    # 2D (plane strain): same but r is (1, 2); reference is 3D then take in-plane
    r_2d = torch.tensor([[3.0, 0.0]], dtype=torch.float64)
    r_3d = torch.cat([r_2d, r_2d.new_zeros(1, 1)], dim=-1)
    u_ref_3d = _displacement_isotropic_mogi_reference(r_3d, M0, mu, nu, r_min)
    u_ref_2d = u_ref_3d[..., :2]
    u_pred_2d = _displacementPointMomentTensor(r_2d, M_iso, mu=mu, nu=nu, r_min=r_min)
    np.testing.assert_allclose(
        u_pred_2d.numpy(), u_ref_2d.numpy(), rtol=1e-5, atol=1e-8,
        err_msg="Isotropic (Mogi) displacement 2D plane strain vs reference",
    )


def test_eshelby_displacement_double_couple():
    """PKF with eshelbyDisplacement uses circular near-field correction; compare to circular_eshelby_displacement and point-source in far field."""
    import math
    from curlew.fields.point import (
        PointKernelField,
        eshelbyDisplacement,
        circular_eshelby_displacement,
        _moment_tensor,
        _displacementPointMomentTensor,
    )

    mu = 1.0
    nu = 0.25
    slip_magnitude = 1.0
    dilation_magnitude = 0.0
    r_min = 0.1
    radius = 0.05
    thickness = 0.1
    patch_area = math.pi * (radius ** 2)

    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    slip_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    observers = np.array([
        [2.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [-1.0, 0.0, 1.0],
    ], dtype=np.float32)

    normals_t = torch.tensor(normals)
    slip_t = torch.tensor(slip_direction).unsqueeze(0).expand(1, 3)
    M = _moment_tensor(
        normals_t, slip_t, dilation_magnitude, slip_magnitude,
        mu=mu, poisson_ratio=nu,
    )
    r_vec = torch.tensor(observers)
    n_ref = normals_t.expand(r_vec.shape[0], 3)
    d_ref = slip_t.expand(r_vec.shape[0], 3)
    M_scaled = M * patch_area
    u_circular_ref = circular_eshelby_displacement(
        r_vec, M_scaled, radius, radius, mu, nu,
        normals=n_ref, slip_direction=d_ref, slip_magnitude=slip_magnitude,
        thickness=thickness,
    )

    value_kwargs = {
        "radius": radius,
        "thickness": thickness,
        "slip_direction": torch.tensor(slip_direction),
        "slip_magnitude": slip_magnitude,
        "dilation_magnitude": dilation_magnitude,
        "mu": mu,
        "nu": nu,
    }
    pkf = PointKernelField(
        name="eshelby_dc",
        input_dim=3,
        output_dim=3,
        positions=positions,
        normals=normals,
        values=eshelbyDisplacement,
        value_kwargs=value_kwargs,
        kernel="closest",
    )
    u_pkf = pkf.forward(torch.tensor(observers))
    if u_pkf.dim() == 3:
        u_pkf = u_pkf.squeeze(1)

    np.testing.assert_allclose(
        u_pkf.numpy(), u_circular_ref.numpy(), rtol=1e-4, atol=1e-6,
        err_msg="eshelbyDisplacement (PKF) vs direct circular_eshelby_displacement",
    )

    # Point-source formula still matches _displacementPointMomentTensor
    u_point = _displacementPointMomentTensor(r_vec, M, mu=mu, nu=nu, r_min=r_min)
    n_fault = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    d_slip = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    M0_dc = mu * slip_magnitude
    u_ref_formula = _displacement_double_couple_reference(
        r_vec.double(), n_fault, d_slip, M0_dc, mu, nu, r_min
    )
    np.testing.assert_allclose(
        u_point.numpy(), u_ref_formula.numpy(), rtol=1e-4, atol=1e-6,
        err_msg="Point-source displacement vs double-couple analytical formula",
    )

    # Far field: circular (sharp) transitions to u_far, so should match point-source when r >> radius
    observers_far = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32)
    r_far = torch.tensor(observers_far)
    n_far = normals_t.expand(r_far.shape[0], 3)
    d_far = slip_t.expand(r_far.shape[0], 3)
    u_circular_far = circular_eshelby_displacement(
        r_far, M_scaled, radius, radius, mu, nu,
        normals=n_far, slip_direction=d_far, slip_magnitude=slip_magnitude,
        thickness=thickness,
    )
    u_point_far = _displacementPointMomentTensor(r_far, M_scaled, mu=mu, nu=nu, r_min=r_min)
    np.testing.assert_allclose(
        u_circular_far.numpy(), u_point_far.numpy(), rtol=0.05, atol=1e-6,
        err_msg="Circular vs point-source in far field (r >> radius)",
    )


def test_eshelby_displacement_single_seed_self_consistent():
    """PKF with eshelbyDisplacement matches direct circular_eshelby_displacement for one seed (2D and 3D)."""
    import math
    from curlew.fields.point import (
        PointKernelField,
        eshelbyDisplacement,
        circular_eshelby_displacement,
        _moment_tensor,
    )

    mu = 2.0
    nu = 0.2
    slip_magnitude = 0.5
    dilation_magnitude = 0.1
    radius = 0.08
    thickness = 0.12
    patch_area = math.pi * (radius ** 2)

    for dim in (2, 3):
        if dim == 2:
            positions = np.array([[0.0, 0.0]], dtype=np.float32)
            normals = np.array([[1.0, 0.0]], dtype=np.float32)
            slip_direction = np.array([0.0, 1.0], dtype=np.float32)
            observers = np.array([[2.0, 0.0], [1.0, 1.0], [-0.5, 1.5]], dtype=np.float32)
        else:
            positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            normals = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            slip_direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            observers = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)

        normals_t = torch.tensor(normals)
        slip_t = torch.tensor(slip_direction).unsqueeze(0).expand(1, dim)
        M = _moment_tensor(
            normals_t, slip_t, dilation_magnitude, slip_magnitude,
            mu=mu, poisson_ratio=nu,
        )
        r_vec = torch.tensor(observers)
        n_ref = normals_t.expand(r_vec.shape[0], dim)
        d_ref = slip_t.expand(r_vec.shape[0], dim)
        M_scaled = M * patch_area
        u_circular_ref = circular_eshelby_displacement(
            r_vec, M_scaled, radius, radius, mu, nu,
            normals=n_ref, slip_direction=d_ref, slip_magnitude=slip_magnitude,
            thickness=thickness,
        )

        value_kwargs = {
            "radius": radius,
            "thickness": thickness,
            "slip_direction": torch.tensor(slip_direction),
            "slip_magnitude": slip_magnitude,
            "dilation_magnitude": dilation_magnitude,
            "mu": mu,
            "nu": nu,
        }
        pkf = PointKernelField(
            name="eshelby",
            input_dim=dim,
            output_dim=dim,
            positions=positions,
            normals=normals,
            values=eshelbyDisplacement,
            value_kwargs=value_kwargs,
            kernel="closest",
        )
        u_pkf = pkf.forward(torch.tensor(observers))
        if u_pkf.dim() == 3:
            u_pkf = u_pkf.squeeze(1)

        np.testing.assert_allclose(
            u_pkf.numpy(), u_circular_ref.numpy(), rtol=1e-4, atol=1e-6,
            err_msg=f"eshelbyDisplacement PKF vs circular_eshelby_displacement (dim={dim})",
        )
