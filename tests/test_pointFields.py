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
