import os
import time

import numpy as np
import pytest
import torch

import curlew
from curlew.fields.eshelby import EshelbyField

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length vector")
    return v / n

def _build_fault_field(
    *,
    n_sources: int,
    fault_length: float,
    r: float,
    t: float,
    mu: float,
    nu: float,
    fault_normal=(1.0, 1.0, 0.0),
    origin=(0.0, 0.0, 0.0),
    slip_magnitude: float = 0.01,
    weight_scale: float = 1.0,
    taper: str = "gaussian",
) -> EshelbyField:
    """ Set up an Eshelby field to benchmark with """
    n_hat = _unit(np.array(fault_normal, dtype=float))

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n_hat, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])
    strike_hat = _unit(np.cross(ref, n_hat))

    origin_np = np.array(origin, dtype=float)

    s_centers = np.linspace(-fault_length / 2, fault_length / 2, n_sources)
    positions = origin_np[None, :] + s_centers[:, None] * strike_hat[None, :]

    if taper == "gaussian":
        sigma = fault_length / 4.0
        taper_w = np.exp(-s_centers**2 / (2 * sigma**2))
        taper_w /= taper_w.max()
    else:
        taper_w = np.ones(n_sources)

    source_w = slip_magnitude * weight_scale * taper_w
    return EshelbyField(
        "faults",
        positions=positions,
        normals=n_hat,
        slips=strike_hat,
        radii=r,
        thicknesses=t,
        stretch=1.0,
        mu=mu,
        nu=nu,
        weights=source_w,
        max_ram_mb=2048,
    )


def _grid_receivers(res: int, lim: float) -> np.ndarray:
    xs = np.linspace(-lim, lim, res)
    ys = np.linspace(-lim, lim, res)
    XX, YY = np.meshgrid(xs, ys)
    return np.stack([XX, YY, np.zeros_like(XX)], axis=-1)  # (res,res,3)


def _time_field(field: EshelbyField, receivers, *, device: str, use_predict: bool, repeats: int = 3) -> float:
    # Use torch tensors to avoid repeated numpy->torch copies.
    x = torch.as_tensor(receivers, dtype=curlew.dtype, device=curlew.device)

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, "mps"):
            # available in recent torch builds on macOS
            torch.mps.synchronize()

    def _eval():
        if use_predict:
            return field.predict(x, max_ram_mb=2048)
        return field.displacement(x, max_ram_mb=2048)

    _ = _eval()
    _sync()

    times = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        _ = _eval()
        _sync()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def _bench_field_and_grid():
    """Shared EshelbyField + receiver grid for timing comparisons (array-based constructor)."""
    n_sources = int(os.environ.get("CURLEW_BENCH_NSOURCES", "100"))
    res = int(os.environ.get("CURLEW_BENCH_RES", "500"))
    lim = float(os.environ.get("CURLEW_BENCH_LIM", "50.0"))

    field = _build_fault_field(
        n_sources=n_sources,
        fault_length=5.0,
        r=1.0,
        t=1e-2,
        mu=1.0,
        nu=0.25,
        taper="gaussian",
        fault_normal=(1.0, 1.0, 0.0),
        weight_scale=3.0,
    )
    receivers = _grid_receivers(res=res, lim=lim)
    return field, receivers, n_sources, res, lim


def test_eshelbyfield_culling_benchmark_report():
    """
    Compare ``displacement`` (all receivers) vs ``predict`` (mask then evaluate inside spheres).

    Default grid ``lim=50`` with ``r=1`` and ``far_radius_mult=20`` (influence radius 20)
    leaves most of the [−lim, lim]² plane outside all sources, so ``predict`` should win.

    Run explicitly with:
        CURLEW_BENCH=1 mamba run -n curlew pytest -q -s tests/test_eshelby_benchmark.py

    Env overrides: ``CURLEW_BENCH_NSOURCES``, ``CURLEW_BENCH_RES``, ``CURLEW_BENCH_LIM``.
    """
    if os.environ.get("CURLEW_BENCH", "0") != "1":
        pytest.skip("Set CURLEW_BENCH=1 to run performance benchmark.")

    results = []

    # CPU benchmark
    curlew.device = "cpu"
    curlew.dtype = torch.float32
    field, receivers, n_sources, res, lim = _bench_field_and_grid()
    t_no = _time_field(field, receivers, device="cpu", use_predict=False)
    t_yes = _time_field(field, receivers, device="cpu", use_predict=True)
    results.append(("cpu", t_no, t_yes))

    # GPU benchmark(s)
    if torch.cuda.is_available():
        curlew.device = "cuda"
        curlew.dtype = torch.float32
        field, receivers, n_sources, res, lim = _bench_field_and_grid()
        t_no = _time_field(field, receivers, device="cuda", use_predict=False)
        t_yes = _time_field(field, receivers, device="cuda", use_predict=True)
        results.append(("cuda", t_no, t_yes))

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        curlew.device = "mps"
        curlew.dtype = torch.float32
        field, receivers, n_sources, res, lim = _bench_field_and_grid()
        t_no = _time_field(field, receivers, device="mps", use_predict=False)
        t_yes = _time_field(field, receivers, device="mps", use_predict=True)
        results.append(("mps", t_no, t_yes))

    # Sanity: on influenced cells, predict matches full displacement (outside sphere predict is zero).
    x_chk = torch.as_tensor(receivers, dtype=curlew.dtype, device=curlew.device)
    u_full = field.displacement(x_chk, max_ram_mb=2048)
    u_pred = field.predict(x_chk, max_ram_mb=2048)
    mask = field.influence_mask(x_chk)
    if bool(mask.any()):
        torch.testing.assert_close(
            u_pred[mask],
            u_full[mask],
            rtol=1e-4,
            atol=1e-5,
        )
    frac = float(mask.float().mean().item())

    out_lines = []
    out_lines.append(
        f"EshelbyField benchmark: n_sources={n_sources}, res={res} ({res * res} receivers), lim={lim}, "
        f"influence_frac={frac:.3f}"
    )
    for dev, t_no, t_yes in results:
        speedup = (t_no / t_yes) if t_yes > 0 else float("inf")
        out_lines.append(
            f"  {dev:>4s}: displacement={t_no:.4f}s  predict={t_yes:.4f}s  speedup={speedup:.2f}x"
        )
        assert t_yes < t_no, (
            f"{dev}: predict ({t_yes:.4f}s) should be faster than displacement ({t_no:.4f}s) when "
            f"most receivers are culled (influence_frac={frac:.3f}). Reduce CURLEW_BENCH_LIM or increase "
            f"default far_radius_mult on the field if this fails."
        )

    print("\n".join(out_lines))

