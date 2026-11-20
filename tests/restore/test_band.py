"""Tests for the data-driven wavelength band (curlew.priors.spectral)."""
import math
import time

import numpy as np
import pytest

from curlew.core import CSet
from curlew.priors.spectral import n_features_for_band, spectral_band, spectral_moments
from curlew.kinematics.euler import euler_velocity


def sine_fold(wavelength=500.0, amplitude=8.0, n=80, layers=(0.0, 60.0, 120.0), length=2000.0):
    """Gentle sinusoidal folds z = c + A sin(kx): contacts on each layer, bedding on the middle one."""
    k = 2 * np.pi / wavelength
    x = np.linspace(0, length, n)
    eq = [np.c_[x, c + amplitude * np.sin(k * x)] for c in layers]
    slope = amplitude * k * np.cos(k * x)
    gv = np.c_[-slope, np.ones_like(x)] / np.sqrt(1 + slope ** 2)[:, None]
    gp = eq[1]
    return CSet(gp=gp, gv=gv, eq=eq, crs="model")


def test_single_wavelength_is_recovered():
    # For one sinusoid, hinge spacing equals its wavelength; λ0 may be dropped when
    # contact height variance and curvature moments sit on the m2² ≤ m0 m4 bound.
    m = spectral_moments(sine_fold())
    assert m["lambda_x"] == pytest.approx(500.0, rel=0.05)
    if m["lambda_0"] is not None:
        assert m["lambda_0"] == pytest.approx(500.0, rel=0.05)


def test_band_starts_at_the_fold_scale():
    (lo, hi), m = spectral_band(sine_fold(), return_info=True)
    lam_lo = min(m["lambda_x"], m["lambda_0"] if m["lambda_0"] is not None else m["lambda_x"])
    assert lo == pytest.approx(lam_lo, rel=1e-12)
    assert hi >= 4 * lo * (1 - 1e-9)


def test_band_scales_with_the_fold():
    lo1, hi1 = spectral_band(sine_fold(wavelength=400.0, amplitude=6.0), support=0)
    lo2, hi2 = spectral_band(sine_fold(wavelength=800.0, amplitude=12.0, length=4000.0), support=0)
    assert lo2 / lo1 == pytest.approx(2.0, rel=0.05)
    assert hi1 >= 4 * lo1 * (1 - 1e-9)


def test_support_floor():
    (lo, hi), m = spectral_band(sine_fold(), return_info=True)
    assert hi >= m["diameter"]


@pytest.mark.slow
def test_cylindrical_3d_projects_axis():
    k = 2 * np.pi / 500.0
    x = np.linspace(0, 2000, 60)   # spacing λ/15: well inside the λx/7 pair limit
    y = np.linspace(0, 3000, 5)
    X, Y = np.meshgrid(x, y)
    Z = 10 * np.sin(k * X)
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    slope = 10 * k * np.cos(k * X.ravel())
    gv = np.c_[-slope, np.zeros_like(slope), np.ones_like(slope)] / np.sqrt(1 + slope ** 2)[:, None]
    m = spectral_moments(CSet(gp=pts, gv=gv, eq=[pts], crs="model"))
    assert m["profile"] and m["n_eff"] == pytest.approx(1.0, abs=0.01)
    assert m["lambda_x"] == pytest.approx(500.0, rel=0.1)


def strained(C, F, heterogeneity=0.0, n=40, seed=0):
    """``C`` pushed forward through the homogeneous strain ``F`` (positions F x, normals F^-T n),
    with ``n`` strain ellipses of B = F F^T, perturbed by a random log-strain of relative size
    ``heterogeneity``."""
    rng = np.random.default_rng(seed)
    Fi = np.linalg.inv(F)
    eq = [h @ F.T for h in C.eq]
    gp = C.gp @ F.T
    gv = C.gv @ Fi
    gv = gv / np.linalg.norm(gv, axis=1, keepdims=True)
    w, v = np.linalg.eigh(F @ F.T)
    L0 = (v * np.log(w)) @ v.T
    sp, se, sr = rng.uniform(gp.min(0), gp.max(0), (n, 2)), [], []
    for _ in range(n):
        E = rng.normal(size=(2, 2))
        E = 0.5 * (E + E.T)
        E -= np.trace(E) / 2 * np.eye(2)
        L = L0 + heterogeneity * np.linalg.norm(L0) * E / np.linalg.norm(E)
        lw, lv = np.linalg.eigh(L)
        se.append(lv)
        sr.append(np.exp(lw[1] - lw[0]))
    return CSet(gp=gp, gv=gv, eq=eq, sp=sp, se=np.stack(se), sr=np.asarray(sr), crs="model")


def test_strain_is_undone():
    # Shorten a 500-wavelength fold along the layering (pure shear, X/Z = 4): the present
    # wavelength is 250, and the strain-aware band recovers the original.
    C = sine_fold(wavelength=500.0, amplitude=6.0)
    F = np.diag([0.5, 2.0])
    S = strained(C, F, heterogeneity=0.0)
    m_present = spectral_moments(CSet(gp=S.gp, gv=S.gv, eq=S.eq, crs="model"))
    (lo, hi), info = spectral_band(S, return_info=True)
    assert m_present["lambda_x"] == pytest.approx(250.0, rel=0.1)
    assert info["lambda_x"] == pytest.approx(500.0, rel=0.1)
    assert info["strain"]["X_over_Z"] == pytest.approx(4.0, rel=1e-6)


def test_bulk_strain_lengthens_l_max():
    C = sine_fold(wavelength=500.0, amplitude=6.0)
    F = np.diag([0.5, 2.0])
    (_, hi_hom), info_hom = spectral_band(strained(C, F, heterogeneity=0.05), return_info=True)
    (_, hi_het), info_het = spectral_band(strained(C, F, heterogeneity=0.8), return_info=True)
    D = info_hom["diameter"]
    assert info_hom["strain"]["L_bulk"] == pytest.approx(
        math.pi * D / math.sqrt(2 * info_hom["strain"]["heterogeneity"]), rel=1e-9)
    assert hi_hom > hi_het      # more nearly homogeneous strain needs longer modes


def test_needs_bedding():
    with pytest.raises(ValueError):
        spectral_band(CSet(vp=np.zeros((5, 2)), vv=np.zeros((5, 1))))


def test_n_features_for_band():
    assert n_features_for_band((100, 400), 100) == 200
    assert n_features_for_band((100, 400), 100, n_directions=32) == 224
    assert n_features_for_band((100, 100), 50) == 50       # at least one octave
    with pytest.raises(ValueError):
        n_features_for_band((0, 1), 10)


@pytest.mark.slow
def test_euler_velocity_accepts_cset_and_modes_per_octave():
    from curlew.kinematics.euler import euler_velocity
    C = sine_fold()
    band = spectral_band(C)
    m = n_features_for_band(band, 64, n_directions=16)
    v = euler_velocity(2, length_scale_range=C, modes_per_octave=64, n_directions=16, stream="alpha")
    assert m % 16 == 0
    assert v.alpha.omega_spatial.shape[0] == m
    lengths = 2 * np.pi / np.linalg.norm(v.alpha.omega_spatial.numpy(), axis=1)
    assert lengths.min() >= band[0] * (1 - 1e-4) and lengths.max() <= band[1] * (1 + 1e-4)


@pytest.mark.slow
def test_samples_reproduce_the_moments():
    # m2 and m4 are declustered, noise-corrected means, so the samples are checked for
    # what is plain: the heights (about each level, one dof per level and d - 1 for the
    # normal) and the counts.
    m = spectral_moments(sine_fold(), return_samples=True)
    s = m["samples"]
    h = np.concatenate(s["heights"])
    dof = len(h) - len(s["heights"]) - (len(m["normal"]) - 1)
    assert np.sum(h ** 2) / dof == pytest.approx(m["m0"])
    assert len(s["pairs"]) == m["n_pairs"]
    assert len(s["tilt_angle"]) == m["n_bedding"]
    assert np.allclose(np.linalg.norm(s["normal"]), 1.0)


def test_global_crs_is_silent():
    import warnings
    C = sine_fold()
    C.crs = "global"
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*crs='global'.*")
        spectral_moments(C)


def test_diameter_in_moments_matches_band():
    (lo, hi), info = spectral_band(sine_fold(), return_info=True)
    assert info["diameter"] > 0
    assert lo < info["diameter"]


def test_sparse_bedding_yields_finite_wavelength():
    x = np.linspace(0, 5000, 12)
    k = 2 * np.pi / 3000.0
    z = 3.0 * np.sin(k * x)
    slope = 3.0 * k * np.cos(k * x)
    gv = np.c_[-slope, np.ones_like(x)] / np.sqrt(1 + slope ** 2)[:, None]
    gp = np.c_[x, z]
    m = spectral_moments(CSet(gp=gp, gv=gv, crs="model"))
    assert m["lambda_x"] > 0
    assert m["n_bedding"] == 12


def test_sr_only_strain_with_bedding_frames():
    C = sine_fold()
    F = np.diag([0.5, 2.0])
    S = strained(C, F, heterogeneity=0.0)
    S2 = CSet(gp=S.gp, gv=S.gv, eq=S.eq, sp=S.sp, sr=S.sr, crs="model")
    lo, hi = spectral_band(S2)
    assert 0 < lo <= hi
    (_, _), info = spectral_band(S2, return_info=True)
    assert info["strain"] is not None
    assert info["strain"]["n_markers"] == len(S2.sr)


def test_cylindrical_hinge_axis_matches_auto():
    """Profile moments with explicit fold axis agree with auto-detected cylindrical gauge."""
    k = 2 * np.pi / 500.0
    x = np.linspace(0, 2000, 24)
    y = np.linspace(0, 3000, 4)
    X, Y = np.meshgrid(x, y)
    Z = 10 * np.sin(k * X)
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    slope = 10 * k * np.cos(k * X.ravel())
    gv = np.c_[-slope, np.zeros_like(slope), np.ones_like(slope)] / np.sqrt(
        1 + slope ** 2)[:, None]
    C = CSet(gp=pts, gv=gv, eq=[pts], crs="model")
    m_hinge = spectral_moments(C, axis=np.array([0.0, 1.0, 0.0]))
    m_auto = spectral_moments(C)
    assert m_hinge["lambda_x"] == pytest.approx(500.0, rel=0.15)
    assert m_hinge["lambda_x"] == pytest.approx(m_auto["lambda_x"], rel=0.05)


@pytest.mark.slow
def test_cylindrical_euler_uses_section_axis_band():
    k = 2 * np.pi / 500.0
    x = np.linspace(0, 2000, 30)
    y = np.linspace(0, 3000, 5)
    X, Y = np.meshgrid(x, y)
    Z = 10 * np.sin(k * X)
    pts = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    slope = 10 * k * np.cos(k * X.ravel())
    gv = np.c_[-slope, np.zeros_like(slope), np.ones_like(slope)] / np.sqrt(
        1 + slope ** 2)[:, None]
    C = CSet(gp=pts, gv=gv, eq=[pts], crs="model")
    euler_velocity(
        3, length_scale_range=C, cylindrical=True, section_axis=1, n_features=4,
    )


def test_orientations_only_ignores_contacts():
    # Bedding alone gives the hinge spacing; the contacts (and so heights) are left out.
    C = sine_fold()
    m = spectral_moments(C, orientations_only=True)
    assert m["lambda_x"] == pytest.approx(500.0, rel=0.1)
    assert m["lambda_0"] is None and m["m0"] is None and m["n_contacts"] == 0
    # Moving the contacts changes nothing when they are ignored.
    shifted = CSet(gp=C.gp, gv=C.gv, eq=[t + 1e3 for t in C.eq], crs="model")
    assert spectral_band(shifted, orientations_only=True) == pytest.approx(spectral_band(C, orientations_only=True))
    band, info = spectral_band(C, orientations_only=True, return_info=True)
    assert info["orientations_only"] and band[0] == pytest.approx(info["lambda_x"])


def test_longer_mode_picks_the_gentle_curvatures():
    from curlew.priors.spectral import _longer_mode
    rng = np.random.default_rng(0)
    gentle = np.exp(rng.normal(np.log(1e-3), 0.2, 40))
    sharp = np.exp(rng.normal(np.log(2e-2), 0.2, 12))     # ~4 octaves tighter
    kappa = np.r_[gentle, sharp]
    keep, modes = _longer_mode(kappa, np.ones(len(kappa)))
    assert modes is not None and modes["gap_octaves"] > 3
    assert keep[:40].all() and not keep[40:].any()
    # one mode, or two modes less than an octave apart: no split
    keep, modes = _longer_mode(gentle, np.ones(len(gentle)))
    assert modes is None and keep.all()
    near = np.r_[gentle, gentle * 1.5]
    keep, modes = _longer_mode(near, np.ones(len(near)))
    assert modes is None and keep.all()


def test_prefer_longer_leaves_a_single_mode_alone():
    C = sine_fold()
    a, b = spectral_moments(C), spectral_moments(C, prefer_longer=True)
    assert b["lambda_x"] == pytest.approx(a["lambda_x"], rel=0.15)
    assert spectral_band(C, prefer_longer=True)[0] == pytest.approx(spectral_band(C)[0], rel=0.15)


@pytest.mark.slow
def test_spectral_moments_runtime():
    C = sine_fold(n=400, length=4000.0)
    t0 = time.perf_counter()
    spectral_moments(C)
    assert time.perf_counter() - t0 < 15.0
