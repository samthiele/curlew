import numpy as np
from curlew.geometry import poisson_disk_indices_3d

def pairwise_min_distance(points: np.ndarray) -> float:
    """
    Compute the minimum pairwise Euclidean distance in a (K, 3) array.
    O(K^2) — fine for small K in tests.
    """
    if len(points) < 2:
        return np.inf
    diff = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    # Exclude diagonal
    d2[np.arange(len(points)), np.arange(len(points))] = np.inf
    return float(np.sqrt(np.min(d2)))

def test_nonpositive_max_points_returns_empty():
    x = np.random.default_rng(0).random(10)
    y = np.random.default_rng(1).random(10)
    z = np.random.default_rng(2).random(10)
    idx = poisson_disk_indices_3d(x, y, z, radius=0.5, max_points=0)
    assert idx.shape == (0,)
    assert idx.dtype == np.int64

def test_reproducibility_with_seed():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.random(n)
    y = rng.random(n)
    z = rng.random(n)
    r = 0.1
    m = 25

    idx1 = poisson_disk_indices_3d(x, y, z, radius=r, max_points=m, seed=7)
    idx2 = poisson_disk_indices_3d(x, y, z, radius=r, max_points=m, seed=7)
    assert np.array_equal(idx1, idx2), "Results should be identical with same seed"


def test_distance_constraint_is_respected_random():
    rng = np.random.default_rng(123)
    n = 500
    x = rng.random(n) * 10.0
    y = rng.random(n) * 10.0
    z = rng.random(n) * 10.0

    radius = 0.4
    max_points = 80

    idx = poisson_disk_indices_3d(x, y, z, radius=radius, max_points=max_points, seed=99)
    pts = np.c_[x[idx], y[idx], z[idx]]

    # Check no two points are closer than radius (allow tiny numerical wiggle)
    min_d = pairwise_min_distance(pts)
    assert min_d >= radius - 1e-12, f"Min distance {min_d} is less than radius {radius}"


def test_respects_max_points_cap():
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.random(n) * 20.0
    y = rng.random(n) * 20.0
    z = rng.random(n) * 20.0

    radius = 0.2   # small enough that many points are feasible
    max_points = 100

    idx = poisson_disk_indices_3d(x, y, z, radius=radius, max_points=max_points, seed=1)
    assert len(idx) == max_points, "Should hit the requested cap when feasible"
    # Ensure uniqueness and valid range
    assert len(np.unique(idx)) == len(idx)
    assert idx.dtype == np.int64
    assert idx.min() >= 0 and idx.max() < n


def test_returns_all_when_well_spaced_grid():
    # Construct a regular grid where nearest-neighbor spacing >= 2 * radius
    radius = 0.5
    spacing = 2.5 * radius  # comfortably larger than radius
    nx = ny = nz = 4
    xs = np.arange(nx) * spacing
    ys = np.arange(ny) * spacing
    zs = np.arange(nz) * spacing
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()
    n = x.size

    idx = poisson_disk_indices_3d(x, y, z, radius=radius, max_points=1000, seed=0)
    # All should be acceptable since every pair is >= spacing > radius
    assert len(idx) == n

    pts = np.c_[x[idx], y[idx], z[idx]]
    assert pairwise_min_distance(pts) >= radius - 1e-12


def test_translation_invariance_of_selection_set():
    # Translating all coordinates should not change which indices are selected
    rng = np.random.default_rng(2025)
    n = 600
    x = rng.normal(0.0, 5.0, size=n)
    y = rng.normal(0.0, 5.0, size=n)
    z = rng.normal(0.0, 5.0, size=n)

    radius = 0.8
    max_points = 60
    seed = 77

    idx1 = poisson_disk_indices_3d(x, y, z, radius=radius, max_points=max_points, seed=seed)

    # Translate by an arbitrary vector — algorithm uses (min) shift internally, so this
    # translation shouldn't affect cell assignments relative to that shift.
    shift = np.array([123.45, -67.89, 0.314])
    idx2 = poisson_disk_indices_3d(x + shift[0], y + shift[1], z + shift[2],
                                radius=radius, max_points=max_points, seed=seed)

    # The *indices* refer to the same original elements; set equality is sufficient
    assert set(idx1.tolist()) == set(idx2.tolist())


def test_no_duplicates_and_indices_valid():
    rng = np.random.default_rng(17)
    n = 300
    x = rng.random(n)
    y = rng.random(n)
    z = rng.random(n)
    idx = poisson_disk_indices_3d(x, y, z, radius=0.15, max_points=80, seed=5)

    # No duplicates and all in range
    assert len(np.unique(idx)) == len(idx)
    assert idx.min() >= 0 and idx.max() < n