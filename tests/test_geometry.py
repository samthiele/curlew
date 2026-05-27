import numpy as np
from curlew.geometry import Grid, grid, poisson_disk_indices_3d


def _pairwise_min_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return np.inf
    diff = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    d2[np.arange(len(points)), np.arange(len(points))] = np.inf
    return float(np.sqrt(np.min(d2)))


def test_gridConstruction():
    extent = ((0, 0), (200, 100))

    G = grid(extent, size=(10, 10))
    assert G.shape == (10, 10)
    assert G.dims == (200, 100)
    assert G.step == (20.0, 10.0)
    assert np.allclose(G.center, (100, 50))

    G2 = grid(extent, size=(20.0, 10.0))
    assert G2.shape == (10, 10)
    assert G2.step == (20.0, 10.0)

    pts = np.array([[0, 0], [200, 100], [50, 25]])
    G3 = grid(pts, size=4)
    assert G3.shape == (4, 4)
    assert G3.dims == (200, 100)

    dims = (200, 100, 50)
    G4 = Grid(dims, step=(1, 1, 1), center=(dims[0] / 2, dims[1] / 2, dims[2] / 2))
    coords = G4.coords(transform=False)
    assert coords.shape == (np.prod(dims), 3)
    block = G4.reshape(coords[:, 0])
    assert block.shape == G4.shape
    assert (block[:, 0, 0] == G4.axes[0]).all()
    block = G4.reshape(coords)
    assert (block[:, 0, 0, 0] == G4.axes[0]).all()
    assert (block[0, :, 0, 1] == G4.axes[1]).all()
    assert (block[0, 0, :, 2] == G4.axes[2]).all()

    dims2d = (200, 100)
    G2d = Grid(dims2d, step=(1, 1), center=(dims2d[0] / 2, dims2d[1] / 2))
    cxy = G2d.coords(transform=False)
    assert cxy.shape == (np.prod(dims2d), 2)
    block = G2d.reshape(cxy[:, 0])
    assert (block[:, 0] == G2d.axes[0]).all()
    block = G2d.reshape(cxy)
    assert (block[:, 0, 0] == G2d.axes[0]).all()
    assert (block[0, :, 1] == G2d.axes[1]).all()

    for G in [
        Grid([4600, 4000, 2500], step=30, center=[200, 100, 50]),
        Grid([4600, 4000], step=30, center=[200, 100]),
    ]:
        for transform in (False, True):
            points = G.coords(transform=transform)
            offset = np.zeros(G.ndim) if not transform else G.center
            for i in range(G.ndim):
                g = G.reshape(points[:, i])
                assert np.min(g) == np.min(G.axes[i]) + offset[i]
                assert np.max(g) == np.max(G.axes[i]) + offset[i]


def test_gridSampling():
    G = grid(((0, 0), (200, 100)), size=(10, 10))

    rng = np.random.default_rng(0)
    np.random.seed(rng.integers(0, 2**31))
    samples = G.sample(N=20)
    assert samples.shape == (20, 2)
    assert len(np.unique(samples, axis=0)) == 20

    radius = 0.4
    max_points = 30
    poisson = G.sample(poissonDisk=(radius, max_points, 99))
    assert len(poisson) <= max_points
    assert len(np.unique(poisson, axis=0)) == len(poisson)
    assert _pairwise_min_distance(poisson) >= radius - 1e-12

    poisson2 = G.sample(poissonDisk=(radius, max_points, 99))
    assert np.array_equal(poisson, poisson2)

    idx = poisson_disk_indices_3d(
        rng.random(500) * 10.0,
        rng.random(500) * 10.0,
        rng.random(500) * 10.0,
        radius=radius,
        max_points=max_points,
        seed=99,
    )
    assert len(idx) == max_points
    assert len(np.unique(idx)) == len(idx)


def test_gridTransform():
    dims = (200, 100)
    center = (dims[0] / 2, dims[1] / 2)
    G = Grid(dims, step=(10, 10), center=center)

    local = G.coords(transform=False)
    world = G.coords(transform=True)
    for i in range(G.ndim):
        assert np.min(local[:, i]) == np.min(G.axes[i])
        assert np.max(local[:, i]) == np.max(G.axes[i])
        assert np.min(world[:, i]) == np.min(G.axes[i]) + G.center[i]
        assert np.max(world[:, i]) == np.max(G.axes[i]) + G.center[i]

    theta = np.pi / 2
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    G_rot = Grid(dims, step=(10, 10), center=center, rotation=rotation)
    local_rot = G_rot.coords(transform=False)
    world_rot = G_rot.coords(transform=True)
    expected = (np.hstack([local_rot, np.ones((len(local_rot), 1))]) @ G_rot.matrix.T)[:, :-1]
    assert np.allclose(world_rot, expected)

    copied = G.copy()
    assert copied.shape == G.shape
    assert np.allclose(copied.coords(), G.coords())


def test_nonAxisAlignedGrid():
    center = np.array([100.0, 50.0])
    dims = (100, 100)
    step = (25, 25)
    theta = np.pi / 2
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    G = Grid(dims, step=step, center=center, rotation=rotation)
    local = G.coords(transform=False)
    world = G.coords(transform=True)

    expected = (np.hstack([local, np.ones((len(local), 1))]) @ G.matrix.T)[:, :-1]
    assert np.allclose(world, expected)

    origin_ix = np.where((local == 0).all(axis=1))[0]
    assert len(origin_ix) == 1
    assert np.allclose(world[origin_ix[0]], center)

    x_step_ix = np.where((local[:, 0] == step[0]) & (local[:, 1] == 0))[0]
    assert len(x_step_ix) == 1
    assert np.allclose(world[x_step_ix[0]], rotation @ np.array([step[0], 0.0]) + center)

    y_step_ix = np.where((local[:, 0] == 0) & (local[:, 1] == step[1]))[0]
    assert len(y_step_ix) == 1
    assert np.allclose(world[y_step_ix[0]], rotation @ np.array([0.0, step[1]]) + center)

    # rotation preserves spacing between neighbouring cells
    assert np.isclose(
        np.linalg.norm(world[x_step_ix[0]] - world[origin_ix[0]]),
        step[0],
    )
    assert np.isclose(
        np.linalg.norm(world[y_step_ix[0]] - world[origin_ix[0]]),
        step[1],
    )
