import numpy as np
import pytest
import curlew


def test_HSet():
    from curlew import ccmap, HSet

    assert len(ccmap(0.5)) == 4

    H = HSet()
    assert H.value_loss != 0
    H = HSet().zero()
    assert H.value_loss == 0
    H = H.copy(value_loss=10)
    assert H.value_loss == 10

def test_CSet():
    from curlew import CSet

    C = CSet()
    C.vp = np.random.rand(10, 3)
    C.vv = np.random.rand(10)
    C.gp = np.random.rand(10, 3)
    C.gv = np.random.rand(10, 3)
    C.iq = (10, [([np.random.rand(10, 3), np.random.rand(10, 3), '<']) for i in range(3)])
    C.eq = [np.random.rand(10, 3), np.random.rand(8, 3)]
    C2 = C.torch()
    C3 = C2.numpy()

    for i in range(len(C2.iq[1])):
        assert np.mean(np.abs(C.iq[1][i][0] - C3.iq[1][i][0])) < 1e-6
        assert np.mean(np.abs(C.iq[1][i][1] - C3.iq[1][i][1])) < 1e-6
        assert C.iq[1][i][2] == C3.iq[1][i][2]
    for i in range(len(C2.eq)):
        assert np.mean(np.abs(C.eq[i] - C3.eq[i])) < 1e-6

    def t(p):
        return p + 1

    for C in [C2, C3]:
        C0 = C.transform(t)
        for a0, a1 in zip(
            [C0.vp, C0.gp, C0.iq[1][0][0], C0.iq[1][0][1], C0.eq[0]],
            [C.vp, C.gp, C.iq[1][0][0], C.iq[1][0][1], C.eq[0]],
        ):
            assert np.median(a0 - a1) == 1

        def f(p):
            mask = np.full(len(p), True)
            mask[0] = False
            return mask

        C0 = C.filter(f)
        for a0, a1 in zip(
            [C0.vp, C0.gp, C0.iq[1][0][0], C0.iq[1][0][1]],
            [C.vp, C.gp, C.iq[1][0][0], C.iq[1][0][1]],
        ):
            assert len(a1) - len(a0) == 1

def test_Geode():
    import torch
    from curlew.core import Geode
    from curlew.geometry import Grid

    # construct a Geode
    x = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    scalar = np.array([0.1, 0.5, 0.9])
    litho = np.array([1, 1, 2])
    struct = np.array([0, 0, 1])

    g = Geode(
        x=x,
        scalar=scalar,
        lithoID=litho,
        structureID=struct,
        lithoLookup={1: "A", 2: "B"},
        structureLookup={0: "s0", 1: "s1"},
        fields={"s0": scalar.copy()},
        crs="global",
    )
    assert len(g) == 3
    assert "global" in g.x

    # check it can be cast to torch (and back to numpy)
    g_torch = g.torch()
    assert isinstance(g_torch.coords(), torch.Tensor)
    g_np = g_torch.numpy()
    assert isinstance(g_np.coords(), np.ndarray)
    assert np.allclose(g_np.scalar, scalar)
    assert np.allclose(g_np.fields["s0"], scalar)

    # check combination works
    g_other = Geode(x=np.array([[3.0, 1.0, 0.0]]), scalar=np.array([0.2]), crs="global")
    combined = Geode.concat([g, g_other])
    assert len(combined) == 4
    assert np.allclose(combined.scalar, np.array([0.1, 0.5, 0.9, 0.2]))

    # check stack values function
    stacked = g.stackValues()
    assert stacked.scalar.shape == scalar.shape
    assert np.min(stacked.scalar) >= 0

    weight = torch.tensor([1.0, 0.0, 1.0])
    older = Geode(
        x=x,
        scalar=torch.zeros(3),
        structureID=torch.tensor(struct),
        lithoID=torch.tensor(litho),
    )
    younger = Geode(
        x=x,
        scalar=torch.ones(3),
        structureID=torch.tensor(struct),
        lithoID=torch.tensor(litho),
    )
    merged = older.combine(younger, weight)
    assert torch.allclose(merged.scalar, weight)
    assert torch.all(merged.structureID == torch.tensor(struct))

    grid = Grid((10, 10), step=(1, 1), center=(5, 5))
    g_grid = Geode(x=grid.coords(), grid=grid, scalar=np.arange(grid.coords().shape[0], dtype=float))
    assert g_grid.grid is grid
    assert g_grid.scalar.shape[0] == grid.coords().shape[0]

def test_transform():
    from curlew.geometry import Transform
    import numpy as np
    import torch

    # check initialisation with identity matrix
    T_id_np = Transform(2)
    pts2_np = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    assert np.allclose(pts2_np, T_id_np(pts2_np)) # should be no change!

    T_id_np = Transform(3)
    pts3_np = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0],
    ])
    assert np.allclose(pts3_np, T_id_np(pts3_np)) # should be no change!
    assert np.allclose(pts3_np, T_id_np.inverse()(T_id_np(pts3_np)))

    # -------------------------------------------------
    # NumPy 2D – translation
    # -------------------------------------------------
    T2_np = Transform(
        np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0,  5.0],
            [0.0, 0.0,  1.0],
        ])
    )

    pts2_np = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    expected2_np = np.array([
        [10.0,  5.0],
        [11.0,  7.0],
        [13.0,  9.0],
    ])

    out2_np = T2_np(pts2_np)
    assert np.allclose(out2_np, expected2_np)
    assert np.allclose(pts2_np, T2_np.inverse()(out2_np))

    # -------------------------------------------------
    # NumPy 3D – scaling
    # -------------------------------------------------
    T3_np = Transform(
        np.array([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    )
    pts3_np = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0],
    ])
    expected3_np = np.array([
        [2.0,  3.0,  4.0],
        [4.0,  9.0, 16.0],
    ])
    out3_np = T3_np.apply(pts3_np)
    assert np.allclose(out3_np, expected3_np)
    assert np.allclose(pts3_np, T3_np.inverse()(out3_np))

    # Torch 2D – translation
    T2_t = Transform(
        torch.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 1.0],
        ])
    )
    pts2_t = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0],
    ], device=curlew.device, dtype=curlew.dtype)
    expected2_t = torch.tensor([
        [3.0, 4.0],
        [4.0, 5.0],
    ], device=curlew.device, dtype=curlew.dtype)
    out2_t = T2_t(pts2_t)
    assert torch.allclose(out2_t, expected2_t)
    assert np.allclose(pts2_t, T2_t.inverse()(out2_t))

    # Torch 3D – translation
    T3_t = Transform(
        torch.tensor([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ], device=curlew.device, dtype=curlew.dtype) )
    pts3_t = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], device=curlew.device, dtype=curlew.dtype)
    expected3_t = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ], device=curlew.device, dtype=curlew.dtype)
    out3_t = T3_t.apply(pts3_t)
    assert torch.allclose(out3_t, expected3_t)
    assert np.allclose(pts3_t, T3_t.inverse()(out3_t))

def test_pebble():
    import torch
    import torch.nn as nn
    from curlew.core import Pebble

    p1 = nn.Parameter(torch.tensor(1.0))
    p2 = nn.Parameter(torch.tensor(2.0))
    opt1 = torch.optim.SGD([p1], lr=0.1)
    opt2 = torch.optim.SGD([p2], lr=0.1)

    pebble1 = Pebble()
    pebble1.push("g1", "data", p1 * p1, optim=opt1)

    pebble2 = Pebble()
    pebble2.push("g2", "reg", p2 * p2, weight=0.5, optim=opt2)

    total = pebble1 + pebble2
    assert set(total.losses) == {"g1", "g2"}
    assert total.weights["g2"]["reg"] == 0.5
    assert total.optim["g1"] is opt1
    assert total.optim["g2"] is opt2

    desc = str(total)
    assert desc.startswith("L=3 ")
    assert "g1/data=1" in desc
    assert "g2/reg=2" in desc
    assert "\n" not in desc

    total.zero()
    total.backward()
    assert p1.grad is not None and p2.grad is not None
    assert torch.allclose(p1.grad, torch.tensor(2.0))
    assert torch.allclose(p2.grad, torch.tensor(2.0))

    p1_before, p2_before = p1.detach().clone(), p2.detach().clone()
    total.step()
    assert not torch.allclose(p1, p1_before)
    assert not torch.allclose(p2, p2_before)
    assert torch.allclose(p1, torch.tensor(0.8))
    assert torch.allclose(p2, torch.tensor(1.8))

    snapshot = total.detach()
    assert snapshot.detached
    assert isinstance(snapshot.losses["g1"]["data"], np.ndarray)
    assert snapshot.optim == {"g1": None, "g2": None}
    assert str(snapshot).startswith("L=3 ")

    with pytest.raises(ValueError, match="detached and active"):
        total + snapshot
    with pytest.raises(RuntimeError, match="detached Pebble"):
        snapshot.backward()
    with pytest.raises(RuntimeError, match="detached Pebble"):
        snapshot.zero()
    with pytest.raises(RuntimeError, match="detached Pebble"):
        snapshot.step()

