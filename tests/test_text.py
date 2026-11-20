"""Tests for curlew.text summaries (Geode, CSet, GeoEvent, GeoModel, fields, Pebble)."""

import numpy as np
import torch
import curlew
from curlew.core import Pebble
from curlew.text import to_text


def _assert_valid_text(text, label, min_len=10, required=()):
    assert isinstance(text, str), f"{label}: expected str, got {type(text)}"
    assert len(text.strip()) >= min_len, f"{label}: summary too short ({len(text)} chars)"
    for token in required:
        assert token in text, f"{label}: expected {token!r} in summary"


def test_toText():
    """Synthetic hutton model: major curlew types produce non-empty text summaries."""
    from curlew.synthetic import hutton, extract_constraints

    curlew.default_dim = 2
    shape = (120, 80) # N.B. this intensionally creates warnings in the model, as this is completely below the unconformity base! 

    M = hutton(shape, breaks=4)
    _assert_valid_text(str(M), "GeoModel", required=("GeoModel", "s0"))
    _assert_valid_text(
        str(M.grid),
        "Grid",
        required=("Grid", "voxels per axis", "resolution", "bounds", "center"),
    )
    assert str(M.grid) == to_text(M.grid)

    for name in ("s0", "s1", "d1"):
        ev = M[name]
        _assert_valid_text(str(ev), f"GeoEvent[{name}]", required=(name, "GeoEvent"))
        _assert_valid_text(str(ev.field), f"field[{name}]", required=("Scalar field",))
        assert str(ev) == to_text(ev)

    Cs = extract_constraints(M, ["s0", "s1"])
    for cname, C in Cs.items():
        _assert_valid_text(str(C), f"CSet[{cname}]", required=("CSet",))
        _assert_valid_text(to_text(C), f"to_text(CSet[{cname}])", required=("CSet",))

    geode = M.predict(M.grid)
    _assert_valid_text(str(geode), "Geode", required=("Model prediction summary",))
    _assert_valid_text(geode.summary(), "Geode.summary", required=("Volumes",))
    assert str(geode) == to_text(geode)
    assert "GeoModel" in str(M) or "s1" in str(M)

    pebble = Pebble(
        losses={"s0": {"grad": torch.tensor(2.0)}, "s1": {"mono": torch.tensor(0.5)}},
        weights={"s0": {"grad": 1.0}, "s1": {"mono": 0.01}},
    )
    _assert_valid_text(str(pebble), "Pebble", required=("L=", "grad"))
    assert str(pebble) == to_text(pebble)

    detached = pebble.detach()
    _assert_valid_text(str(detached), "Pebble(detached)", required=("L=",))

    # Geode warnings for negligible lithology (small grid)
    from curlew.core import Geode
    from curlew.geometry import Grid

    G = Grid((4, 4), step=(1.0, 1.0), center=(0.0, 0.0))
    litho_sparse = np.zeros(16, dtype=int)
    litho_sparse[15] = 2
    g_warn = Geode(
        grid=G,
        lithoID=litho_sparse,
        lithoLookup={0: "A", 2: "Rare"},
        structureID=np.zeros(16, dtype=int),
        structureLookup={0: "S0"},
    )
    text_warn = g_warn.summary(volume_fraction_warn=0.1)
    assert "WARNING" in text_warn
    assert "Rare" in text_warn
