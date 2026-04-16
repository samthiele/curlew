"""
pytest-benchmark tests for Michell (fault) workflow: forward, inverse, predict.

Run: mamba activate curlew && pytest benchmarks/ --benchmark-only
"""
import sys
from pathlib import Path

_benchmarks_dir = Path(__file__).resolve().parent
if str(_benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(_benchmarks_dir))
from benchmark_memory import record_benchmark_memory, get_peak_memory_mb

import numpy as np
from curlew.geology.geomodel import GeoModel
from curlew.geometry import grid
from curlew.fields.fourier import NFF
import curlew
from curlew.synthetic import michell
from curlew import HSet
from curlew.geology import strati, fault

curlew.default_dim = 2
dims = (2000, 1000)
nepoch = 50

# Shared results so later benchmarks reuse outputs (no recalculation)
_michell_C = None
_michell_M = None

# Shared constraint grid so later benchmarks reuse it (no recalculation)
G = grid(dims, step=(10, 10), center=(dims[0] / 2, dims[1] / 2), sampleArgs=dict(N=1024))


def test_benchmark_01_forward_michell(benchmark, request):
    """Benchmark: generate synthetic Michell (C) and constraint grid. Result feeds inverse."""
    def forward_michell():
        C, _ = michell(dims, offset=225)
        C = C[:-1]  # drop value constraints as they're not needed
        for _c in C:
            _c.grid = G
            _c.delta = 10
        return C

    global _michell_C
    _michell_C = benchmark(forward_michell)
    record_benchmark_memory(request.node.name, get_peak_memory_mb())


def test_benchmark_02_inverse_michell(benchmark, request):
    """Benchmark: build fault model, prefit, freeze fault geometry, fit slip. Result feeds predict."""
    global _michell_C, _michell_M
    assert _michell_C is not None, "Run test_benchmark_01_forward_michell first"
    C = _michell_C

    def inverse_michell():
        H = HSet(value_loss=1, grad_loss=1, mono_loss='0.1', thick_loss="1.0")
        s0 = strati('basement', C=C[0], H=H, type=NFF, base=-np.inf,
                    hidden_layers=[16], rff_features=32, length_scales=[2000])
        H = HSet(value_loss=1, grad_loss=1, mono_loss="0.01")
        s1 = fault('fault', C=C[1], H=H, type=NFF, shortening=(-1, 0),
                   offset=(250, 0, 300), width=0,
                   hidden_layers=[16], rff_features=32, length_scales=[6000])
        M = GeoModel([s0, s1])
        M.prefit(epochs=nepoch, best=True, vb=False, early_stop=None)
        M.freeze(s1, geometry=True, params=False)
        M.fit(epochs=nepoch, learning_rate=0.1, early_stop=None)
        return M

    _michell_M = benchmark(inverse_michell)
    record_benchmark_memory(request.node.name, get_peak_memory_mb())


def test_benchmark_03_predict_michell(benchmark, request):
    """Benchmark: predict on grid using fitted fault model from inverse (no refit)."""
    global _michell_M
    assert _michell_M is not None, "Run test_benchmark_02_inverse_michell first"
    M = _michell_M
    G2 = grid(dims, step=(2, 2), center=(dims[0] / 2, dims[1] / 2), sampleArgs=dict(N=1024))
    sxy = G2.coords()
    benchmark(lambda: M.predict(sxy))
    record_benchmark_memory(request.node.name, get_peak_memory_mb())
