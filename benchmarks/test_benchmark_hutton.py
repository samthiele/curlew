"""
pytest-benchmark tests for geology workflows (Hutton: forward, inverse, predict).

Run: mamba activate curlew && pytest benchmarks/ --benchmark-only
"""
import csv
import json
import numpy as np
import sys
from pathlib import Path

_benchmarks_dir = Path(__file__).resolve().parent
if str(_benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(_benchmarks_dir))
from benchmark_memory import record_benchmark_memory, get_peak_memory_mb
from curlew.geology.geomodel import GeoModel
from curlew.geometry import grid
from curlew.fields.fourier import NFF
import curlew
from curlew.synthetic import hutton
from curlew import HSet
from curlew.geology import strati

curlew.default_dim = 2
dims = (2000, 1000)
nepoch = 50

# Shared results so later benchmarks reuse outputs (no recalculation)
_hutton_C_Ms = None
_hutton_M = None

# Shared constraint grid so later benchmarks reuse it (no recalculation)
G = grid(dims, step=(10, 10), center=(dims[0] / 2, dims[1] / 2), sampleArgs=dict(N=1024))


def benchmark_results_to_csv(benchmarks_root=None):
    """
    Load all pytest-benchmark JSON files under .benchmarks and write friendly CSV files.

    Each JSON file is converted to a CSV in the same directory with columns:
    name, mean_s, stddev_s, min_s, max_s, median_s, rounds, ops_per_sec, total_s,
    run_id, machine, python_version.

    Parameters
    ----------
    benchmarks_root : path-like, optional
        Root directory containing platform subdirs (e.g. .benchmarks). Default: repo root /.benchmarks.
    """
    if benchmarks_root is None:
        benchmarks_root = Path(__file__).resolve().parent.parent / ".benchmarks"
    benchmarks_root = Path(benchmarks_root)
    if not benchmarks_root.is_dir():
        return
    for json_path in benchmarks_root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        benchmarks = data.get("benchmarks")
        if not benchmarks:
            continue
        machine = data.get("machine_info", {})
        run_id = json_path.stem
        csv_path = json_path.with_suffix(".csv")
        fieldnames = [
            "name", "mean_s", "stddev_s", "min_s", "max_s", "median_s",
            "rounds", "ops_per_sec", "total_s", "run_id", "machine", "python_version",
        ]
        rows = []
        for b in benchmarks:
            stats = b.get("stats", {})
            rows.append({
                "name": b.get("name", ""),
                "mean_s": stats.get("mean", ""),
                "stddev_s": stats.get("stddev", ""),
                "min_s": stats.get("min", ""),
                "max_s": stats.get("max", ""),
                "median_s": stats.get("median", ""),
                "rounds": stats.get("rounds", ""),
                "ops_per_sec": stats.get("ops", ""),
                "total_s": stats.get("total", ""),
                "run_id": run_id,
                "machine": machine.get("node", ""),
                "python_version": machine.get("python_version", ""),
            })
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {csv_path}")


def test_benchmark_01_forward_hutton(benchmark, request):
    """Benchmark: generate synthetic (C, Ms) and constraint grid. Result feeds inverse."""
    def forward_hutton():
        C, Ms = hutton(dims, breaks=10, cmap='prism', pval=1.0)
        for _c in C:
            _c.grid = G
            _c.delta = 10
        return (C, Ms)

    global _hutton_C_Ms
    _hutton_C_Ms = benchmark(forward_hutton)
    record_benchmark_memory(request.node.name, get_peak_memory_mb())


def test_benchmark_02_inverse_hutton(benchmark, request):
    """Benchmark: build model and prefit using (C, Ms) from forward. Result feeds predict."""
    global _hutton_C_Ms
    assert _hutton_C_Ms is not None, "Run test_benchmark_01_forward_hutton first"
    C, Ms = _hutton_C_Ms
    def inverse_hutton():
        H = HSet(value_loss='1.0', mono_loss='0.01', thick_loss='1.0')
        s0 = strati('basement', C=C[0], H=H, type=NFF, base=-np.inf,
                    #hidden_layers=[], activation=None,rff_features=64, length_scales=[500 / 2 * np.pi])
                    hidden_layers=[32], rff_features=64, length_scales=[500 / 2 * np.pi])
        s1 = strati('unconformity', C=C[1], H=H.copy(mono_loss="1.0", thick_loss=1.0), type=NFF, base="base",
                    #hidden_layers=[], activation=None, rff_features=64, length_scales=[2000 / 2 * np.pi])
                    hidden_layers=[32], rff_features=64, length_scales=[2000 / 2 * np.pi])
        s1.isosurfaces = Ms['s1'].isosurfaces
        s0.isosurfaces = Ms['s0'].isosurfaces
        s1.addIsosurface("base", seed=Ms.fields[1].field.origin)
        M = GeoModel([s0, s1])
        M.prefit(epochs=nepoch, best=True, vb=False)
        return M
      
    global _hutton_M
    _hutton_M = benchmark(inverse_hutton)
    record_benchmark_memory(request.node.name, get_peak_memory_mb())

def test_benchmark_03_predict_hutton(benchmark, request):
    """Benchmark: predict on grid using fitted model from inverse (no refit)."""
    global _hutton_M
    assert _hutton_M is not None, "Run test_benchmark_02_inverse_hutton first"
    M = _hutton_M
    G2 = grid(dims, step=(2, 2), center=(dims[0] / 2, dims[1] / 2), sampleArgs=dict(N=1024))
    sxy = G2.coords()

    benchmark(lambda: M.predict(sxy))
    record_benchmark_memory(request.node.name, get_peak_memory_mb())


if __name__ == "__main__":
    benchmark_results_to_csv()
