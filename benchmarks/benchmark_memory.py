"""
Manual peak-memory tracking for benchmark tests. Results are written to .benchmarks/
in the same folder and naming convention as the benchmark JSON/CSV (e.g. <stem>_memory.csv).
"""
import csv
import sys
from pathlib import Path

_BENCHMARK_MEMORY_RECORDS = []


def _default_memory_csv_path():
    """Path under .benchmarks/ matching the latest benchmark run (same folder and naming as JSON/CSV)."""
    root = Path(__file__).resolve().parent.parent / ".benchmarks"
    if not root.is_dir():
        return root / "memory.csv"
    latest_json = None
    for j in root.rglob("*.json"):
        if latest_json is None or j.stat().st_mtime >= latest_json.stat().st_mtime:
            latest_json = j
    if latest_json is None:
        return root / "memory.csv"
    return latest_json.parent / f"{latest_json.stem}_memory.csv"


def get_peak_memory_mb():
    """
    Return current process peak resident set size in MB (Unix/macOS).
    On Windows this returns 0.0 (resource module is Unix-only).
    """
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes, Linux: KB
        if sys.platform == "darwin":
            return rss / (1024 * 1024)
        return rss / 1024  # KB -> MB
    except (ImportError, OSError):
        return 0.0


def record_benchmark_memory(test_name, peak_mb):
    """Append a benchmark memory record for later writing to CSV."""
    _BENCHMARK_MEMORY_RECORDS.append({"name": test_name, "peak_mb": peak_mb})


def write_memory_csv(path=None):
    """
    Write collected benchmark memory records to .benchmarks/ (same folder/naming as benchmark JSON/CSV).
    Default: same platform dir as latest benchmark JSON, file named <stem>_memory.csv.
    """
    if path is None:
        path = _default_memory_csv_path()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(_BENCHMARK_MEMORY_RECORDS)
    _BENCHMARK_MEMORY_RECORDS.clear()
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "peak_mb"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")
