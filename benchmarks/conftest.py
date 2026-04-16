"""Pytest configuration. Writes benchmark memory CSV at end of session if any records were collected."""
import sys
from pathlib import Path


def pytest_sessionfinish(session, exitstatus):
    """After all tests, write benchmark memory results to .benchmarks/ (same folder/naming as benchmark JSON/CSV)."""
    try:
        # Use the same module instance the benchmark tests imported (so _BENCHMARK_MEMORY_RECORDS has data)
        benchmarks_dir = Path(__file__).resolve().parent
        if str(benchmarks_dir) not in sys.path:
            sys.path.insert(0, str(benchmarks_dir))
        import benchmark_memory
        benchmark_memory.write_memory_csv()
    except Exception as e:
        print(f"benchmark memory CSV write failed: {e}", file=sys.stderr)
        raise
