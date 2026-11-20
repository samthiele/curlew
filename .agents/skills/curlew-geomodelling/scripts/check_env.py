#!/usr/bin/env python3
"""Probe that torch and curlew are importable. Exit 0 on success, 1 with MISSING_DEPS."""
import sys

missing = []
try:
    import torch  # noqa: F401
except ImportError:
    missing.append("pytorch")
try:
    import curlew  # noqa: F401
except ImportError:
    missing.append("curlew")

if missing:
    print("MISSING_DEPS: " + ", ".join(missing))
    sys.exit(1)

print("SUCCESS: curlew and torch available.")
print("PYTHON:", sys.executable)
