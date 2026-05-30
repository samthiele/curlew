#!/usr/bin/env python3
"""Inject a version switcher banner into pdoc HTML pages for GitHub Pages."""

from __future__ import annotations

import os
import sys
from pathlib import Path

MARKER = "curlew-doc-version-bar"


def _pages_hrefs() -> tuple[str, str]:
    base = os.environ.get("PAGES_BASE", "/curlew").rstrip("/")
    return f"{base}/", f"{base}/dev/"


def banner_html(active: str) -> str:
    stable_href, dev_href = _pages_hrefs()
    stable_active = active == "stable"
    stable_color = "#fff" if stable_active else "#7eb8ff"
    dev_color = "#fff" if not stable_active else "#7eb8ff"
    stable_weight = "600" if stable_active else "400"
    dev_weight = "600" if not stable_active else "400"
    stable_decoration = "none" if stable_active else "underline"
    dev_decoration = "none" if not stable_active else "underline"
    return (
        f'<div id="{MARKER}" style="position:sticky;top:0;z-index:9999;'
        "background:#1a1a2e;color:#eee;padding:8px 16px;"
        "font-family:system-ui,sans-serif;font-size:14px;"
        'border-bottom:1px solid #444;display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
        '<span style="opacity:0.85;">curlew documentation</span>'
        '<span style="opacity:0.5;">|</span>'
        f'<a href="{stable_href}" style="color:{stable_color};text-decoration:{stable_decoration};'
        f'font-weight:{stable_weight};">Stable (main)</a>'
        f'<a href="{dev_href}" style="color:{dev_color};text-decoration:{dev_decoration};'
        f'font-weight:{dev_weight};">Development (dev)</a>'
        "</div>"
    )


def inject_file(path: Path, banner: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    for needle in ("<body>", "<body ", "<BODY>"):
        idx = text.find(needle)
        if idx == -1:
            continue
        close = text.find(">", idx)
        if close == -1:
            continue
        path.write_text(text[: close + 1] + banner + text[close + 1 :], encoding="utf-8")
        return True
    print(f"warning: no <body> tag in {path}", file=sys.stderr)
    return False


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: inject_docs_banner.py <html-root> <stable|dev>", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1])
    active = sys.argv[2]
    if active not in ("stable", "dev"):
        print("active must be 'stable' or 'dev'", file=sys.stderr)
        sys.exit(2)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    banner = banner_html(active)
    n = 0
    for html in root.rglob("*.html"):
        if inject_file(html, banner):
            n += 1
    print(f"injected banner into {n} file(s) under {root} ({active})")


if __name__ == "__main__":
    main()
