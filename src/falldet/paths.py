"""Where data and reports live (``data/`` is gitignored; see AGENTS.md for the raw dataset layout)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
