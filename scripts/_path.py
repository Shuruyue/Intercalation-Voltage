"""Lets the scripts import the package from a source checkout without installing it."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
