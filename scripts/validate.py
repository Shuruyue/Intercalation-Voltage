"""Run every check of VALIDATION.md; exit code 1 if any fails.

Usage: python scripts/validate.py [--markdown]      (--markdown rewrites VALIDATION.md)
"""

import sys

import _path  # noqa: F401

from intercalation_voltage import paths, validation

if __name__ == "__main__":
    rows = validation.run_all()
    for r in rows:
        print(f"{r['result']:4s}  {r['category']:7s} {r['check'][:90]:90s} {r['value']:.4g}")
    if "--markdown" in sys.argv:
        (paths.ROOT / "VALIDATION.md").write_text(validation.markdown(rows), encoding="utf-8", newline="\n")
    sys.exit(1 if any(r["result"] == "FAIL" for r in rows) else 0)
