"""Print every number quoted in the README, computed from results/*.json and calc/ outputs.

Usage: python scripts/summary.py
"""

import _path  # noqa: F401

from intercalation_voltage import summary

if __name__ == "__main__":
    for k, v in summary.compute().items():
        print(f"{k:22s} {v}")
