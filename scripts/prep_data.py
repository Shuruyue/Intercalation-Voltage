"""Clean the Materials Project snapshot and build the feature matrix (< 1 min).

Usage: python scripts/prep_data.py      -> data/clean.csv, data/features.npz, results/data_log.json
"""

import json

import _path  # noqa: F401

from intercalation_voltage import data

if __name__ == "__main__":
    print(json.dumps(data.build(), indent=1))
