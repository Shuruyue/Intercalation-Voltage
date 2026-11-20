"""All machine-learning experiments, then the distance check (about 20 min on one core: about 75 boosted-tree fits).

Usage: python scripts/run_ml.py      -> results/learning.json, results/distance.json, results/li_na_pairs.csv (+ .npz)
"""

import _path  # noqa: F401

from intercalation_voltage import distance, learning

if __name__ == "__main__":
    learning.run()
    distance.run()
