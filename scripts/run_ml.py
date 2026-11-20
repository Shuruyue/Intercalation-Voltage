"""All machine-learning experiments, the distance check and the screening (about 25 min on one core: about 85
boosted-tree fits).

Usage: python scripts/run_ml.py      -> results/learning.json, distance.json, screening.json, li_na_pairs.csv (+ .npz)
"""

import _path  # noqa: F401

from intercalation_voltage import distance, learning, screening

if __name__ == "__main__":
    learning.run()
    distance.run()
    screening.run()
