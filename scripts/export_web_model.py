"""Export the Li model and the distance data for the page, with the Python check cases (about 30 s).

Usage: python scripts/export_web_model.py      -> build/web_model.json, results/web_check.json
"""

import _path  # noqa: F401

from intercalation_voltage import web_model

if __name__ == "__main__":
    M, cases = web_model.export()
    print(f"{len(M['trees'])} trees, {M['n_train']} x {M['n_keep']} training matrix, {len(cases)} check cases")
