"""All figures of the README -> docs/figures/ (a few seconds).

Usage: python scripts/make_figures.py
"""

import _path  # noqa: F401

from intercalation_voltage import plotting

if __name__ == "__main__":
    for f in plotting.ALL:
        f()
        print("done", f.__name__)
