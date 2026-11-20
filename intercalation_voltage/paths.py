"""Locations inside the repository; every script runs from anywhere."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CALC = ROOT / "calc"
FIGURES = ROOT / "docs" / "figures"
WEB = ROOT / "web"

RAW_CSV = DATA / "mp_insertion_electrodes_2019-12-03.csv"
CLEAN_CSV = DATA / "clean.csv"
FEATURES = DATA / "features.npz"


def result(name):
    return RESULTS / name
