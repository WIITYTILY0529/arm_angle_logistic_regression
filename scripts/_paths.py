"""Stable project paths for analysis entry points."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

for directory in (FIGURES_DIR / "logistic", FIGURES_DIR / "gam",
                  FIGURES_DIR / "movement_bias", FIGURES_DIR / "player_profiles"):
    directory.mkdir(parents=True, exist_ok=True)
