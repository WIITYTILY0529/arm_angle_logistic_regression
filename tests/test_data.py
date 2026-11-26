from pathlib import Path

from arm_angle import MATCHUPS, load_matchup_data, standardize_series


def test_matchup_data_and_standardization():
    project_root = Path(__file__).resolve().parents[1]
    frame = load_matchup_data(project_root / "data")
    assert set(frame["matchup"].unique()) == set(MATCHUPS)
    standardized = standardize_series(frame["arm_angle"].dropna())
    assert abs(float(standardized.mean())) < 1e-10
