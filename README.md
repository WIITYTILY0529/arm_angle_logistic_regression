# Arm Angle and Pitcher Movement Profiles

This project quantifies whether each pitcher's arsenal leans toward horizontal or vertical movement, and measures how much of that tendency is explained by arm angle. It uses Statcast data from 2020–2025, with each pitcher-season split by batter handedness.

This work was presented as a poster at the SABR Analytics Conference in March 2026: [`writing/poster/sabr_analytics_2026_poster.pdf`](writing/poster/sabr_analytics_2026_poster.pdf).

**Key result:** arm angle is clearly but only partly related to movement profile. An arm angle of about 38.6° best separates vertical- from horizontal-dominant pitchers, with 67.7% accuracy. In a GAM, arm angle explains about 46% of the variation in how a pitcher's most-different pitches separate (separation bias, pseudo-R² 0.46), and about 22–26% of the usage-weighted movement tendency (movement bias). Both relationships have p < 0.001. Other factors, such as seam orientation, supination and pronation, and seam-shifted wake, account for the rest.

## Research question

1. How can a pitcher's overall movement tendency (horizontal vs. vertical) be summarized in a single number?
2. How much does arm angle determine that tendency?

Arm angle shapes the spin axis and therefore movement, but grip and forearm action also matter and are hard to measure. This project isolates arm angle to see how much it alone explains.

## Data

| Input | Description | Location |
| --- | --- | --- |
| Pitcher × pitch-type aggregates | Statcast regular seasons 2020–2025, one row per pitcher-season and pitch type, for pitcher-seasons with at least 400 pitches. Split into four handedness matchups (`l_vs_l`, `l_vs_r`, `r_vs_l`, `r_vs_r`), with one CSV per pitch type (`ff`, `si`, `fc`, `sl`, `st`, `cu`, `ch`, `fs`). Key columns: `pitches`, `arm_angle`, `api_break_x_arm`, `api_break_z_induced`. | `data/<matchup>/<matchup>_<pitch>.csv` |
| Earlier summary table | One row per pitcher with per-pitch-type columns (not used by the current scripts) | `data/processed_summary.csv` |

The 400-pitch minimum is roughly 4–5 starts or 30–40 relief outings. Slow curves and knuckle curves are grouped as curveballs, forkballs as splitters, and sliders and sweepers are kept separate. Each pitcher-season appears once per batter handedness, because pitch mix changes by matchup; this gives 5,718 pitcher-season-matchup rows.

The CSVs are Baseball Savant Statcast search exports, kept as exported. See [`data/README.md`](data/README.md). Statcast data comes from [Baseball Savant](https://baseballsavant.mlb.com/); follow MLB's terms of use before redistributing it.

## Method

### Movement scores

1. For each pitch type, standardize horizontal (`api_break_x_arm`) and vertical (`api_break_z_induced`) movement against all pitchers who throw that pitch.
2. Flip signs so that positive means movement in the pitch's intended direction (for example, more rise on a four-seamer, more drop on a curveball, more sweep on a slider).
3. Multiply by each pitch's usage rate and sum: **H score** (horizontal) and **Z score** (vertical). Frequently thrown pitches with strong movement contribute most.

### Separation scores

For each pitcher-season-matchup, take the largest horizontal and the largest vertical movement difference between any two of the pitcher's pitch types, and standardize each across all rows.

### Bias (softsign)

Each pair of scores is reduced to a single value with the softsign function, Bias = d / (1 + |d|), where d is horizontal minus vertical. Bias ranges from −1 (vertical-dominant) to +1 (horizontal-dominant).

- **Movement bias:** from the H and Z scores.
- **Separation bias:** from the separation scores.

### Models

| Analysis | Model | Script |
| --- | --- | --- |
| Pitcher type vs. arm angle | Logistic regression (balanced class weights) of vertical-dominant (Z ≥ H) vs. horizontal-dominant on average arm angle, all matchups pooled; the decision boundary is −intercept / slope | `scripts/run_integrated_logistic.py` |
| Movement bias vs. arm angle | GAM (`pygam` LinearGAM, 25 splines, λ chosen by grid search) | `scripts/run_softsign_gam.py` |
| Separation bias vs. arm angle | Same GAM specification | `scripts/run_separation_gam.py` |
| Single pitch types | GAM (20 splines) of softsign(standardized H − V) per pitch type, without usage weighting (the poster's supplementary analysis); a usage-weighted variant is also kept | `scripts/movement_bias_unweighted.py`; variant `scripts/movement_bias_weighted.py` |
| H and Z separately | GAMs of each score on arm angle | `scripts/run_horizontal_gam.py`, `scripts/run_vertical_gam.py` |

An earlier approach in the report reduced the H and Z scores to one dimension with PCA and modelled the first component with a GAM (`experiments/gam/`).

## Results

| Result | Value | Source |
| --- | --- | --- |
| Logistic decision boundary | 38.62° | Poster; reproduced by `run_integrated_logistic.py` (38.6246°) |
| Logistic accuracy (in-sample) | 0.677 | Poster; reproduced (0.6775) |
| Separation-bias GAM pseudo-R² | 0.46, p < 0.0001 | Poster; reproduced by `run_separation_gam.py` (0.4643, n = 5,718) |
| Movement-bias GAM pseudo-R² | 0.225 in the poster text; 0.264 in the poster figure; p < 0.0001 | Poster; `run_softsign_gam.py` reproduces 0.264 (see the note below) |
| PCA (earlier analysis) | PC1 loadings [0.71, −0.71], about 60% of variance; arm angle explains about 27% of PC1 | [`writing/reports/Report.docx`](writing/reports/Report.docx) |

**Note on the movement-bias GAM:** the poster reports two different values for this model. Its text gives a pseudo-R² of 0.225, which comes from softsign of the *standardized* H and Z scores (reproduced: 0.2246). Its movement-bias figure shows R² = 0.264, which comes from softsign of the *raw* H − Z difference; this is what the current `run_softsign_gam.py` computes (reproduced: 0.2636). Both describe the same relationship with different scaling.

Interpretation (from the poster):

- **Movement bias:** above 0° arm angle, higher arm angles go with more vertical movement. Below 0° the curve trends toward horizontal, but the sample there is small.
- **Separation bias:** the curve is smooth and monotonic in slope, suggesting a stable structural relationship rather than noise.
- **Logistic regression:** it uses only which score is larger, not by how much, and it partly hides very low-arm-angle pitchers, who often have a higher Z score than H score.
- **Single pitch types:** fastballs and curveballs show the strongest relationship with arm angle; most other pitch types show only weak associations.

| Figure folder | Content |
| --- | --- |
| `outputs/figures/logistic/` | Logistic regression of pitcher type on arm angle (`integrated_logi_reg.png`) and earlier versions |
| `outputs/figures/movement_bias/` | Movement-bias GAM (`softsign gam.png`) and per-pitch-type GAMs |
| `outputs/figures/gam/` | H and Z score distributions |
| `outputs/figures/exploratory/` | Separation-bias GAM (`hzmaxdiffstdgam.png`), polynomial regression, FF/CU comparison |
| `outputs/figures/player_profiles/` | Example pitcher movement profiles (Joe Ryan, Paul Skenes) |

## Reproduce

Requires Python 3.10 or newer (developed with Python 3.12 on Windows).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\run_integrated_logistic.py
.\.venv\Scripts\python.exe scripts\run_softsign_gam.py
.\.venv\Scripts\python.exe scripts\run_separation_gam.py
.\.venv\Scripts\python.exe -m pytest
```

On macOS or Linux, use `.venv/bin/python`. Run the commands from the project root. Each script takes from a few seconds to about a minute.

| Script | Output |
| --- | --- |
| `scripts/run_integrated_logistic.py` | Prints the boundary and accuracy; writes `outputs/figures/logistic/integrated_logi_reg.png` |
| `scripts/run_softsign_gam.py` | Prints the GAM summary, p-value, and pseudo-R²; opens a plot window |
| `scripts/run_separation_gam.py` | Prints n, pseudo-R², and p-value; writes `outputs/figures/exploratory/hzmaxdiffstdgam.png` |
| `scripts/run_horizontal_gam.py`, `run_vertical_gam.py` | GAM of the H or Z score; plot window |
| `scripts/movement_bias_unweighted.py` | Per-pitch-type movement bias GAMs without usage weighting, with an FF and CU detail plot; plot window |
| `scripts/movement_bias_weighted.py` | The same with usage weighting; plot window |
| `scripts/compare_scores.py` | Pitchers whose H/V tendency flips between matchups; console |
| `scripts/generate_player_profiles.py` | Movement profile plots for individual pitchers; `outputs/figures/player_profiles/` |
| `scripts/plot_score_density.py` | H and Z score distributions; `outputs/figures/gam/h_and_v_density.png` |

The two scripts in `experiments/gam/` reproduce the earlier PCA analysis. They read from `data/` but are not maintained as part of the main workflow.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/` | Statcast aggregates by handedness matchup and pitch type |
| `src/arm_angle/` | Shared data loading and standardization helpers |
| `scripts/` | Main analyses |
| `experiments/gam/` | Earlier PCA + GAM analysis, for left-handed (`l_vs_r`) and right-handed (`r_vs_l`) pitchers |
| `outputs/figures/` | Figures |
| `tests/` | Tests for the shared helpers |
| `writing/` | Report, article draft, abstracts, presentation, and poster |

| Writing | Contents |
| --- | --- |
| `writing/reports/Report.docx` | Research report |
| `writing/reports/Fangraphs.docx` | Article draft |
| `writing/abstracts/Abstract.docx`, `Abstract_final.docx` | Abstract drafts |
| `writing/presentation/Presentation abstract.pptx`, `.pdf` | Slides with speaker notes |
| `writing/presentation/Script.docx`, `Script_Eng.docx` | Talk script |
| `writing/poster/poster.pptx`, [`sabr_analytics_2026_poster.pdf`](writing/poster/sabr_analytics_2026_poster.pdf) | Poster presented at the SABR Analytics Conference, March 2026 |

## Limitations

- Arm angle is the only explanatory variable. Seam orientation, supination and pronation, and seam-shifted wake are not modelled.
- The logistic accuracy is in-sample, and its binary label ignores how large the H–Z difference is.
- The same pitcher appears in several seasons and in both matchups, so rows are not independent; the GAM p-values (which `pygam` also warns are approximate after smoothing selection) are therefore optimistic.
- Very low arm angles (below 0°) have few pitchers, so the curves there are uncertain.
- Average arm angle is the mean over pitch types, not weighted by usage.

## References

- Baseball Savant, Statcast pitch movement and arm angle.
- Servén and Brummitt, *pyGAM: Generalized Additive Models in Python* (2018).
- Hastie and Tibshirani, *Generalized Additive Models* (1990).

## License

The code in this repository is released under the [MIT License](LICENSE). Statcast data remains subject to the terms of its original source.
