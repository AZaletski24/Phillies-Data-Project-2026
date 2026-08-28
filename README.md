# Projecting Pitcher Strikeout Percentage, 2025

Phillies Research & Information · Quantitative Analyst Associate Trial Project

## The data

`data/k_2026.csv` is **not** in this repository. It is Phillies-supplied trial material and
is deliberately gitignored. To run the analysis, drop that file into `data/`, or point the
scripts at it directly:

```bash
python k_projection.py --data /path/to/k_2026.csv
```

The panel is one row per player-season with columns `Season, Name, PlayerId, Team, K%, TBF,
Stuff+, Age`; K% is a fraction, not a percentage. Every number in the committed outputs was
produced from the 4,371-row, 2021-2025 file as supplied.

## Run it

```bash
pip install -r requirements.txt
python k_projection.py              # fit, project, score, write outputs
python k_projection.py --self-test  # prove the 2025 rows never touch a fitted object
python build/make_report.py         # typeset output/methodology_2025.pdf from those outputs
python build/make_summary.py        # typeset the two-page output/summary_2025.pdf
```

Deterministic (fixed seed, no network). Runtime ~40s for the analysis, a few seconds for
the documents. All three scripts resolve their default paths relative to the repository,
so they can be run from any working directory.

## Deliverables

| File | What it is |
|---|---|
| `k_projection.py` | The analysis: data audit, model, validation, figures. Code with light documentation and inline citations. |
| `output/methodology_2025.pdf` | The written methodology, in full. Standalone document. |
| `output/summary_2025.pdf` | Two-page executive summary: problem, model, held-out result, caveats. |
| `output/k_pct_projections_2025.csv` | 1,328 projections with 80% prediction intervals. |
| `output/backtest_2025.csv` | Rolling-origin backtest, every model × season × usage threshold, with the recency weights each fold selected from its own history. |
| `output/ablation_2025.csv` | Component ablation on the held-out season. |
| `output/run_manifest_2025.json` | Every fitted constant, for auditing. |
| `output/fig*.png` | Five figures. |

## The short version

K% is a binomial rate observed over wildly unequal samples (TBF 1 to 886; a third of
player-seasons fall below the ~70 BF stabilization point). The problem is therefore
how much of each pitcher's observed rate to believe, not which regressor to use.

The model is an empirical-Bayes projection in the Marcel/Steamer tradition, with the
regression constant **derived rather than assumed**: variance decomposition on the
panel gives **k of about 77 batters faced** of league-average prior, which independently
reproduces the published ~70 BF stabilization point.

**Held-out 2025 result** (459 pitchers, ≥100 BF; 2025 never used to fit anything):

| Model | RMSE | R² |
|---|---|---|
| League average | 5.29 pp | -0.000 |
| Last season's K%, unregressed | 5.72 pp | -0.169 |
| **Blend of EB + ridge (shipped)** | **4.04 pp** | **0.418** |

80% prediction intervals covered 81.5% of outcomes.

## On the target season

The instructions ask for 2025 while barring data from Opening Day 2025 onward, but the
file supplies a complete 2025 season. Those rows are treated strictly as an answer
key: read once, after projections are written, to score them. `--self-test` runs the
whole pipeline against a copy of the file with all 873 rows of 2025 deleted and
asserts the projections come out identical. They do, to 0.00e+00.
