# Premier League Predictor

Weekly win / draw / loss predictor for Premier League matches.
Built as a learning project — the model is implemented from scratch, no black-box ML libraries.

## Approach

**v1: Dixon–Coles Poisson model.** Each team gets a learned attack strength and
defense strength (0 = league average), plus one global home-advantage term and the
`rho` low-score correction from the Dixon–Coles (1997) paper. Parameters are fit by
maximum likelihood (L-BFGS-B with a hand-derived analytic gradient). For each
fixture the model builds a scoreline probability matrix (0–10 goals each side) and
sums it into P(home win) / P(draw) / P(away win).

**Principle:** bookmaker odds are used only as a benchmark, never as a model input.

## Project layout

```
src/
  load_data.py    # loads football-data.co.uk CSVs into one chronological table
  dixon_coles.py  # the model: Poisson log-PMF, tau() correction, MLE fit
  predict.py      # scoreline matrix -> win/draw/loss probabilities
  backtest.py     # walk-forward validation harness
data/raw/         # match results, 2021-22 through 2026-27 (football-data.co.uk)
outputs/          # backtest predictions and metrics (generated, not committed)
```

## Data

Historical results from [football-data.co.uk](https://www.football-data.co.uk)
(seasons 2021-22 to 2026-27, 1,950 games, 29 teams). Refresh with:

```
cd data/raw
for s in 2122 2223 2324 2425 2526 2627; do
  curl -sL -o E0_$s.csv "https://www.football-data.co.uk/mmz4281/$s/E0.csv"
done
```

## Usage

```
pip install -r requirements.txt
cd src
python load_data.py   # sanity checks: no duplicates, no leakage of future games
python backtest.py    # walk-forward validation, model vs baselines
```

## v1 results

Walk-forward backtest on 2026-27 matchweeks 1–5 (50 games; each week fit only on
games played strictly before it):

|            | Accuracy | Log loss | Brier |
|------------|----------|----------|-------|
| Model (v1) | 0.40     | 1.151    | 0.640 |
| Bookmakers | 0.44     | 1.052    | 0.634 |
| Always-home| 0.36     | —        | —     |

Competitive with bookmakers on proper scoring rules in weeks 1–4. Week 5 exposed
the main weakness: promoted Coventry (0 wins) got an extreme −5.78 attack rating
from 4 games, was given ~0.4% to win at Nott'm Forest, won anyway — and the
overconfident miss dominated the week's log loss. Draws are also underpredicted
(0.237 predicted vs 0.320 actual).

## Roadmap (v2)

1. **Shrinkage** — L2/Gaussian prior pulling team strengths toward league average,
   so tiny samples can't produce extreme ratings.
2. **Time-decay weighting** — recent games count more (exponential decay, as in the
   original Dixon–Coles paper).
3. **Elo challenger** — a second model to compare against Dixon–Coles; keep the winner.
