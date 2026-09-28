# Premier League Predictor

Weekly win / draw / loss predictor for Premier League matches.
Built as a learning project — the model is implemented from scratch, no black-box ML libraries.

## Approach

**v2: Dixon–Coles + shrinkage + time decay, with an Elo challenger.**
Shrinkage (Gaussian prior, σ=0.5, on team attack/defense strengths) stops tiny
samples from producing extreme ratings; exponential time decay (180-day
half-life, as in the original Dixon–Coles paper) weights recent games more.
Both hyper-parameters were tuned by walk-forward log loss on the 2024-25 and
2025-26 seasons (760 games) — never on the evaluation weeks. A separate Elo
model (k=32, home bonus 50, margin-of-victory multiplier, tuned the same way)
maps rating differences to win/draw/loss via ordered logit, as a challenger.

**Principle:** bookmaker odds are used only as a benchmark, never as a model input.

## Project layout

```
src/
  load_data.py    # loads football-data.co.uk CSVs into one chronological table
  dixon_coles.py  # the model: Poisson log-PMF, tau() correction, MLE/MAP fit
  elo.py          # challenger: Elo ratings + ordered-logit outcome mapping
  predict.py      # scoreline matrix -> win/draw/loss probabilities
  backtest.py     # v1 walk-forward validation harness (frozen)
  tune.py         # hyper-parameter tuning on 2024-25/2025-26 (never on eval weeks)
  backtest_v2.py  # final evaluation: v1 vs v2 vs Elo vs baselines
data/raw/         # match results, 2021-22 through 2026-27 (football-data.co.uk)
outputs/          # backtest predictions, metrics, tuned params (generated)
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
python backtest.py    # v1 walk-forward validation (frozen baseline)
python tune.py        # tune v2 hyper-parameters on 2024-25/2025-26
python backtest_v2.py # final evaluation: v1 vs v2 vs Elo vs baselines
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

## v2 results

Same 50-game walk-forward, adding v2 Dixon–Coles (σ=0.5 shrinkage, 180-day
half-life) and the Elo challenger (k=32, home bonus 50, margin-of-victory on):

|            | Accuracy | Log loss | Brier |
|------------|----------|----------|-------|
| v1 (frozen)| 0.40     | 1.151    | 0.640 |
| **v2 DC**  | **0.46** | **1.037**| **0.620** |
| Elo        | 0.46     | 1.050    | 0.632 |
| Bookmakers | 0.44     | 1.052    | 0.634 |
| Always-home| 0.36     | —        | —     |

Per-week log loss (v1 / v2 / Elo / book): week 1: 0.975/0.948/1.036/0.986 ·
week 2: 0.878/0.927/0.928/0.976 · week 3: 1.058/1.042/1.048/1.047 ·
week 4: 1.156/1.125/1.173/1.173 · week 5: 1.689/1.144/1.066/1.080.

What changed: the Coventry failure is gone — v2 gave Coventry 14.5% at Forest
instead of ~0.4%, and week 5 log loss fell from 1.689 to 1.144. Draw
calibration improved (0.275 predicted vs 0.320 actual, up from 0.237), though
draws remain the hardest outcome for every model (Elo: 0.245). v2 beats the
bookmakers on all three metrics; Elo is a close second and was the most robust
in chaotic week 5.

**Verdict: carry forward v2 Dixon–Coles** (shrinkage + 180-day decay). Keep Elo
as the standing challenger — with only 50 evaluation games the gap is small,
and Elo handled the upset-heavy week best.

## Tuning detail (held-out seasons 2024-25/2025-26, 760 games)

| prior σ | 0.15 | 0.25 | 0.35 | 0.40 | 0.50 | 0.60 | 0.75 | none |
|---------|------|------|------|------|------|------|------|------|
| log loss|1.033|1.027|1.026|1.025|1.025|1.025|1.026|1.037|

σ=0.5 chosen (flat optimum 0.35–0.75; any shrinkage clearly beats none).
Half-life at σ=0.5: 90d → 1.0150, **180d → 1.0079**, 365d → 1.0100,
730d → 1.0155, none → 1.0249. Elo grid winner: k=32, home bonus 50,
margin-of-victory on (1.0093; next best 1.0099).

## Roadmap (v3 ideas)

- [x] **Shrinkage** — done (σ=0.5 Gaussian prior on team strengths).
- [x] **Time-decay weighting** — done (180-day half-life).
- [x] **Elo challenger** — done; kept as standing challenger.
- [ ] **Ensemble** — average v2 DC and Elo probabilities; Elo's week-5
      robustness suggests the blend may beat either alone.
- [ ] **Draw calibration** — every model underpredicts draws; investigate a
      stronger low-score correction or post-hoc calibration.
- [ ] **Expected-goals features** — xG for/against from Understat or FBref as
      a team-strength input, less noisy than raw goals.
