"""Hyper-parameter tuning for v2, on HELD-OUT seasons only.

Tuning window: the 2024-25 and 2025-26 seasons (76 matchweeks, 760 games).
The 2026-27 weeks 1-5 used for final evaluation are NEVER touched here --
tuning on your test set is cheating, even if it feels harmless.

Protocol per config: walk forward over every matchweek of the two tuning
seasons; for each week, train on all games strictly before the week's first
game date, predict the week's 10 fixtures, accumulate log loss. The config
with the lowest POOLED log loss wins (log loss, not accuracy: we care about
the quality of the probabilities).

Tuning order (greedy but standard):
  1. Dixon-Coles prior_sigma grid, no time decay.
  2. Dixon-Coles half_life_days grid, with the best sigma from step 1.
  3. Elo k x home_bonus x margin-of-victory grid.

Writes outputs/best_params.json with the winners, which backtest_v2.py
reads for the final evaluation.
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all
from dixon_coles import fit as dc_fit
from predict import predict_matchweek as dc_predict
from elo import fit as elo_fit, predict_matchweek as elo_predict

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
EPS = 1e-15
TUNE_SEASONS = ["2024-25", "2025-26"]
OUT_PATH = Path(__file__).resolve().parents[1] / "outputs"


def build_splits(df, seasons=TUNE_SEASONS):
    """Precompute (train, fixtures, actual) for every tuning week.

    train is every game with date strictly before the week's first game --
    the same no-leakage cutoff the backtest uses.
    """
    splits = []
    for season in seasons:
        current = df[df["season"] == season]
        for week in range(1, int(current["matchweek"].max()) + 1):
            week_games = current[current["matchweek"] == week]
            cutoff = week_games["date"].min()
            train = df[df["date"] < cutoff]
            fixtures = list(zip(week_games["home_team"],
                                week_games["away_team"]))
            actual = week_games["result"].map(OUTCOME_INDEX).to_numpy()
            splits.append((train, fixtures, actual))
    return splits


def pooled_logloss(splits, fit_fn, predict_fn):
    """Walk-forward pooled log loss for one config over all splits."""
    ll = 0.0
    n = 0
    for train, fixtures, actual in splits:
        model = fit_fn(train)
        preds = predict_fn(model, fixtures)
        probs = np.array([[p["p_home"], p["p_draw"], p["p_away"]]
                          for p in preds])
        p = np.clip(probs[np.arange(len(actual)), actual], EPS, 1.0)
        ll += -np.log(p).sum()
        n += len(actual)
    return ll / n


# --- worker functions (module-level so they pickle for the process pool) ---

def _eval_dc(args):
    sigma, half_life, splits = args
    return pooled_logloss(
        splits,
        lambda train: dc_fit(train, prior_sigma=sigma,
                             half_life_days=half_life),
        dc_predict)


def _eval_elo(args):
    k, home_bonus, mov, splits = args
    return pooled_logloss(
        splits,
        lambda train: elo_fit(train, k=k, home_bonus=home_bonus, mov=mov),
        elo_predict)


def tune_grid(worker, configs, label, n_weeks):
    """Evaluate configs in parallel; print a ranked table; return best."""
    print(f"\n--- {label} ({len(configs)} configs, {n_weeks} weeks each) ---")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as pool:
        losses = list(pool.map(worker, configs))
    rows = sorted(zip(configs, losses), key=lambda r: r[1])
    for cfg, loss in rows:
        print(f"  {str(cfg[:-1]):45s} logloss={loss:.4f}")
    print(f"  ({time.time()-t0:.0f}s wall)")
    return rows[0][0][:-1], rows  # best hyper-params (sans splits), ranking


def main():
    df = load_all()
    splits = build_splits(df)
    n_games = sum(len(s[2]) for s in splits)
    print(f"tuning on {len(splits)} matchweeks, {n_games} games "
          f"({', '.join(TUNE_SEASONS)})")

    # 1) shrinkage strength, no decay (+ a quick refinement around the best)
    sigmas = [0.15, 0.25, 0.35, 0.5, 0.75, None]
    (best_sigma, _), _ = tune_grid(
        _eval_dc, [(s, None, splits) for s in sigmas],
        "Dixon-Coles: prior_sigma (no time decay)", len(splits))
    refine = [r for r in (best_sigma * 0.8, best_sigma * 1.2)
              if r not in sigmas and best_sigma is not None]
    if refine:
        (best_sigma, _), _ = tune_grid(
            _eval_dc, [(s, None, splits) for s in [best_sigma] + refine],
            "Dixon-Coles: prior_sigma refinement", len(splits))

    # 2) half-life, with best sigma
    half_lives = [90, 180, 365, 730, None]
    (_, best_hl), _ = tune_grid(
        _eval_dc, [(best_sigma, h, splits) for h in half_lives],
        f"Dixon-Coles: half_life_days (sigma={best_sigma})", len(splits))

    # 3) Elo hyper-parameters
    elo_configs = [(k, hb, mov, splits)
                   for k in (16, 24, 32)
                   for hb in (50, 80)
                   for mov in (False, True)]
    (best_k, best_hb, best_mov), _ = tune_grid(
        _eval_elo, elo_configs, "Elo: k x home_bonus x mov", len(splits))

    best = {"dc": {"prior_sigma": best_sigma, "half_life_days": best_hl},
            "elo": {"k": best_k, "home_bonus": best_hb, "mov": best_mov}}
    OUT_PATH.mkdir(exist_ok=True)
    with open(OUT_PATH / "best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("\nwinners:", json.dumps(best))
    print(f"wrote {OUT_PATH / 'best_params.json'}")


if __name__ == "__main__":
    main()
