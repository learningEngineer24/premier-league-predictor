"""Final walk-forward evaluation: v1 vs v2 Dixon-Coles vs Elo, 2026-27 weeks 1-5.

The same honest protocol as backtest.py: for each matchweek N, train on all
games with date strictly before week N's first game, predict the week's 10
fixtures. Five probability forecasts per game:

  * v1_dc   - frozen v1 Dixon-Coles: fit(train) with no shrinkage, no decay
  * v2_dc   - Dixon-Coles with tuned shrinkage + time decay
              (hyper-parameters from outputs/best_params.json -- tuned on
              2024-25/2025-26, never on these weeks)
  * elo     - Elo ratings + ordered logit (hyper-parameters likewise tuned)
  * book    - bookmaker-implied probabilities; BENCHMARK ONLY, never a feature
  * home    - always predict home win (1, 0, 0); the dumb baseline

Metrics per model: log loss, Brier score, accuracy -- overall (pooled over
the 50 games) and per matchweek. Plus calibration by outcome class, with
special attention to draws (v1 underpredicted them: 0.237 vs 0.320 actual).

Usage: cd src && python backtest_v2.py   (run tune.py first)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all
from dixon_coles import fit as dc_fit
from predict import predict_matchweek as dc_predict
from elo import fit as elo_fit, predict_matchweek as elo_predict
from backtest import (bookmaker_probs, log_loss, brier_score, accuracy,
                      OUTCOME_INDEX, calibration_report)

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR.parent / "outputs"   # pl-predictor/outputs


def evaluate(df, season="2026-27", max_week=5):
    """Run the multi-model walk-forward backtest.

    Returns (predictions_df, metrics_df). predictions_df has one row per
    game with model_h/d/a, v2_h/d/a, elo_h/d/a, book_h/d/a columns.
    """
    with open(OUT_PATH / "best_params.json") as f:
        best = json.load(f)
    dc_p = best["dc"]
    elo_p = best["elo"]
    print(f"v2_dc params: {dc_p} | elo params: {elo_p}")

    models = {
        "v1_dc": (lambda tr: dc_fit(tr), dc_predict),
        "v2_dc": (lambda tr: dc_fit(tr,
                                   prior_sigma=dc_p["prior_sigma"],
                                   half_life_days=dc_p["half_life_days"]),
                  dc_predict),
        "elo": (lambda tr: elo_fit(tr, k=elo_p["k"],
                                  home_bonus=elo_p["home_bonus"],
                                  mov=elo_p["mov"]),
                elo_predict),
    }

    current = df[df["season"] == season].copy()
    all_rows, metric_rows = [], []

    for week in range(1, max_week + 1):
        week_games = current[current["matchweek"] == week]
        cutoff = week_games["date"].min()          # strict: train < cutoff
        train = df[df["date"] < cutoff]
        fixtures = list(zip(week_games["home_team"], week_games["away_team"]))
        actual = week_games["result"].map(OUTCOME_INDEX).to_numpy()

        # fit every model once per week, predict the week's fixtures
        probs = {}
        for name, (fit_fn, predict_fn) in models.items():
            preds = predict_fn(fit_fn(train), fixtures)
            probs[name] = np.array([[p["p_home"], p["p_draw"], p["p_away"]]
                                    for p in preds])
        probs["book"] = np.array([bookmaker_probs(g)
                                  for _, g in week_games.iterrows()])
        probs["home"] = np.tile([1.0, 0.0, 0.0], (len(actual), 1))

        for (_, g), i in zip(week_games.iterrows(),
                             range(len(week_games))):
            row = {"season": season, "matchweek": week,
                   "date": g["date"].date(),
                   "home_team": g["home_team"], "away_team": g["away_team"],
                   "actual": g["result"]}
            for name in ("v1_dc", "v2_dc", "elo", "book"):
                for k, cls in enumerate(["h", "d", "a"]):
                    row[f"{name}_{cls}"] = probs[name][i, k]
            all_rows.append(row)

        mrow = {"matchweek": week, "n_games": len(actual),
                "train_games": len(train)}
        for name in ("v1_dc", "v2_dc", "elo", "book", "home"):
            mrow[f"{name}_logloss"] = log_loss(probs[name], actual)
            mrow[f"{name}_brier"] = brier_score(probs[name], actual)
            mrow[f"{name}_acc"] = accuracy(probs[name], actual)
        metric_rows.append(mrow)
        print(f"week {week}: " + " | ".join(
            f"{name} ll={mrow[f'{name}_logloss']:.3f}"
            for name in ("v1_dc", "v2_dc", "elo", "book")))

    return pd.DataFrame(all_rows), pd.DataFrame(metric_rows)


def overall_table(pred_df, models=("v1_dc", "v2_dc", "elo", "book", "home")):
    """Pooled metrics over all games; returns a DataFrame."""
    actual = pred_df["actual"].map(OUTCOME_INDEX).to_numpy()
    rows = []
    for name in models:
        if name == "home":
            pr = np.tile([1.0, 0.0, 0.0], (len(pred_df), 1))
        else:
            pr = pred_df[[f"{name}_h", f"{name}_d",
                          f"{name}_a"]].to_numpy()
        rows.append({"model": name,
                     "logloss": log_loss(pr, actual),
                     "brier": brier_score(pr, actual),
                     "acc": accuracy(pr, actual)})
    return pd.DataFrame(rows)


def calibration_by_class(pred_df, model):
    """Mean predicted vs actual rate per outcome class, for one model."""
    actual = pred_df["actual"].map(OUTCOME_INDEX).to_numpy()
    recs = []
    for k, cls in enumerate(["h", "d", "a"]):
        p = pred_df[f"{model}_{cls}"].to_numpy()
        recs.append({"class": {"h": "home win", "d": "draw",
                               "a": "away win"}[cls],
                     "n": len(p),
                     "mean_predicted": p.mean(),
                     "actual_rate": (actual == k).mean()})
    return pd.DataFrame(recs)


if __name__ == "__main__":
    df = load_all()
    pred_df, metrics_df = evaluate(df)

    print("\n=== per-matchweek log loss ===")
    cols = ["matchweek"] + [f"{m}_logloss"
                            for m in ("v1_dc", "v2_dc", "elo", "book")]
    print(metrics_df[cols].to_string(index=False, float_format="%.3f"))

    print("\n=== overall (50 games) ===")
    print(overall_table(pred_df).to_string(index=False, float_format="%.3f"))

    print("\n=== calibration by outcome class ===")
    for model in ("v1_dc", "v2_dc", "elo"):
        print(f"\n-- {model} --")
        print(calibration_by_class(pred_df, model)
              .to_string(index=False, float_format="%.3f"))

    OUT_PATH.mkdir(exist_ok=True)
    pred_df.to_csv(OUT_PATH / "backtest_v2_predictions.csv", index=False)
    metrics_df.to_csv(OUT_PATH / "backtest_v2_metrics.csv", index=False)
    overall_table(pred_df).to_csv(OUT_PATH / "backtest_v2_overall.csv",
                                  index=False)
    print(f"\nwrote {OUT_PATH / 'backtest_v2_predictions.csv'}")
