"""Walk-forward backtest of the v1 Dixon-Coles model on the 2026-27 season.

Protocol (this is the honest way to evaluate a weekly prediction system):
  for each matchweek N in 1..5 of the 2026-27 season:
      train = every game with date STRICTLY BEFORE matchweek N's first game
              (five full prior seasons + 2026-27 weeks < N)
      fit the model on train, predict the 10 games of week N

No future information leaks: the cutoff is a date, not a matchweek label, so
even rescheduled games can't sneak in.

We compare three probability forecasts per game:
  * model     - the Dixon-Coles v1 fit
  * always_home - P = (1, 0, 0); the "dumb" baseline every model must beat
  * bookmaker - AvgH/AvgD/AvgA converted to probabilities (overround removed);
                BENCHMARK ONLY, never a model input

Metrics (all computed on the predicted probability vectors):
  * log loss  - mean(-log(p of the actual outcome)); a proper scoring rule:
                confident-and-wrong is punished harshly
  * Brier     - mean over games of sum_k (p_k - onehot_k)^2; lower is better
  * accuracy  - fraction of games where the most-likely outcome was correct
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all
from dixon_coles import fit
from predict import predict_matchweek

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
EPS = 1e-15  # floors probabilities so log loss stays finite


def bookmaker_probs(row):
    """Average odds -> implied probabilities with the overround removed."""
    inv = np.array([1.0 / row["avg_h"], 1.0 / row["avg_d"], 1.0 / row["avg_a"]])
    return inv / inv.sum()


def log_loss(probs, actual_idx):
    p = np.clip(probs[np.arange(len(actual_idx)), actual_idx], EPS, 1.0)
    return float(-np.log(p).mean())


def brier_score(probs, actual_idx):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual_idx)), actual_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def accuracy(probs, actual_idx):
    return float((probs.argmax(axis=1) == actual_idx).mean())


def backtest(df, season="2026-27", max_week=5):
    """Run the walk-forward backtest; returns (predictions_df, metrics_df)."""
    current = df[df["season"] == season].copy()
    all_rows, metric_rows = [], []

    for week in range(1, max_week + 1):
        week_games = current[current["matchweek"] == week]
        cutoff = week_games["date"].min()          # strict: train < cutoff
        train = df[df["date"] < cutoff]

        model = fit(train)
        fixtures = list(zip(week_games["home_team"], week_games["away_team"]))
        preds = predict_matchweek(model, fixtures)

        model_probs, book_probs, actual = [], [], []
        for (_, g), p in zip(week_games.iterrows(), preds):
            model_probs.append([p["p_home"], p["p_draw"], p["p_away"]])
            book_probs.append(bookmaker_probs(g))
            actual.append(OUTCOME_INDEX[g["result"]])
            all_rows.append({"season": season, "matchweek": week,
                             "date": g["date"].date(),
                             "home_team": g["home_team"],
                             "away_team": g["away_team"],
                             "actual": g["result"],
                             **{f"model_{k}": v for k, v in
                                zip(["h", "d", "a"],
                                    [p["p_home"], p["p_draw"], p["p_away"]])},
                             **{f"book_{k}": v for k, v in
                                zip(["h", "d", "a"], bookmaker_probs(g))}})
        model_probs = np.array(model_probs)
        book_probs = np.array(book_probs)
        home_probs = np.tile([1.0, 0.0, 0.0], (len(actual), 1))
        actual = np.array(actual)

        metric_rows.append({
            "matchweek": week, "n_games": len(actual),
            "train_games": len(train),
            "model_logloss": log_loss(model_probs, actual),
            "book_logloss": log_loss(book_probs, actual),
            "home_logloss": log_loss(home_probs, actual),
            "model_brier": brier_score(model_probs, actual),
            "book_brier": brier_score(book_probs, actual),
            "home_brier": brier_score(home_probs, actual),
            "model_acc": accuracy(model_probs, actual),
            "book_acc": accuracy(book_probs, actual),
            "home_acc": accuracy(home_probs, actual),
        })
        m = metric_rows[-1]
        print(f"week {week}: trained on {len(train)} games | "
              f"model acc {m['model_acc']:.2f} / book {m['book_acc']:.2f} / "
              f"home {m['home_acc']:.2f} | "
              f"model logloss {m['model_logloss']:.3f} / "
              f"book {m['book_logloss']:.3f}")

    return pd.DataFrame(all_rows), pd.DataFrame(metric_rows)


def calibration_report(pred_df):
    """Pool every predicted outcome-probability into bins; compare vs reality.

    With only 50 games (150 probabilities) this is noisy, but systematic
    miscalibration -- e.g. draws never happening at their predicted rate --
    still shows up.
    """
    recs = []
    for _, r in pred_df.iterrows():
        actual = OUTCOME_INDEX[r["actual"]]
        for k, cls in enumerate(["h", "d", "a"]):
            recs.append({"p": r[f"model_{cls}"], "hit": int(k == actual),
                         "class": cls})
    cal = pd.DataFrame(recs)
    cal["bin"] = pd.cut(cal["p"], bins=np.arange(0, 1.01, 0.1))
    out = (cal.groupby("bin", observed=True)
              .agg(n=("hit", "size"),
                   mean_predicted=("p", "mean"),
                   actual_rate=("hit", "mean"))
              .reset_index())
    # per-class view: is any outcome type systematically off?
    by_class = (cal.groupby("class", observed=True)
                   .agg(n=("hit", "size"),
                        mean_predicted=("p", "mean"),
                        actual_rate=("hit", "mean")))
    return out, by_class


if __name__ == "__main__":
    df = load_all()
    pred_df, metrics_df = backtest(df)

    print("\n=== per-matchweek metrics ===")
    print(metrics_df.to_string(index=False, float_format="%.3f"))

    # Overall = pooled over all 50 games (equivalent to micro-average here).
    probs = pred_df[["model_h", "model_d", "model_a"]].to_numpy()
    book = pred_df[["book_h", "book_d", "book_a"]].to_numpy()
    home = np.tile([1.0, 0.0, 0.0], (len(pred_df), 1))
    actual = pred_df["actual"].map(OUTCOME_INDEX).to_numpy()
    print("\n=== overall (50 games) ===")
    for name, pr in [("model", probs), ("bookmaker", book),
                     ("always_home", home)]:
        print(f"{name:12s} logloss={log_loss(pr, actual):.3f} "
              f"brier={brier_score(pr, actual):.3f} "
              f"acc={accuracy(pr, actual):.3f}")

    print("\n=== calibration (all predicted probabilities, 10% bins) ===")
    bins, by_class = calibration_report(pred_df)
    print(bins.to_string(index=False, float_format="%.3f"))
    print("\n=== calibration by outcome class ===")
    print(by_class.to_string(float_format="%.3f"))

    out_path = Path(__file__).resolve().parents[1] / "outputs"
    out_path.mkdir(exist_ok=True)
    pred_df.to_csv(out_path / "backtest_predictions.csv", index=False)
    metrics_df.to_csv(out_path / "backtest_metrics.csv", index=False)
    print(f"\nwrote {out_path / 'backtest_predictions.csv'}")
