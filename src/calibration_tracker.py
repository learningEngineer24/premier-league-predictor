"""Calibration tracker with naive baselines (built Oct 29, 2026).

Standing measurement layer for the weekly review cycle. For every
reviewed matchweek (MW5 onward) it scores, on the same games:

  ensemble    0.55 * v2 + 0.45 * Elo (the live pick; MW5-6 restated
              with the ensemble retrospectively -- those weeks ran
              v2-only live, same convention as the home-bias analysis)
  v2          Dixon-Coles v2 alone
  elo         Elo challenger alone
  market      bookmaker closing Avg odds, overround normalized
              (same method as score_bookmaker.py)
  base_rate   league outcome frequencies from the two completed prior
              seasons (2024-25 + 2025-26, 760 games), predicted for
              every game without knowing the teams
  uniform     1/3, 1/3, 1/3 every game (log loss is a constant ln 3)
  always_home accuracy only -- a degenerate 100%-home predictor has no
              finite log loss, so LL/Brier are reported as n/a

Plus calibration for the ensemble (and market, in-the-large):
  * calibration-in-the-large: mean predicted probability per outcome
    vs the actual frequency
  * top-pick reliability: predicted probability of the picked outcome,
    binned, vs how often the pick was right (bins with n < 10 are
    flagged "too thin to read")
  * home-probability reliability: ensemble home prob, binned, vs the
    actual home-win rate (the home-bias rework made this the standing
    calibration question)

Scores are pooled per-game for cumulative rows (not averages of weekly
averages). Unplayed weeks (predictions exist, no results yet) are
skipped with a note. Nothing here tunes or selects anything; it is
measurement only, and it never touches the 2025-26 locked exam set as
an evaluation target (base rates use final results of completed
seasons as league frequencies, the same public fact the Oct 19
baseline computation used).

Standing weekly step: after score_bookmaker.py on Tuesdays, run
  cd src && python3 calibration_tracker.py
which rewrites outputs/calibration_tracker.{csv,json} and prints the
tables for the review.

Usage: cd src && python3 calibration_tracker.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR.parent / "outputs"
RAW = SRC_DIR.parent / "data" / "raw" / "E0_2627.csv"
RES = {"H": 0, "D": 1, "A": 2}
W_V2, W_ELO = 0.55, 0.45
FIRST_WEEK = 5


def blend(v2, elo):
    p = W_V2 * np.asarray(v2) + W_ELO * np.asarray(elo)
    return p / p.sum()


def base_rates():
    """Outcome frequencies over the two completed prior seasons."""
    df = load_all()
    prior = df[df["season"].isin(["2024-25", "2025-26"])]
    counts = prior["result"].value_counts()
    n = int(counts.sum())
    probs = np.array([counts["H"], counts["D"], counts["A"]], dtype=float)
    return probs / probs.sum(), n


def load_week(week, fd):
    """Return per-game dicts for one reviewed week, or None if unplayed.

    Each game: home, away, actual (0/1/2), probs per predictor.
    Market/base-rate/uniform/always-home are attached by the caller.
    """
    games = []
    if week == 5:
        b = pd.read_csv(OUT_PATH / "backtest_v2_predictions.csv")
        b = b[(b["season"] == "2026-27") & (b["matchweek"] == 5)]
        for _, r in b.iterrows():
            games.append({
                "home": r["home_team"], "away": r["away_team"],
                "v2": [r["v2_dc_h"], r["v2_dc_d"], r["v2_dc_a"]],
                "elo": [r["elo_h"], r["elo_d"], r["elo_a"]],
            })
    else:
        path = OUT_PATH / f"matchweek{week}_predictions.csv"
        if not path.exists():
            return None
        pred = pd.read_csv(path)
        for _, r in pred.iterrows():
            g = {"home": r["home_team"], "away": r["away_team"],
                 "v2": [r["v2_h"], r["v2_d"], r["v2_a"]],
                 "elo": [r["elo_h"], r["elo_d"], r["elo_a"]]}
            if "ens_h" in pred.columns:
                g["ens_saved"] = [r["ens_h"], r["ens_d"], r["ens_a"]]
            games.append(g)
    for g in games:
        m = fd[(fd["HomeTeam"] == g["home"]) & (fd["AwayTeam"] == g["away"])]
        if m.empty or pd.isna(m.iloc[0]["FTR"]):
            return None  # week not fully played/posted yet
        row = m.iloc[0]
        g["actual"] = RES[row["FTR"]]
        inv = np.array([1 / row["AvgH"], 1 / row["AvgD"], 1 / row["AvgA"]])
        g["market"] = inv / inv.sum()
        g["ensemble"] = blend(g["v2"], g["elo"])
    return games


def score(games, key):
    """(acc, logloss, brier) for a probability predictor over games."""
    ll, br, hit = [], [], 0
    for g in games:
        p = np.asarray(g[key], dtype=float)
        a = g["actual"]
        ll.append(-np.log(max(p[a], 1e-15)))
        oh = np.zeros(3)
        oh[a] = 1.0
        br.append(float(np.sum((p - oh) ** 2)))
        hit += int(p.argmax() == a)
    return hit, float(np.mean(ll)), float(np.mean(br))


PREDICTORS = ["ensemble", "v2", "elo", "market", "base_rate", "uniform"]


def main():
    fd = pd.read_csv(RAW)
    br_probs, br_n = base_rates()
    print(f"base rates from {br_n} prior-season games: "
          f"H {br_probs[0]:.3f} D {br_probs[1]:.3f} A {br_probs[2]:.3f}")

    rows, all_games = [], []
    week = FIRST_WEEK
    while True:
        games = load_week(week, fd)
        if games is None:
            if week == FIRST_WEEK:
                week += 1
                continue
            break
        for g in games:
            g["base_rate"] = br_probs
            g["uniform"] = np.array([1 / 3, 1 / 3, 1 / 3])
        all_games.extend(games)
        for name in PREDICTORS:
            hit, ll, br = score(games, name)
            rows.append({"scope": f"week{week}", "predictor": name,
                         "n": len(games), "acc": hit, "logloss": ll,
                         "brier": br})
        ah = sum(1 for g in games if g["actual"] == 0)
        rows.append({"scope": f"week{week}", "predictor": "always_home",
                     "n": len(games), "acc": ah, "logloss": None,
                     "brier": None})
        week += 1

    for name in PREDICTORS:
        hit, ll, br = score(all_games, name)
        rows.append({"scope": "cumulative", "predictor": name,
                     "n": len(all_games), "acc": hit, "logloss": ll,
                     "brier": br})
    ah = sum(1 for g in all_games if g["actual"] == 0)
    rows.append({"scope": "cumulative", "predictor": "always_home",
                 "n": len(all_games), "acc": ah, "logloss": None,
                 "brier": None})
    table = pd.DataFrame(rows)

    # ---- calibration ----
    n = len(all_games)
    actual_rate = np.array([np.mean([g["actual"] == k for g in all_games])
                            for k in range(3)])
    in_large = {}
    for name in ("ensemble", "market"):
        mean_pred = np.mean([g[name] for g in all_games], axis=0)
        in_large[name] = {
            k: {"mean_pred": float(mean_pred[i]),
                "actual_rate": float(actual_rate[i])}
            for i, k in enumerate(["H", "D", "A"])}

    def bins(key_fn, bins_):
        out = []
        for lo, hi in bins_:
            sel = [g for g in all_games if lo <= key_fn(g)[0] < hi]
            if not sel:
                out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": 0})
                continue
            out.append({
                "bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel),
                "mean_pred": float(np.mean([key_fn(g)[0] for g in sel])),
                "hit_rate": float(np.mean([key_fn(g)[1] for g in sel])),
                "thin": len(sel) < 10,
            })
        return out

    top_bins = bins(
        lambda g: (max(g["ensemble"]),
                   int(np.argmax(g["ensemble"]) == g["actual"])),
        [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)])
    home_bins = bins(
        lambda g: (g["ensemble"][0], int(g["actual"] == 0)),
        [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.85)])

    out = {"n_games": n, "weeks": f"{FIRST_WEEK}-{week - 1}",
           "base_rates": {"H": float(br_probs[0]), "D": float(br_probs[1]),
                          "A": float(br_probs[2]), "n_prior_games": br_n},
           "scoreboard": rows,
           "calibration_in_the_large": in_large,
           "top_pick_reliability_ensemble": top_bins,
           "home_prob_reliability_ensemble": home_bins}
    table.to_csv(OUT_PATH / "calibration_tracker.csv", index=False)
    with open(OUT_PATH / "calibration_tracker.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- print report ----
    cum = table[table["scope"] == "cumulative"].set_index("predictor")
    print(f"\n=== Cumulative scoreboard, MW{FIRST_WEEK}-{week - 1} "
          f"({n} games) ===")
    print(f"{'predictor':12s} {'acc':>7s} {'logloss':>8s} {'brier':>7s}")
    for name in PREDICTORS + ["always_home"]:
        r = cum.loc[name]
        ll = f"{r['logloss']:.3f}" if pd.notna(r["logloss"]) else "n/a"
        brs = f"{r['brier']:.3f}" if pd.notna(r["brier"]) else "n/a"
        print(f"{name:12s} {int(r['acc']):>3d}/{n:<3d} {ll:>8s} {brs:>7s}")
    print("\n=== Calibration-in-the-large (mean predicted vs actual) ===")
    for name, d in in_large.items():
        print(name + ": " + "  ".join(
            f"{k} {v['mean_pred']:.1%} vs {v['actual_rate']:.1%}"
            for k, v in d.items()))
    print("\n=== Ensemble top-pick reliability ===")
    for b in top_bins:
        if b["n"] == 0:
            print(f"{b['bin']}: n=0")
        else:
            flag = "  (too thin to read)" if b["thin"] else ""
            print(f"{b['bin']}: n={b['n']}  mean conf {b['mean_pred']:.1%}  "
                  f"hit {b['hit_rate']:.1%}{flag}")
    print("\n=== Ensemble home-prob reliability ===")
    for b in home_bins:
        if b["n"] == 0:
            print(f"{b['bin']}: n=0")
        else:
            flag = "  (too thin to read)" if b["thin"] else ""
            print(f"{b['bin']}: n={b['n']}  mean pred {b['mean_pred']:.1%}  "
                  f"home won {b['hit_rate']:.1%}{flag}")
    print(f"\nwrote {OUT_PATH / 'calibration_tracker.csv'} and "
          f"{OUT_PATH / 'calibration_tracker.json'}")


if __name__ == "__main__":
    main()
