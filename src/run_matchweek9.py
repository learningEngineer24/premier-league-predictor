"""Weekly runner: generate Matchweek 9 (Oct 31-Nov 2, 2026) predictions.

SPEC CHANGE (decided by Nebo Oct 29, 2026, in the model-rework chat):
  * v2 Dixon-Coles now fits with a Gaussian prior on the home-advantage
    term, home_prior_sigma=0.15, on top of the usual strength prior
    (prior_sigma=0.5) and 180-day half-life.
  * Evidence: 2024-25 tune-window walk-forward (340 games, MW5-38) --
    base log loss 0.9946 vs prior015 0.9910 (diff -0.0036, paired
    bootstrap 95% CI [-0.0072, -0.0001], the only home-shrink variant
    whose CI excluded zero). Post-fit scalings (x0.75 / x0.50) were
    rejected: bigger point gains but CIs covering zero, and no refit.
  * Protocol: selection made on the 2024-25 tune window only; the
    2025-26 locked set was not touched. Elo is unchanged (its home
    prices were already market-aligned: +0.2pp vs v2's +1.0pp over
    MW5-8), as is the 0.55/0.45 blend.
  * This script also prints the old-spec (no home prior) ensemble for
    comparison; the saved CSV is the NEW spec, which is the live one.

Protocol (same honesty rules as the walk-forward backtest):
  * data/raw/E0_2627.csv refreshed with all 10 MW8 results first
    (done Oct 29 by the bookmaker backfill; 80 games played).
  * Train on all games with date strictly before the week's first fixture.
  * Save versioned output to outputs/matchweek9_predictions.csv.

Usage: cd src && python run_matchweek9.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all
from dixon_coles import fit as dc_fit
from predict import predict_matchweek as dc_predict
from elo import fit as elo_fit, predict_matchweek as elo_predict

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR.parent / "outputs"

HOME_PRIOR_SIGMA = 0.15  # adopted Oct 29, 2026 (see docstring)

# Matchweek 9 fixtures, Oct 31-Nov 2 2026 (verified Oct 29 across
# sportradar, NBC Sports, DAZN, bets.co.za; all 20 teams exactly once).
FIXTURES = [
    ("2026-10-31", "Chelsea", "Man United"),
    ("2026-10-31", "Bournemouth", "Leeds"),
    ("2026-10-31", "Brentford", "Nott'm Forest"),
    ("2026-10-31", "Coventry", "Sunderland"),
    ("2026-10-31", "Hull", "Ipswich"),
    ("2026-10-31", "Man City", "Brighton"),
    ("2026-10-31", "Tottenham", "Crystal Palace"),
    ("2026-10-31", "Aston Villa", "Fulham"),
    ("2026-11-01", "Liverpool", "Arsenal"),
    ("2026-11-02", "Newcastle", "Everton"),
]
OUTCOME = {0: "Home win", 1: "Draw", 2: "Away win"}


def blend(dc, elo, w_v2, w_elo):
    eh = w_v2 * dc["p_home"] + w_elo * elo["p_home"]
    ed = w_v2 * dc["p_draw"] + w_elo * elo["p_draw"]
    ea = w_v2 * dc["p_away"] + w_elo * elo["p_away"]
    s = eh + ed + ea
    return eh / s, ed / s, ea / s


def main():
    with open(OUT_PATH / "best_params.json") as f:
        best = json.load(f)
    with open(OUT_PATH / "ensemble_params.json") as f:
        ens = json.load(f)
    dc_p, elo_p = best["dc"], best["elo"]
    w_v2, w_elo = ens["w_v2"], ens["w_elo"]
    assert abs(w_v2 + w_elo - 1.0) < 1e-9

    df = load_all()
    cutoff = pd.Timestamp("2026-10-31")
    train = df[df["date"] < cutoff]
    n_played = len(df[df["season"] == "2026-27"])
    print(f"training games: {len(train)} (2026-27 played so far: {n_played})")
    assert n_played == 80, "unexpected number of played games in 2026-27"

    dc_new = dc_fit(train, prior_sigma=dc_p["prior_sigma"],
                    half_life_days=dc_p["half_life_days"],
                    home_prior_sigma=HOME_PRIOR_SIGMA, verbose=True)
    dc_old = dc_fit(train, prior_sigma=dc_p["prior_sigma"],
                    half_life_days=dc_p["half_life_days"], verbose=True)
    print(f"home_adv: old spec {dc_old.home_adv:.3f} -> "
          f"new spec {dc_new.home_adv:.3f}")
    elo_model = elo_fit(train, k=elo_p["k"], home_bonus=elo_p["home_bonus"],
                        mov=elo_p["mov"])

    fixtures = [(h, a) for _, h, a in FIXTURES]
    new_preds = dc_predict(dc_new, fixtures)
    old_preds = dc_predict(dc_old, fixtures)
    elo_preds = elo_predict(elo_model, fixtures)

    rows = []
    for (d, h, a), dc, old, elo in zip(FIXTURES, new_preds, old_preds,
                                       elo_preds):
        eh, ed, ea = blend(dc, elo, w_v2, w_elo)
        oh, od, oa = blend(old, elo, w_v2, w_elo)
        rows.append({
            "matchweek": 9, "date": d,
            "home_team": h, "away_team": a,
            "v2_h": dc["p_home"], "v2_d": dc["p_draw"], "v2_a": dc["p_away"],
            "elo_h": elo["p_home"], "elo_d": elo["p_draw"],
            "elo_a": elo["p_away"],
            "ens_h": eh, "ens_d": ed, "ens_a": ea,
            "old_ens_h": oh, "old_ens_d": od, "old_ens_a": oa,
        })
    out = pd.DataFrame(rows)
    path = OUT_PATH / "matchweek9_predictions.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")

    print("\n=== Matchweek 9 predictions (NEW spec: v2 home prior 0.15, "
          "0.55 v2 + 0.45 Elo) ===")
    for r in rows:
        p = [r["ens_h"], r["ens_d"], r["ens_a"]]
        top = max(range(3), key=lambda i: p[i])
        o = [r["old_ens_h"], r["old_ens_d"], r["old_ens_a"]]
        otop = max(range(3), key=lambda i: o[i])
        flip = (f"  [old spec: {OUTCOME[otop]} "
                f"H {o[0]:.0%} D {o[1]:.0%} A {o[2]:.0%}]") if top != otop \
            else f"  [old home {o[0]:.0%} -> new {p[0]:.0%}]"
        print(f"{r['date'][5:]} {r['home_team']:15s} v {r['away_team']:13s} "
              f"H {p[0]:.0%}  D {p[1]:.0%}  A {p[2]:.0%}  "
              f"-> {OUTCOME[top]}{flip}")


if __name__ == "__main__":
    main()
