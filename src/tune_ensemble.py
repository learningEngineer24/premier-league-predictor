"""v3 (item 1): tune the v2 Dixon-Coles + Elo ensemble weight.

Blend:  p = w * p_v2 + (1 - w) * p_elo,  w in [0, 1].

Same honest protocol as tune.py: tune ONLY on the 2024-25 and 2025-26
seasons (760 games). The 2026-27 weeks 1-5 evaluation set is never touched.
Both base models use their tuned hyper-parameters from
outputs/best_params.json. Selection metric: pooled walk-forward log loss
(probability quality, not accuracy).

Efficiency: fit both models once per split, cache the probability matrices,
then vectorize the weight grid -- no refitting per weight.

Writes the winning weight to outputs/ensemble_params.json.

Usage: cd src && python tune_ensemble.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all
from dixon_coles import fit as dc_fit
from predict import predict_matchweek as dc_predict
from elo import fit as elo_fit, predict_matchweek as elo_predict
from tune import build_splits, TUNE_SEASONS, EPS

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR.parent / "outputs"


def prob_matrix(predict_fn, model, fixtures):
    preds = predict_fn(model, fixtures)
    return np.array([[p["p_home"], p["p_draw"], p["p_away"]] for p in preds])


def main():
    with open(OUT_PATH / "best_params.json") as f:
        best = json.load(f)
    dc_p, elo_p = best["dc"], best["elo"]

    df = load_all()
    splits = build_splits(df, seasons=TUNE_SEASONS)
    n_games = sum(len(s[2]) for s in splits)
    print(f"tuning ensemble on {len(splits)} matchweeks, {n_games} games "
          f"({', '.join(TUNE_SEASONS)})")

    # Fit both base models once per split; cache probability matrices.
    t0 = time.time()
    all_v2, all_elo, all_actual = [], [], []
    for i, (train, fixtures, actual) in enumerate(splits):
        v2 = dc_fit(train, prior_sigma=dc_p["prior_sigma"],
                    half_life_days=dc_p["half_life_days"])
        elo = elo_fit(train, k=elo_p["k"], home_bonus=elo_p["home_bonus"],
                      mov=elo_p["mov"])
        all_v2.append(prob_matrix(dc_predict, v2, fixtures))
        all_elo.append(prob_matrix(elo_predict, elo, fixtures))
        all_actual.append(actual)
        if (i + 1) % 20 == 0:
            print(f"  fit {i + 1}/{len(splits)} weeks ({time.time()-t0:.0f}s)")
    P_V2 = np.vstack(all_v2)
    P_ELO = np.vstack(all_elo)
    Y = np.concatenate(all_actual)
    print(f"  fits done ({time.time()-t0:.0f}s wall)")

    # Grid-search the blend weight (vectorized over all games at once).
    print("\nweight (v2 share) -> pooled log loss:")
    results = []
    for w in np.arange(0, 1.01, 0.05):
        P = w * P_V2 + (1 - w) * P_ELO
        p = np.clip(P[np.arange(len(Y)), Y], EPS, 1.0)
        ll = float(-np.log(p).mean())
        results.append((w, ll))
        print(f"  w={w:.2f}  logloss={ll:.4f}")

    results.sort(key=lambda r: r[1])
    best_w, best_ll = results[0]
    # Report the pure-model endpoints for context.
    w0_ll = next(ll for w, ll in results if w == 0.0)
    w1_ll = next(ll for w, ll in results if w == 1.0)
    print(f"\nbest w={best_w:.2f} (logloss={best_ll:.4f})")
    print(f"pure Elo (w=0): {w0_ll:.4f}   pure v2 (w=1): {w1_ll:.4f}")

    # Paired bootstrap CI for (ensemble - v2) per-game log-loss difference.
    # If the CI covers 0, the gain is noise -- report it honestly.
    P_best = best_w * P_V2 + (1 - best_w) * P_ELO
    ll_best = -np.log(np.clip(P_best[np.arange(len(Y)), Y], EPS, 1.0))
    ll_v2 = -np.log(np.clip(P_V2[np.arange(len(Y)), Y], EPS, 1.0))
    diff = ll_best - ll_v2
    rng = np.random.default_rng(20261008)
    boots = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                      for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"paired diff (ensemble - v2): mean={diff.mean():+.4f}, "
          f"bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")

    with open(OUT_PATH / "ensemble_params.json", "w") as f:
        json.dump({"w_v2": round(float(best_w), 2),
                   "w_elo": round(float(1 - best_w), 2),
                   "tuned_logloss": round(best_ll, 4),
                   "paired_diff_mean": round(float(diff.mean()), 4),
                   "paired_diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
                   "tuned_on": list(TUNE_SEASONS)}, f, indent=2)
    print("wrote outputs/ensemble_params.json")


if __name__ == "__main__":
    main()
