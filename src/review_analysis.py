"""Review-driven analysis: uncertainty quantification, ablation, draw diagnosis.

Written in response to the Sep 28, 2026 external review. Three analyses:

1. BOOTSTRAP: 95% CIs (10k resamples) on PAIRED per-game metric differences
   between models on the 50-game 2026-27 weeks 1-5 evaluation set.
   Reports mean diff, SE, and 95% CI for log loss, Brier, accuracy.

2. ABLATION: walk-forward (same honest protocol as backtest_v2.py) over
   2026-27 weeks 1-5 with four Dixon-Coles configs:
     neither  = v1 (no shrinkage, no decay)
     shrink   = sigma=0.5 only
     decay    = 180-day half-life only
     both     = v2
   Attributes the v1->v2 improvement to its components.

3. DRAW DIAGNOSIS: walk-forward over the 760-game TUNING window
   (2024-25/2025-26, never the eval weeks) with v2 params: mean predicted
   draw probability vs actual draw rate. Decides whether the "draw problem"
   seen in the 50-game eval window is structural or a small-sample artifact.

Usage: cd src && python review_analysis.py
Writes outputs/review_analysis.json and prints a summary.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all
from dixon_coles import fit as dc_fit
from predict import predict_matchweek as dc_predict

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR.parent / "outputs"
EPS = 1e-15
N_BOOT = 10_000
SEED = 7

with open(OUT_PATH / "best_params.json") as f:
    BEST = json.load(f)
DC_P = BEST["dc"]  # {"prior_sigma": 0.5, "half_life_days": 180}


# ---------------------------------------------------------------- bootstrap
def per_game_logloss(probs, actual):
    p = np.clip(probs[np.arange(len(actual)), actual], EPS, 1.0)
    return -np.log(p)


def per_game_brier(probs, actual):
    n = len(actual)
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), actual] = 1.0
    return np.sum((probs - onehot) ** 2, axis=1)


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=SEED):
    """CI on mean(a - b) via bootstrap. Returns dict."""
    rng = np.random.default_rng(seed)
    d = a - b
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    return {
        "mean": float(d.mean()),
        "se": float(d.std(ddof=1) / np.sqrt(len(d))),
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
    }


def run_bootstrap():
    df = pd.read_csv(OUT_PATH / "backtest_v2_predictions.csv")
    actual = df["actual"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    models = {
        "v1": df[["v1_dc_h", "v1_dc_d", "v1_dc_a"]].to_numpy(),
        "v2": df[["v2_dc_h", "v2_dc_d", "v2_dc_a"]].to_numpy(),
        "elo": df[["elo_h", "elo_d", "elo_a"]].to_numpy(),
        "book": df[["book_h", "book_d", "book_a"]].to_numpy(),
    }
    ll = {m: per_game_logloss(p, actual) for m, p in models.items()}
    br = {m: per_game_brier(p, actual) for m, p in models.items()}
    acc = {m: (p.argmax(axis=1) == actual).astype(float)
           for m, p in models.items()}

    pairs = [("v2", "book"), ("v2", "elo"), ("v2", "v1"), ("elo", "book")]
    out = {}
    for x, y in pairs:
        out[f"{x}_vs_{y}"] = {
            "logloss": paired_bootstrap(ll[x], ll[y]),
            "brier": paired_bootstrap(br[x], br[y]),
            "accuracy": paired_bootstrap(acc[x], acc[y]),
        }
    # single-game attribution: how much of v1->v2 logloss gain is one game?
    d = ll["v1"] - ll["v2"]          # positive = v2 better
    top = int(np.argmax(d))
    out["v1_to_v2_single_game_share"] = {
        "total_gain_nats": float(d.sum()),
        "top_game_gain_nats": float(d[top]),
        "top_game_share": float(d[top] / d.sum()),
        "top_game_row": int(top),
        "top_game_fixture": f"{df.iloc[top]['home_team']} vs "
                            f"{df.iloc[top]['away_team']} "
                            f"(week {int(df.iloc[top]['matchweek'])})",
    }
    return out


# ------------------------------------------------------------------ ablation
def _fit_predict(args):
    """Worker: fit one DC config on train, predict fixtures. Returns h/d/a."""
    cfg_name, train, fixtures = args
    if cfg_name == "neither":
        m = dc_fit(train)
    elif cfg_name == "shrink":
        m = dc_fit(train, prior_sigma=DC_P["prior_sigma"])
    elif cfg_name == "decay":
        m = dc_fit(train, half_life_days=DC_P["half_life_days"])
    elif cfg_name == "both":
        m = dc_fit(train, prior_sigma=DC_P["prior_sigma"],
                   half_life_days=DC_P["half_life_days"])
    preds = dc_predict(m, fixtures)
    return np.array([[p["p_home"], p["p_draw"], p["p_away"]] for p in preds])


def run_ablation():
    df = load_all()
    cur = df[df["season"] == "2026-27"]
    configs = ["neither", "shrink", "decay", "both"]
    jobs, meta = [], []
    for week in range(1, 6):
        wg = cur[cur["matchweek"] == week]
        cutoff = wg["date"].min()
        train = df[df["date"] < cutoff]
        fixtures = list(zip(wg["home_team"], wg["away_team"]))
        actual = wg["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        for cfg in configs:
            jobs.append((cfg, train, fixtures))
            meta.append((week, cfg, actual))
    with ProcessPoolExecutor(max_workers=4) as ex:
        prob_list = list(ex.map(_fit_predict, jobs))
    ll_by_cfg, n_by_cfg = {}, {}
    for (week, cfg, actual), probs in zip(meta, prob_list):
        ll = per_game_logloss(probs, actual).sum()
        ll_by_cfg[cfg] = ll_by_cfg.get(cfg, 0.0) + ll
        n_by_cfg[cfg] = n_by_cfg.get(cfg, 0) + len(actual)
    return {cfg: {"pooled_logloss": ll_by_cfg[cfg] / n_by_cfg[cfg],
                  "n": n_by_cfg[cfg]} for cfg in configs}


# ------------------------------------------------------------ draw diagnosis
def _fit_predict_draw(args):
    train, fixtures = args
    m = dc_fit(train, prior_sigma=DC_P["prior_sigma"],
               half_life_days=DC_P["half_life_days"])
    preds = dc_predict(m, fixtures)
    return np.array([p["p_draw"] for p in preds])


def run_draw_diagnosis():
    df = load_all()
    jobs, actuals = [], []
    for season in ["2024-25", "2025-26"]:
        cur = df[df["season"] == season]
        for week in range(1, int(cur["matchweek"].max()) + 1):
            wg = cur[cur["matchweek"] == week]
            cutoff = wg["date"].min()
            train = df[df["date"] < cutoff]
            fixtures = list(zip(wg["home_team"], wg["away_team"]))
            jobs.append((train, fixtures))
            actuals.append(
                (wg["result"] == "D").to_numpy().astype(float))
    with ProcessPoolExecutor(max_workers=4) as ex:
        draw_probs = list(ex.map(_fit_predict_draw, jobs))
    pred = np.concatenate(draw_probs)
    act = np.concatenate(actuals)
    return {
        "n_games": int(len(act)),
        "mean_predicted_draw": float(pred.mean()),
        "actual_draw_rate": float(act.mean()),
        "se_actual": float(np.sqrt(act.mean() * (1 - act.mean()) / len(act))),
        "note": ("v2 params, walk-forward, tuning window only "
                 "(2024-25/2025-26); eval weeks untouched"),
    }


# ---------------------------------------------------------------------- main
def main():
    print("== 1. bootstrap CIs on 50-game eval ==")
    boot = run_bootstrap()
    for pair, metrics in boot.items():
        if pair == "v1_to_v2_single_game_share":
            continue
        print(f"  {pair}:")
        for metric, s in metrics.items():
            print(f"    {metric:8s} diff={s['mean']:+.4f} "
                  f"se={s['se']:.4f} 95%CI=[{s['ci95_lo']:+.4f},"
                  f"{s['ci95_hi']:+.4f}]")
    sg = boot["v1_to_v2_single_game_share"]
    print(f"  v1->v2: {sg['total_gain_nats']:.2f} nats total, "
          f"{sg['top_game_share']:.1%} from one game: {sg['top_game_fixture']}")

    print("== 2. ablation (pooled log loss, 50 games) ==")
    abl = run_ablation()
    for cfg in ["neither", "shrink", "decay", "both"]:
        print(f"  {cfg:8s} {abl[cfg]['pooled_logloss']:.4f}")

    print("== 3. draw diagnosis (tuning window, 760 games) ==")
    dd = run_draw_diagnosis()
    print(f"  predicted draw {dd['mean_predicted_draw']:.3f} vs actual "
          f"{dd['actual_draw_rate']:.3f} (se {dd['se_actual']:.3f})")

    with open(OUT_PATH / "review_analysis.json", "w") as f:
        json.dump({"bootstrap": boot, "ablation": abl,
                   "draw_diagnosis": dd}, f, indent=2)
    print("wrote outputs/review_analysis.json")


if __name__ == "__main__":
    main()
