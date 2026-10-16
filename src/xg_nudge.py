"""Experimental: one-parameter xG nudge on the tuned v2 Dixon-Coles model.

Context (Oct 15, 2026): xG columns (HxG/AxG) exist only for the 2026-27
season -- football-data.co.uk publishes no xG for earlier E0 seasons, so a
full xG-strengths model cannot be tuned honestly. Instead v2 stays the
backbone and each team's recent chance quality nudges the Poisson means
through ONE parameter, beta:

    team xG rating  = decayed, shrunk mean of per-game (xG_for - xG_against),
                      current season only, games strictly before the cutoff
    log lam = log lam_v2 + beta * (xGD_home - xGD_away) / 2
    log mu  = log mu_v2  - beta * (xGD_home - xGD_away) / 2

beta >= 0; beta = 0 reproduces v2 exactly. Shrinkage (3 effective games
toward 0) and the v2 time-decay half-life are FIXED, not tuned -- with
~50 games of xG, one tuned parameter is already ambitious.

Honest protocol: expanding-window walk-forward on 2026-27 weeks 2-5. To
predict week w, beta is tuned by grid search on weeks 2..w-1 only; week 2
uses beta = 0 (no xG history to tune on). Pooled metrics compare the
beta-nudged model against v2 on the same 40 games.

Usage: cd src && python3 xg_nudge.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all
from dixon_coles import fit as dc_fit, tau
from backtest import log_loss, brier_score, accuracy, OUTCOME_INDEX

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR.parent / "outputs"

BETA_GRID = np.arange(0.0, 0.51, 0.05)
SHRINK_GAMES = 3.0   # fixed shrinkage toward 0, in effective-game units
EVAL_WEEKS = (2, 3, 4, 5)


def outcome_probs_from_means(lam, mu, rho, max_goals=10):
    """(P(home), P(draw), P(away)) from Poisson means -- mirrors predict.py."""
    goals = np.arange(max_goals + 1)
    joint = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    ii, jj = np.meshgrid(goals, goals, indexing="ij")
    joint = joint * tau(ii, jj, lam, mu, rho)
    joint = np.clip(joint, 0.0, None)
    joint = joint / joint.sum()
    return float(np.tril(joint, k=-1).sum()), float(np.trace(joint)), \
        float(np.triu(joint, k=1).sum())


def xg_ratings(xg_hist, cutoff, half_life_days, shrink_games=SHRINK_GAMES):
    """Decayed, shrunk per-team mean of (xG_for - xG_against).

    xg_hist: games with xG strictly before cutoff. Returns dict team -> rating.
    """
    days_ago = (cutoff - xg_hist["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (days_ago / half_life_days)
    num, den = {}, {}
    for (_, g), wi in zip(xg_hist.iterrows(), w):
        for team, xgd in ((g["home_team"], g["hxg"] - g["axg"]),
                          (g["away_team"], g["axg"] - g["hxg"])):
            num[team] = num.get(team, 0.0) + wi * xgd
            den[team] = den.get(team, 0.0) + wi
    return {t: num[t] / (den[t] + shrink_games) for t in num}


def nudged_probs(lam, mu, rho, beta, xgd_home, xgd_away):
    """Apply the beta nudge to v2's Poisson means, return outcome probs."""
    delta = beta * (xgd_home - xgd_away) / 2.0
    lam_n = lam * np.exp(delta)
    mu_n = mu * np.exp(-delta)
    return outcome_probs_from_means(lam_n, mu_n, rho)


def tune_beta(week_cache, tune_weeks, grid=BETA_GRID):
    """Grid-search beta on tune_weeks (expanding window); ties -> smaller beta."""
    best_beta, best_ll = 0.0, np.inf
    for beta in grid:
        lls = []
        for w in tune_weeks:
            c = week_cache[w]
            pr = np.array([nudged_probs(c["lam"][i], c["mu"][i], c["rho"],
                                        beta, c["xgd_h"][i], c["xgd_a"][i])
                           for i in range(len(c["actual"]))])
            lls.append(log_loss(pr, c["actual"]) * len(c["actual"]))
        total_ll = sum(lls) / sum(len(week_cache[w]["actual"])
                                  for w in tune_weeks)
        if total_ll < best_ll - 1e-12:   # strict: ties keep the smaller beta
            best_beta, best_ll = beta, total_ll
    return best_beta, best_ll


def main():
    df = load_all()
    with open(OUT_PATH / "best_params.json") as f:
        dc_p = json.load(f)["dc"]
    print(f"v2 params: {dc_p}")

    current = df[df["season"] == "2026-27"].copy()
    xg_season = current[current["hxg"].notna()].copy()

    # --- one v2 fit per evaluation week; cache everything beta needs ---
    week_cache = {}
    for w in EVAL_WEEKS:
        week_games = current[current["matchweek"] == w]
        cutoff = week_games["date"].min()
        train = df[df["date"] < cutoff]
        model = dc_fit(train, prior_sigma=dc_p["prior_sigma"],
                       half_life_days=dc_p["half_life_days"])
        ratings = xg_ratings(xg_season[xg_season["date"] < cutoff], cutoff,
                             dc_p["half_life_days"])
        fixtures = list(zip(week_games["home_team"], week_games["away_team"]))
        lam = np.array([model.expected_goals(h, a)[0] for h, a in fixtures])
        mu = np.array([model.expected_goals(h, a)[1] for h, a in fixtures])
        week_cache[w] = {
            "model": model, "rho": model.rho,
            "fixtures": fixtures, "lam": lam, "mu": mu,
            "xgd_h": np.array([ratings.get(h, 0.0) for h, _ in fixtures]),
            "xgd_a": np.array([ratings.get(a, 0.0) for _, a in fixtures]),
            "actual": week_games["result"].map(OUTCOME_INDEX).to_numpy(),
            "n_tune_games": len(xg_season[xg_season["date"] < cutoff]),
        }
        print(f"week {w}: cutoff {cutoff.date()}, "
              f"xG history games: {week_cache[w]['n_tune_games']}")

    # --- expanding-window beta tuning + evaluation ---
    rows = []
    for w in EVAL_WEEKS:
        tune_weeks = [v for v in EVAL_WEEKS if v < w]
        beta_star, tune_ll = (0.0, np.nan) if not tune_weeks else \
            tune_beta(week_cache, tune_weeks)
        c = week_cache[w]
        pr_v2 = np.array([outcome_probs_from_means(c["lam"][i], c["mu"][i],
                                                   c["rho"])
                          for i in range(len(c["actual"]))])
        pr_n = np.array([nudged_probs(c["lam"][i], c["mu"][i], c["rho"],
                                      beta_star, c["xgd_h"][i], c["xgd_a"][i])
                         for i in range(len(c["actual"]))])
        rows.append({"week": w, "beta_star": beta_star,
                     "v2_logloss": log_loss(pr_v2, c["actual"]),
                     "nudged_logloss": log_loss(pr_n, c["actual"]),
                     "v2_brier": brier_score(pr_v2, c["actual"]),
                     "nudged_brier": brier_score(pr_n, c["actual"]),
                     "v2_acc": accuracy(pr_v2, c["actual"]),
                     "nudged_acc": accuracy(pr_n, c["actual"])})
        print(f"week {w}: beta*={beta_star:.2f} "
              f"(tuned on weeks {tune_weeks or 'none'}) | "
              f"v2 ll={rows[-1]['v2_logloss']:.3f} vs "
              f"nudged ll={rows[-1]['nudged_logloss']:.3f}")

    # --- pooled verdict (40 games) ---
    print("\n=== pooled over weeks 2-5 (40 games) ===")
    for m in ("logloss", "brier", "acc"):
        v2 = np.mean([r[f"v2_{m}"] for r in rows])
        nn = np.mean([r[f"nudged_{m}"] for r in rows])
        print(f"{m:8s}  v2={v2:.4f}  nudged={nn:.4f}  "
              f"delta={nn - v2:+.4f} {'(better)' if (nn < v2 if m != 'acc' else nn > v2) else ''}")

    # --- paired uncertainty on the log-loss delta (the honest error bar) ---
    diffs = []
    for w in EVAL_WEEKS:
        tune_weeks = [v for v in EVAL_WEEKS if v < w]
        beta_star = 0.0 if not tune_weeks else tune_beta(week_cache,
                                                         tune_weeks)[0]
        c = week_cache[w]
        for i in range(len(c["actual"])):
            p_v2 = np.array(outcome_probs_from_means(c["lam"][i], c["mu"][i],
                                                     c["rho"]))
            p_n = np.array(nudged_probs(c["lam"][i], c["mu"][i], c["rho"],
                                        beta_star, c["xgd_h"][i],
                                        c["xgd_a"][i]))
            a = c["actual"][i]
            diffs.append(-np.log(p_n[a]) - -np.log(p_v2[a]))
    diffs = np.array(diffs)
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    print(f"\npaired log-loss delta (nudged - v2): {diffs.mean():+.4f} "
          f"+/- {1.96 * se:.4f} (95% CI, n={len(diffs)})")

    # --- beta tuning curve on all tunable weeks (shape check) ---
    print("\n=== beta grid curve, tuned on weeks 2-4 ===")
    for beta in BETA_GRID:
        lls = []
        for w in (2, 3, 4):
            c = week_cache[w]
            pr = np.array([nudged_probs(c["lam"][i], c["mu"][i], c["rho"],
                                         beta, c["xgd_h"][i], c["xgd_a"][i])
                           for i in range(len(c["actual"]))])
            lls.append(log_loss(pr, c["actual"]))
        print(f"beta={beta:.2f}  pooled_ll={np.mean(lls):.4f}")

    # --- eyeball the ratings the nudge is using (week 5 cutoff) ---
    print("\n=== xG ratings at week-5 cutoff (shrunk, decayed xGD/game) ===")
    r5 = xg_ratings(xg_season[xg_season["date"]
                              < current[current["matchweek"] == 5]["date"].min()],
                    current[current["matchweek"] == 5]["date"].min(),
                    dc_p["half_life_days"])
    for team, r in sorted(r5.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{team:20s} {r:+.3f}")


if __name__ == "__main__":
    main()
