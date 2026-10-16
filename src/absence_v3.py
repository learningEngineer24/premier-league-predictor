"""v3 absence model: adjust v2's Poisson means for missing regular starters.

Design (built overnight Oct 15-16, 2026; all judgment calls labeled):

PROBLEM
    Teams missing key players score less and concede more, but "injuries"
    as raw counts are useless -- a missing star striker is not a missing
    backup fullback. We model *observable absence of regular starters*,
    impact-weighted, from historical starting lineups (ESPN free API,
    data/lineups/). Cause (injury/suspension/rotation/rest) is deliberately
    ignored: for prediction, missing minutes are missing minutes. This is
    the reframing adopted Oct 15 after the injury-data scout found no clean
    historical injury database.

JUDGMENT CALLS (each labeled as such):
 1. REGULAR = started >= 60% of the team's league games this season,
    strictly before the prediction cutoff, requiring >= 4 team games played.
    Rationale: 60% separates nailed-on starters from rotation; the 4-game
    minimum means weeks 1-4 carry NO absence signal (all scores zero) --
    honest about early-season, when nobody has earned "regular" status yet.
    Mid-season signings are handled conservatively: a January arrival can't
    reach 60% of the team's games quickly, so they simply don't count until
    they do. No minutes data exists (ESPN gives starter flags only).
    The threshold is a CLI parameter (--regular-share, default 0.60);
    the Oct 16 retest at 0.75 keeps the 60% artifacts untouched.
 2. IMPACT WEIGHT = the player's start-share (0.6-1.0). We only have starter
    flags, not minutes -- start-share is the honest proxy, and we say so
    rather than inventing minutes. A 1.0-share ever-present missing counts
    more than a 0.6-share fringe regular missing.
 3. ATTACK/DEFENSE SPLIT by ESPN position code:
      attack  = {F, CF, CF-L, CF-R, RCF, RF, LF, ST, AM, AM-L, AM-R,
                 LM, RM, LW, RW}
      defense = {G, GK, CD, CD-L, CD-R, CB, LB, RB, WB, LWB, RWB,
                 DM, CM, CM-L, CM-R, M, SW}
    Putting CM/M in defense is a judgment call: central midfielders'
    absence hurts chance suppression more than chance creation on average,
    but reasonable people could split them. Wing-backs (WB/LWB/RWB) are
    defense for the same reason. Any starter position not in either set
    raises an assertion -- silent misclassification is worse than a crash.
 4. MODEL = one parameter gamma >= 0 on v2's Poisson means:
      log lam = log lam_v2 - gamma * (home attack absence)
                            + gamma * (away defense absence)
      log mu  = log mu_v2  - gamma * (away attack absence)
                            + gamma * (home defense absence)
    where each absence score = sum of start-shares of regulars not in the
    starting XI. gamma = 0 reproduces v2 exactly. One parameter, because
    the xg_nudge experiment (Oct 15) showed even one parameter overfits on
    small samples -- here we have 330 tuning games, much healthier.
 5. NO LEAKAGE: regulars are computed strictly from pre-cutoff games each
    week; absence scores use the predicted week's actual lineups, which are
    public ~1 hour before kickoff -- exactly what production would see.
    We use match RESULTS only for scoring, never for features.

VALIDATION (the same discipline as the xG nudge):
    Phase 1 (tune): 2024-25 weeks 5-38, expanding window -- to predict week w,
      gamma is grid-searched on weeks 5..w-1 only. Final gamma* = the week-38
      value (tuned on 330 games).
    Phase 2 (evaluate): 2025-26 weeks 5-38 with FIXED gamma*, v2 vs
      v2+absence, paired bootstrap CIs (10k resamples, seed 7 -- same as
      review_analysis.py). This is fully out-of-sample for gamma.
    Phase 3 (secondary): apply fixed gamma* to 2026-27 weeks 5-6, FLAGGED:
      that window is the live track, so using it for the include decision
      is mildly circular -- reported for color, not for the verdict.

Usage: cd src && python3 absence_v3.py
Writes outputs/absence_v3_predictions.csv, outputs/absence_v3_metrics.csv,
outputs/absence_v3_summary.json and prints the verdict.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import load_all
from dixon_coles import fit as dc_fit, tau
from backtest import OUTCOME_INDEX

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR.parent / "outputs"
LINEUP_DIR = SRC_DIR.parent / "data" / "lineups"
SEED = 7
N_BOOT = 10_000
EPS = 1e-15

GAMMA_GRID = np.arange(0.0, 0.401, 0.025)  # 0..0.40 step 0.025
EVAL_WEEKS_TUNE = list(range(5, 39))        # 2024-25 weeks with absence data
EVAL_WEEKS_TEST = list(range(5, 39))        # 2025-26 weeks with absence data

# --- ESPN displayName -> football-data.co.uk team name, every team seen ---
ESPN_TO_FD = {
    "AFC Bournemouth": "Bournemouth",
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}

ATTACK_POS = {"F", "CF", "CF-L", "CF-R", "RCF", "RF", "LF", "ST",
              "AM", "AM-L", "AM-R", "LM", "RM", "LW", "RW"}
DEFENSE_POS = {"G", "GK", "CD", "CD-L", "CD-R", "CB", "LB", "RB",
               "WB", "LWB", "RWB", "DM", "CM", "CM-L", "CM-R", "M", "SW"}

REGULAR_MIN_SHARE = 0.60
REGULAR_MIN_GAMES = 4


# ------------------------------------------------------------------ loading
def load_lineups():
    """All seasons' lineups, ESPN names mapped to fd names. Fails loudly."""
    frames = []
    for path in sorted(LINEUP_DIR.glob("lineups_*.csv")):
        df = pd.read_csv(path)
        unmapped = set(df["team"]) | set(df["home_team"]) | set(df["away_team"])
        unmapped -= set(ESPN_TO_FD)
        assert not unmapped, f"unmapped ESPN names in {path.name}: {unmapped}"
        for col in ("team", "home_team", "away_team"):
            df[col] = df[col].map(ESPN_TO_FD)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        frames.append(df)
    lp = pd.concat(frames, ignore_index=True)

    # every starter position must classify (judgment call #3 guard)
    starter_pos = set(lp.loc[lp["starter"] == 1, "position"].dropna())
    unclassified = starter_pos - ATTACK_POS - DEFENSE_POS - {"SUB"}
    assert not unclassified, f"unclassified positions: {unclassified}"

    # sanity: 22 starters per game
    starters = lp.groupby(["season", "event_id"])["starter"].sum()
    bad = starters[starters != 22]
    assert bad.empty, f"games without 22 starters: {len(bad)}"
    return lp


def pos_class(pos):
    """'attack' / 'defense' for a position code; None for bench/unknown."""
    if pos in ATTACK_POS:
        return "attack"
    if pos in DEFENSE_POS:
        return "defense"
    return None


# ------------------------------------------------- regulars + absence scores
def compute_regulars(lp_season, cutoff):
    """{team: {player: (start_share, pos_class)}} from pre-cutoff games.

    Judgment call #1: regular = started >= 60% of team games, team needs
    >= 4 games played. Before that, nobody is a regular (no absence signal
    in weeks 1-4 -- honest about early season).
    """
    hist = lp_season[lp_season["date"] < cutoff]
    games = hist[["event_id", "team"]].drop_duplicates()
    team_games = games.groupby("team")["event_id"].nunique()
    starts = (hist[hist["starter"] == 1]
              .groupby(["team", "player"])["event_id"].nunique())

    regulars = {}
    for team, n_games in team_games.items():
        if n_games < REGULAR_MIN_GAMES:
            continue
        team_starts = starts.get(team, pd.Series(dtype=float))
        for player, n_starts in team_starts.items():
            share = n_starts / n_games
            if share >= REGULAR_MIN_SHARE:
                # modal position across the player's starts
                pos = (hist[(hist["team"] == team) &
                            (hist["player"] == player) &
                            (hist["starter"] == 1)]["position"]
                       .mode().iloc[0])
                pc = pos_class(pos)
                if pc is not None:
                    regulars.setdefault(team, {})[player] = (share, pc)
    return regulars


def fixture_absence(lp_game, regulars, home, away):
    """(home_att, home_def, away_att, away_def) absence scores for a fixture.

    Each score = sum of start-shares of that team's regulars (by side of
    the pitch) who are NOT in this game's starting XI. Judgment call #2:
    start-share is the impact weight (honest proxy for missing minutes).
    """
    xi = {team: set(lp_game[(lp_game["team"] == team) &
                            (lp_game["starter"] == 1)]["player"])
          for team in (home, away)}
    out = {}
    for team, side in ((home, "home"), (away, "away")):
        for cls in ("attack", "defense"):
            out[f"{side}_{cls[:3]}"] = sum(
                share for player, (share, pc) in regulars.get(team, {}).items()
                if pc == cls and player not in xi[team])
    return out["home_att"], out["home_def"], out["away_att"], out["away_def"]


# ------------------------------------------------------- model + validation
def outcome_probs_from_means(lam, mu, rho, max_goals=10):
    """(P(home), P(draw), P(away)) from Poisson means -- mirrors predict.py."""
    goals = np.arange(max_goals + 1)
    joint = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    ii, jj = np.meshgrid(goals, goals, indexing="ij")
    joint = np.clip(joint * tau(ii, jj, lam, mu, rho), 0.0, None)
    joint = joint / joint.sum()
    return (float(np.tril(joint, k=-1).sum()), float(np.trace(joint)),
            float(np.triu(joint, k=1).sum()))


def apply_gamma(lam, mu, rho, gamma, hatt, hdef, aatt, adef):
    """Judgment call #4: one-parameter absence nudge on v2's Poisson means."""
    lam_n = lam * np.exp(-gamma * hatt + gamma * adef)
    mu_n = mu * np.exp(-gamma * aatt + gamma * hdef)
    return outcome_probs_from_means(lam_n, mu_n, rho)


def _week_job(args):
    """Worker: fit v2 for one week, compute absence-adjusted inputs.

    Returns dict with lam/mu/rho/absence arrays + actuals. Heavy part is the
    v2 fit; everything else is slicing.
    """
    (season, week, cutoff, fixtures, train, lp_season,
     lineup_index, dc_p) = args
    model = dc_fit(train, prior_sigma=dc_p["prior_sigma"],
                   half_life_days=dc_p["half_life_days"])
    regulars = compute_regulars(lp_season, cutoff)

    lam, mu, abs_rows, actual, ok = [], [], [], [], True
    for (h, a, res, fdate) in fixtures:
        key = (season, h, a, fdate)   # fd names; lineups already mapped
        ev = lineup_index.get(key)
        if ev is None:
            ok = False
            break
        lg = lp_season[lp_season["event_id"] == ev]
        hatt, hdef, aatt, adef = fixture_absence(lg, regulars, h, a)
        l, m = model.expected_goals(h, a)
        lam.append(l)
        mu.append(m)
        abs_rows.append((hatt, hdef, aatt, adef))
        actual.append(OUTCOME_INDEX[res])
    if not ok:
        return None
    return {"season": season, "week": week,
            "lam": np.array(lam), "mu": np.array(mu), "rho": model.rho,
            "abs": np.array(abs_rows),
            "actual": np.array(actual),
            "fixtures": [(h, a) for (h, a, _, _) in fixtures]}


def build_week_cache(df, lp, seasons_weeks, dc_p, workers=4):
    """v2 fits + absence inputs per (season, week). Parallel over weeks."""
    jobs = []
    for season, weeks in seasons_weeks:
        cur = df[df["season"] == season]
        lp_season = lp[lp["season"] == season].copy()
        # (season, home_fd, away_fd, date) -> event_id; lineups already
        # mapped to fd names in load_lineups()
        lineup_index = {}
        for (h, a, d), idx in lp_season.groupby(
                ["home_team", "away_team", "date"]).groups.items():
            lineup_index[(season, h, a, d)] = lp_season.loc[idx[0],
                                                            "event_id"]
        for week in weeks:
            wg = cur[cur["matchweek"] == week]
            cutoff = wg["date"].min().date()
            train = df[df["date"] < wg["date"].min()]
            fixtures = [(r["home_team"], r["away_team"], r["result"],
                         r["date"].date())
                        for _, r in wg.iterrows()]
            jobs.append((season, week, cutoff, fixtures, train,
                         lp_season, lineup_index, dc_p))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_week_job, jobs))
    assert all(r is not None for r in results), \
        "some weeks failed lineup matching -- investigate before trusting this"
    return {(r["season"], r["week"]): r for r in results}


def predict_week(c, gamma):
    """Outcome probs for a cached week under gamma (0 = v2)."""
    n = len(c["actual"])
    if gamma == 0.0:
        return np.array([outcome_probs_from_means(c["lam"][i], c["mu"][i],
                                                  c["rho"])
                         for i in range(n)])
    return np.array([apply_gamma(c["lam"][i], c["mu"][i], c["rho"], gamma,
                                 *c["abs"][i]) for i in range(n)])


def per_game_metrics(probs, actual):
    p = np.clip(probs[np.arange(len(actual)), actual], EPS, 1.0)
    ll = -np.log(p)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    br = np.sum((probs - onehot) ** 2, axis=1)
    acc = (probs.argmax(axis=1) == actual).astype(float)
    return ll, br, acc


def tune_gamma(cache, season, weeks, grid=GAMMA_GRID):
    """Expanding-window gamma*: to predict week w, grid-search on weeks < w.

    Returns {week: gamma*}. Ties keep the smaller gamma (anti-overfit).
    """
    out = {}
    for w in weeks:
        tune_weeks = [v for v in weeks if v < w]
        if not tune_weeks:
            out[w] = 0.0
            continue
        best_g, best_ll = 0.0, np.inf
        for gamma in grid:
            tot = sum(per_game_metrics(
                predict_week(cache[(season, v)], gamma),
                cache[(season, v)]["actual"])[0].sum()
                for v in tune_weeks)
            n = sum(len(cache[(season, v)]["actual"]) for v in tune_weeks)
            if tot / n < best_ll - 1e-12:
                best_g, best_ll = gamma, tot / n
        out[w] = best_g
    return out


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=SEED):
    """95% CI on mean(a - b). Same pattern as review_analysis.py."""
    rng = np.random.default_rng(seed)
    d = a - b
    boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                      for _ in range(n_boot)])
    return {"mean": float(d.mean()),
            "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "ci95_lo": float(np.percentile(boots, 2.5)),
            "ci95_hi": float(np.percentile(boots, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regular-share", type=float, default=0.60,
                    help="start-share threshold for 'regular' status "
                         "(0.60 default; 0.75 retest requested Oct 16)")
    args = ap.parse_args()
    global REGULAR_MIN_SHARE
    REGULAR_MIN_SHARE = args.regular_share
    tag = "" if args.regular_share == 0.60 \
        else f"_{int(round(args.regular_share * 100))}"
    print(f"REGULAR_MIN_SHARE = {REGULAR_MIN_SHARE}", flush=True)
    df = load_all()
    with open(OUT_PATH / "best_params.json") as f:
        dc_p = json.load(f)["dc"]
    print(f"v2 params: {dc_p}", flush=True)
    lp = load_lineups()
    print(f"lineups: {len(lp)} rows, {lp['event_id'].nunique()} games, "
          f"seasons {sorted(lp['season'].unique())}", flush=True)

    # every fd fixture in the eval weeks must have a lineup game
    # 2026-27: fd file lags the season (has MW1-5); use whichever of weeks
    # 5-6 exist in the fd data -- week 6 needs tomorrow's MW7 data refresh.
    w26 = [w for w in (5, 6)
           if w in set(df[df["season"] == "2026-27"]["matchweek"])]
    cache = build_week_cache(
        df, lp,
        [("2024-25", EVAL_WEEKS_TUNE), ("2025-26", EVAL_WEEKS_TEST),
         ("2026-27", w26)],
        dc_p)
    print(f"week cache built: {len(cache)} weeks", flush=True)

    # ---- Phase 1: tune gamma on 2024-25 (expanding window) ----
    gamma_star = tune_gamma(cache, "2024-25", EVAL_WEEKS_TUNE)
    traj = [gamma_star[w] for w in EVAL_WEEKS_TUNE]
    final_gamma = gamma_star[EVAL_WEEKS_TUNE[-1]]
    print(f"gamma* trajectory (2024-25): "
          f"{' '.join(f'{g:.3f}' for g in traj[::4])} ... -> {final_gamma:.3f}",
          flush=True)

    # pooled tuning curve on 2024-25 for shape
    print("tuning curve (pooled logloss, 2024-25 wks 5-38):", flush=True)
    for gamma in GAMMA_GRID[::2]:
        tot = n = 0
        for w in EVAL_WEEKS_TUNE:
            c = cache[("2024-25", w)]
            tot += per_game_metrics(
                predict_week(c, gamma), c["actual"])[0].sum()
            n += len(c["actual"])
        print(f"  gamma={gamma:.3f}  ll={tot / n:.4f}", flush=True)

    # ---- Phase 2: evaluate fixed gamma* on 2025-26 (fully out-of-sample) ----
    rows, all_v2, all_abs = [], [], []
    for w in EVAL_WEEKS_TEST:
        c = cache[("2025-26", w)]
        p_v2 = predict_week(c, 0.0)
        p_abs = predict_week(c, final_gamma)
        ll_v2, br_v2, ac_v2 = per_game_metrics(p_v2, c["actual"])
        ll_ab, br_ab, ac_ab = per_game_metrics(p_abs, c["actual"])
        all_v2.append((ll_v2, br_v2, ac_v2))
        all_abs.append((ll_ab, br_ab, ac_ab))
        for i, (h, a) in enumerate(c["fixtures"]):
            rows.append({"season": "2025-26", "week": w, "home": h,
                         "away": a, "actual": c["actual"][i],
                         "gamma": final_gamma,
                         "v2_h": p_v2[i, 0], "v2_d": p_v2[i, 1],
                         "v2_a": p_v2[i, 2],
                         "abs_h": p_abs[i, 0], "abs_d": p_abs[i, 1],
                         "abs_a": p_abs[i, 2]})
    ll_v2 = np.concatenate([x[0] for x in all_v2])
    br_v2 = np.concatenate([x[1] for x in all_v2])
    ac_v2 = np.concatenate([x[2] for x in all_v2])
    ll_ab = np.concatenate([x[0] for x in all_abs])
    br_ab = np.concatenate([x[1] for x in all_abs])
    ac_ab = np.concatenate([x[2] for x in all_abs])

    boot = {
        # positive diff = v2+absence BETTER (absence minus v2 of loss)
        "logloss": paired_bootstrap(ll_v2, ll_ab),
        "brier": paired_bootstrap(br_v2, br_ab),
        "accuracy": paired_bootstrap(ac_ab, ac_v2),
    }
    print(f"\n== Phase 2: 2025-26 wks 5-38, gamma*={final_gamma:.3f} "
          f"(n={len(ll_v2)}) ==", flush=True)
    print(f"  logloss  v2={ll_v2.mean():.4f} abs={ll_ab.mean():.4f} "
          f"diff=({boot['logloss']['mean']:+.4f}) "
          f"95%CI=[{boot['logloss']['ci95_lo']:+.4f},"
          f"{boot['logloss']['ci95_hi']:+.4f}]", flush=True)
    print(f"  brier    v2={br_v2.mean():.4f} abs={br_ab.mean():.4f} "
          f"diff=({boot['brier']['mean']:+.4f}) "
          f"95%CI=[{boot['brier']['ci95_lo']:+.4f},"
          f"{boot['brier']['ci95_hi']:+.4f}]", flush=True)
    print(f"  accuracy v2={ac_v2.mean():.3f} abs={ac_ab.mean():.3f} "
          f"diff=({boot['accuracy']['mean']:+.4f}) "
          f"95%CI=[{boot['accuracy']['ci95_lo']:+.4f},"
          f"{boot['accuracy']['ci95_hi']:+.4f}]", flush=True)

    # ---- Phase 3 (secondary, flagged): 2026-27 available weeks ----
    print("\n== Phase 3 (SECONDARY, selection caveat): 2026-27 ==",
          flush=True)
    for w in w26:
        c = cache[("2026-27", w)]
        p_v2 = predict_week(c, 0.0)
        p_abs = predict_week(c, final_gamma)
        ll_v, _, ac_v = per_game_metrics(p_v2, c["actual"])
        ll_a, _, ac_a = per_game_metrics(p_abs, c["actual"])
        print(f"  week {w}: v2 ll={ll_v.mean():.3f} acc={ac_v.mean():.3f} | "
              f"abs ll={ll_a.mean():.3f} acc={ac_a.mean():.3f}", flush=True)
        for i, (h, a) in enumerate(c["fixtures"]):
            rows.append({"season": "2026-27", "week": w, "home": h,
                         "away": a, "actual": c["actual"][i],
                         "gamma": final_gamma,
                         "v2_h": p_v2[i, 0], "v2_d": p_v2[i, 1],
                         "v2_a": p_v2[i, 2],
                         "abs_h": p_abs[i, 0], "abs_d": p_abs[i, 1],
                         "abs_a": p_abs[i, 2]})

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_PATH / f"absence_v3_predictions{tag}.csv", index=False)
    summary = {
        "regular_min_share": REGULAR_MIN_SHARE,
        "gamma_star": final_gamma,
        "gamma_trajectory_2024_25": {str(w): gamma_star[w]
                                     for w in EVAL_WEEKS_TUNE},
        "phase2_2025_26_weeks_5_38": {
            "n": int(len(ll_v2)),
            "v2": {"logloss": float(ll_v2.mean()),
                   "brier": float(br_v2.mean()),
                   "accuracy": float(ac_v2.mean())},
            "v2_absence": {"logloss": float(ll_ab.mean()),
                           "brier": float(br_ab.mean()),
                           "accuracy": float(ac_ab.mean())},
            "paired_bootstrap_absence_minus_v2_loss": boot,
        },
        "note": ("Phase 2 is fully out-of-sample for gamma (tuned on "
                 "2024-25 only). Phase 3 (2026-27) is secondary: that "
                 "window is the live track, so it must not decide "
                 "inclusion. Positive logloss/brier diff = absence better; "
                 "positive accuracy diff = absence better."),
    }
    with open(OUT_PATH / f"absence_v3_summary{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT_PATH / f'absence_v3_predictions{tag}.csv'} and "
          f"{OUT_PATH / f'absence_v3_summary{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
