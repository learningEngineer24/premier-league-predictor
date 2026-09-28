"""Elo ratings model for soccer, with an ordered-logit outcome mapping.

The challenger to Dixon-Coles. Two parts:

1. Ratings: every team has one number (start 1500). After each game, in
   chronological order,
       E_home = 1 / (1 + 10 ** (-((R_home + H) - R_away) / 400))
       R_home += K * mov_mult * (S - E_home)
       R_away -= K * mov_mult * (S - E_home)
   where S is 1 / 0.5 / 0 for home win / draw / away win, H is the home
   bonus in rating points, K the k-factor, and mov_mult an optional
   margin-of-victory multiplier (bigger wins move ratings more):
       mov_mult = 1 + 0.5 * ln(1 + |goal difference|)   (decisive games only)

2. Outcome mapping: ratings alone give P(home win) via E_home, but draws
   need their own treatment. We fit an ordered logit on the training games'
   pre-match rating differences d = (R_home + H) - R_away:
       P(home win)       = sigmoid(a1 + b * d)
       P(home win/draw)  = sigmoid(a2 + b * d),   a1 < a2
       P(draw)           = the difference of the two
       P(away win)       = 1 - P(home win/draw)
   a1, a2, b are fit by maximum likelihood (3 parameters, thousands of
   games -- no overfitting risk). This is sometimes called "ordered logistic
   regression" or the proportional-odds model.

Honesty rules (same as the Dixon-Coles backtest): at each walk-forward step
the ratings are replayed from scratch using only games before the cutoff
date, and the ordered logit is fit only on those games. Bookmaker odds are
never touched.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # the sigmoid function


class EloModel:
    """Ratings plus a fitted ordered-logit mapping.

    Attributes:
        ratings: dict team -> rating as of the cutoff date
        k, home_bonus, mov: hyper-parameters used
        a1, a2, b: ordered-logit parameters (a1 < a2)
    """

    def __init__(self, ratings, k, home_bonus, mov, a1, a2, b):
        self.ratings = dict(ratings)
        self.k = k
        self.home_bonus = home_bonus
        self.mov = mov
        self.a1, self.a2, self.b = a1, a2, b

    def rating(self, team):
        """Current rating; unseen teams (promoted sides) start at 1500."""
        return self.ratings.get(team, 1500.0)

    def effective_diff(self, home_team, away_team):
        """Pre-match rating difference including the home bonus."""
        return (self.rating(home_team) + self.home_bonus
                - self.rating(away_team))

    def outcome_probs(self, home_team, away_team):
        """(P(home win), P(draw), P(away win)) for one fixture."""
        d = self.effective_diff(home_team, away_team)
        p_home = expit(self.a1 + self.b * d)
        p_home_or_draw = expit(self.a2 + self.b * d)
        # Numerical guard: the two sigmoids can tie at extreme diffs.
        p_draw = max(p_home_or_draw - p_home, 1e-12)
        p_away = max(1.0 - p_home_or_draw, 1e-12)
        tot = p_home + p_draw + p_away
        return p_home / tot, p_draw / tot, p_away / tot


def _mov_multiplier(home_goals, away_goals):
    """1 + 0.5*ln(1+margin) for decisive games, 1.0 for draws."""
    margin = abs(home_goals - away_goals)
    if margin == 0:
        return 1.0
    return 1.0 + 0.5 * math.log(1 + margin)


def replay_ratings(df, k, home_bonus, mov, base_rating=1500.0):
    """Replay every game in df (chronological) updating Elo ratings.

    Returns (ratings dict, diffs list, outcomes list): for each game, the
    pre-match effective rating difference and the outcome code 0/1/2
    (home/draw/away), which is what the ordered logit trains on.
    """
    ratings = {}
    diffs, outcomes = [], []
    for row in df.itertuples():
        rh = ratings.get(row.home_team, base_rating)
        ra = ratings.get(row.away_team, base_rating)
        d = (rh + home_bonus) - ra
        diffs.append(d)
        outcomes.append({"H": 0, "D": 1, "A": 2}[row.result])

        e_home = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
        s = {"H": 1.0, "D": 0.5, "A": 0.0}[row.result]
        mult = _mov_multiplier(row.home_goals, row.away_goals) if mov else 1.0
        delta = k * mult * (s - e_home)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
    return ratings, np.array(diffs), np.array(outcomes)


def _ordered_logit_nll(theta, d, y):
    """Negative log-likelihood for the ordered logit.

    theta = (a1, log_delta, b) with a2 = a1 + exp(log_delta) > a1.
    y in {0: home win, 1: draw, 2: away win}.
    """
    a1, log_delta, b = theta
    a2 = a1 + math.exp(log_delta)
    # Clip every probability away from 0/1 BEFORE the log: at extreme
    # rating differences a sigmoid can underflow to exactly 0.0, and
    # log(0) would poison the objective (and the optimizer's finite
    # differences) with -inf.
    p_home = np.clip(expit(a1 + b * d), 1e-12, 1.0 - 1e-12)
    p_hd = np.clip(expit(a2 + b * d), 1e-12, 1.0 - 1e-12)
    p_draw = np.clip(p_hd - p_home, 1e-12, 1.0)
    p_away = np.clip(1.0 - p_hd, 1e-12, 1.0)
    nll = -(np.log(np.where(y == 0, p_home,
                    np.where(y == 1, p_draw, p_away)))).sum()
    return nll


def fit_ordered_logit(d, y):
    """MLE for (a1, a2, b); returns (a1, a2, b)."""
    res = minimize(_ordered_logit_nll, x0=np.array([0.0, 0.0, 0.005]),
                   args=(d, y), method="L-BFGS-B",
                   options={"maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"ordered logit did not converge: {res.message}")
    a1, log_delta, b = res.x
    return a1, a1 + math.exp(log_delta), b


def fit(df, k=24.0, home_bonus=70.0, mov=False):
    """Fit an EloModel on a matches DataFrame (must be chronological).

    Expects columns: date, home_team, away_team, home_goals, away_goals,
    result (H/D/A).
    """
    df = df.sort_values("date", kind="mergesort")
    ratings, diffs, outcomes = replay_ratings(df, k, home_bonus, mov)
    a1, a2, b = fit_ordered_logit(diffs, outcomes)
    return EloModel(ratings, k, home_bonus, mov, a1, a2, b)


def predict_matchweek(model, fixtures):
    """List of dicts with p_home/p_draw/p_away for each fixture."""
    out = []
    for home_team, away_team in fixtures:
        p_h, p_d, p_a = model.outcome_probs(home_team, away_team)
        out.append({"home_team": home_team, "away_team": away_team,
                    "p_home": p_h, "p_draw": p_d, "p_away": p_a})
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from load_data import load_all
    df = load_all()
    cutoff = df[(df["season"] == "2026-27")
                & (df["matchweek"] == 1)]["date"].min()
    train = df[df["date"] < cutoff]
    m = fit(train, k=24.0, home_bonus=70.0, mov=True)
    print(f"a1={m.a1:.3f} a2={m.a2:.3f} b={m.b:.5f}")
    for t, r in sorted(m.ratings.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {t:15s} {r:7.1f}")
    p = m.outcome_probs("Arsenal", "Coventry")
    print("Arsenal v Coventry: P(H)=%.2f P(D)=%.2f P(A)=%.2f" % p)
