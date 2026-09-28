"""Turn a fitted Dixon-Coles model into win/draw/loss probabilities.

For a fixture we compute each side's expected goals (lambda, mu), lay out the
full scoreline probability matrix P(home goals = i, away goals = j) for
i, j in 0..max_goals using the Poisson PMF times the Dixon-Coles tau
correction, normalise it, and then simply add up cells:

    P(home win) = sum of cells below the diagonal (i > j)
    P(draw)     = sum of the diagonal (i == j)
    P(away win) = sum of cells above the diagonal (i < j)

max_goals=10 truncates a negligible tail (P(11+ goals) is ~1e-9 at typical
scoring rates); we renormalise after truncation so probabilities sum to 1.
"""

import numpy as np
from scipy.stats import poisson

from dixon_coles import tau


def scoreline_matrix(model, home_team, away_team, max_goals=10):
    """(max_goals+1) x (max_goals+1) matrix of P(scoreline)."""
    lam, mu = model.expected_goals(home_team, away_team)
    goals = np.arange(max_goals + 1)
    px = poisson.pmf(goals, lam)          # P(home scores i), shape (G+1,)
    py = poisson.pmf(goals, mu)           # P(away scores j)
    joint = np.outer(px, py)              # independence assumption ...
    ii, jj = np.meshgrid(goals, goals, indexing="ij")
    joint = joint * tau(ii, jj, lam, mu, model.rho)   # ... then tau corrects it
    joint = np.clip(joint, 0.0, None)     # guard against tiny negative tau
    return joint / joint.sum()            # renormalise (truncation + clipping)


def outcome_probs(model, home_team, away_team, max_goals=10):
    """(P(home win), P(draw), P(away win)) for one fixture."""
    m = scoreline_matrix(model, home_team, away_team, max_goals)
    p_draw = np.trace(m)
    p_home = np.tril(m, k=-1).sum()   # below diagonal: home scored more
    p_away = np.triu(m, k=1).sum()    # above diagonal: away scored more
    return p_home, p_draw, p_away


def predict_matchweek(model, fixtures, max_goals=10):
    """Probabilities for a list of (home_team, away_team) fixtures.

    Returns a list of dicts with keys home_team, away_team, p_home, p_draw,
    p_away. Teams unseen in training are handled inside the model as
    league-average strength.
    """
    out = []
    for home_team, away_team in fixtures:
        p_h, p_d, p_a = outcome_probs(model, home_team, away_team, max_goals)
        out.append({"home_team": home_team, "away_team": away_team,
                    "p_home": p_h, "p_draw": p_d, "p_away": p_a})
    return out
