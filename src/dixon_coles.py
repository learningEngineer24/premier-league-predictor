"""Dixon-Coles Poisson model for soccer, implemented from scratch.

The idea (Dixon & Coles, 1997): the number of goals each team scores in a
match follows a Poisson distribution whose mean depends on the two teams'
strengths:

    home goals ~ Poisson(lambda),  away goals ~ Poisson(mu)
    lambda = exp(attack_home + defense_away + home_advantage)
    mu     = exp(attack_away + defense_home)

where `attack_i` is team i's attacking strength and `defense_j` is team j's
defensive *weakness* (positive = concedes more than average). Each team
therefore has two learned numbers.

Two refinements make it "Dixon-Coles" rather than plain Poisson:
  1. home_advantage: a single global term; home teams score more.
  2. rho: independent Poissons underpredict low-scoring draws (0-0, 1-1).
     The tau() correction nudges probability mass between the 0-0, 0-1,
     1-0 and 1-1 scorelines.

Identifiability: adding a constant to every attack strength (and subtracting
it from every defense) leaves every lambda/mu unchanged, so the raw
parameters are not unique. We fix this with a sum-to-zero constraint --
we optimise the strengths of the first T-1 teams and define the last team's
strength as minus the sum of the rest. A strength of 0 therefore means
exactly "league average", which is also the prior we give promoted teams
with no history.

Fitting: maximum likelihood via scipy.optimize (L-BFGS-B). v2 adds two
regularisation ideas on top of plain MLE:
  * L2 shrinkage: a Gaussian prior, strength ~ N(0, sigma^2), on every team's
    attack and defense. This is MAP estimation rather than pure MLE: teams
    with tiny samples (e.g. a promoted side after 4 games) get pulled toward
    league-average instead of exploding to +/-5. sigma is tuned by
    walk-forward log loss on held-out seasons.
  * time decay: exponential down-weighting of older matches in the
    likelihood, w = 0.5 ** (days_ago / half_life_days), exactly as in the
    original Dixon-Coles paper. Recent form counts more; half_life_days is
    tuned the same way.
Both default to off (sigma=None, half_life_days=None) so fit(df) is the v1
model, byte-for-byte.
"""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


# ---------------------------------------------------------------------------
# Poisson + Dixon-Coles correction
# ---------------------------------------------------------------------------

def poisson_logpmf(k, lam):
    """log P(X = k) for X ~ Poisson(lam), computed in log-space for stability.

    Accepts scalars or numpy arrays.
    """
    k = np.asarray(k, dtype=float)
    lam = np.asarray(lam, dtype=float)
    return k * np.log(lam) - lam - gammaln(k + 1)


def tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction factor.

    Multiplies the independent-Poisson probability of scoreline (x, y).
    Only the four lowest scorelines are adjusted; everything else keeps
    tau = 1. Works elementwise on scalars or broadcastable arrays.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    shape = np.broadcast_shapes(x.shape, y.shape, lam.shape, mu.shape)
    xb = np.broadcast_to(x, shape)
    yb = np.broadcast_to(y, shape)
    lamb = np.broadcast_to(lam, shape)
    mub = np.broadcast_to(mu, shape)
    t = np.ones(shape, dtype=float)
    m = (xb == 0) & (yb == 0)
    t[m] = 1 - lamb[m] * mub[m] * rho
    m = (xb == 0) & (yb == 1)
    t[m] = 1 + lamb[m] * rho
    m = (xb == 1) & (yb == 0)
    t[m] = 1 + mub[m] * rho
    m = (xb == 1) & (yb == 1)
    t[m] = 1 - rho
    return t


# ---------------------------------------------------------------------------
# Fitted model container
# ---------------------------------------------------------------------------

class DixonColesModel:
    """A fitted Dixon-Coles model.

    Attributes:
        teams: list of team names (index i <-> strengths below)
        attack: np.ndarray, attacking strength per team (mean 0)
        defense: np.ndarray, defensive weakness per team (mean 0)
        home_adv: float, global home advantage (log scale)
        rho: float, low-score dependence correction
    """

    def __init__(self, teams, attack, defense, home_adv, rho):
        self.teams = list(teams)
        self.attack = np.asarray(attack, dtype=float)
        self.defense = np.asarray(defense, dtype=float)
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self._index = {t: i for i, t in enumerate(self.teams)}

    def strengths(self, team):
        """(attack, defense) for a team; (0, 0) = league average.

        Teams with no history in the training data (e.g. newly promoted
        sides) get league-average strength rather than zeros-that-mean-
        something-else: under the sum-to-zero constraint, 0 IS the average.
        """
        i = self._index.get(team)
        if i is None:
            return 0.0, 0.0
        return self.attack[i], self.defense[i]

    def expected_goals(self, home_team, away_team):
        """(lambda, mu): expected goals for the home and away team."""
        a_h, d_h = self.strengths(home_team)
        a_a, d_a = self.strengths(away_team)
        lam = math.exp(a_h + d_a + self.home_adv)
        mu = math.exp(a_a + d_h)
        return lam, mu


# ---------------------------------------------------------------------------
# Maximum-likelihood fitting
# ---------------------------------------------------------------------------

def _unpack_params(theta, n_teams):
    """Split the flat parameter vector; last team gets the sum-to-zero value."""
    n = n_teams - 1
    attack_free = theta[0:n]
    defense_free = theta[n:2 * n]
    home_adv, rho = theta[2 * n], theta[2 * n + 1]
    attack = np.append(attack_free, -attack_free.sum())
    defense = np.append(defense_free, -defense_free.sum())
    return attack, defense, home_adv, rho


def _nll_and_grad(theta, h, a, x, y, n, w=None, prior_prec=0.0):
    """Negative log-likelihood and its analytic gradient.

    h, a: int arrays of home/away team indices; x, y: goals; n: # teams.
    theta: flat parameter vector (see _unpack_params).
    w: optional per-match likelihood weights (time decay); None = all ones.
    prior_prec: 1/sigma^2 of the Gaussian prior on team strengths
        (L2 shrinkage); 0.0 = pure MLE. The prior applies only to team
        strengths, not to home_adv or rho.

    The gradient is derived by hand with the chain rule -- this is the
    "boring but important" part of MLE: for each match,
        ll = log tau + x*log(lam) - lam + y*log(mu) - mu + const
    so d ll/d lam = x/lam - 1 + (1/tau)(d tau/d lam), and lam itself is
    exp(attack_home + defense_away + home_adv). The sum-to-zero constraint
    (last team's strength = -sum of the rest) contributes the "- g[last]"
    terms below: nudging a free parameter also nudges the last team's
    strength in the opposite direction.
    """
    attack, defense, home_adv, rho = _unpack_params(theta, n)
    lam = np.exp(attack[h] + defense[a] + home_adv)
    mu = np.exp(attack[a] + defense[h])
    t = tau(x, y, lam, mu, rho)
    if np.any(t <= 0):
        return 1e12, np.zeros_like(theta)  # invalid rho region
    if w is None:
        w = np.ones_like(x)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    # d ll / d lam and d ll / d mu, including the tau correction
    dll_dlam = x / lam - 1.0
    dll_dlam[m00] += -mu[m00] * rho / t[m00]
    dll_dlam[m01] += rho / t[m01]
    dll_dmu = y / mu - 1.0
    dll_dmu[m00] += -lam[m00] * rho / t[m00]
    dll_dmu[m10] += rho / t[m10]
    # d ll / d rho comes only through tau
    dll_drho = np.zeros_like(t)
    dll_drho[m00] = -lam[m00] * mu[m00] / t[m00]
    dll_drho[m01] = lam[m01] / t[m01]
    dll_drho[m10] = mu[m10] / t[m10]
    dll_drho[m11] = -1.0 / t[m11]

    # d ll / d(log lam) = lam * d ll/d lam -- these weight each team's
    # contribution to the gradient by how much its strength moved the mean.
    # Time-decay weights scale each match's contribution to the data fit.
    w_lam = dll_dlam * lam * w
    w_mu = dll_dmu * mu * w
    w_rho = (dll_drho * w).sum()

    # Full-team gradients: attack[i] touches lam when i is home, mu when away.
    g_att = (np.bincount(h, weights=w_lam, minlength=n)
             + np.bincount(a, weights=w_mu, minlength=n))
    g_def = (np.bincount(a, weights=w_lam, minlength=n)
             + np.bincount(h, weights=w_mu, minlength=n))
    last = n - 1
    grad_ll = np.concatenate([
        g_att[:last] - g_att[last],      # free attack params
        g_def[:last] - g_def[last],      # free defense params
        [w_lam.sum()],                   # home advantage
        [w_rho],                         # rho
    ])

    nll = -(w * (np.log(t) + poisson_logpmf(x, lam)
                 + poisson_logpmf(y, mu))).sum()

    # --- L2 shrinkage (Gaussian prior): penalise extreme strengths ---
    # The prior covers ALL n teams, including the last team whose strength
    # is derived as -sum of the free ones. For a free attack parameter
    # theta_j: d/d theta_j of
    #   (1/2 sigma^2) * [sum_{j<last} theta_j^2 + (sum_{j<last} theta_j)^2]
    # is (theta_j - attack_last)/sigma^2. Same for defense.
    nll_prior, grad_prior = 0.0, np.zeros_like(theta)
    if prior_prec > 0.0:
        n_free = n - 1
        nll_prior = 0.5 * prior_prec * (
            (attack ** 2).sum() + (defense ** 2).sum())
        grad_prior[:n_free] = prior_prec * (attack[:last] - attack[last])
        grad_prior[n_free:2 * n_free] = prior_prec * (
            defense[:last] - defense[last])

    # Objective minimised:  nll_total = -loglik + prior_nll, so
    #   d(nll_total)/d theta = -d(loglik)/d theta + d(prior_nll)/d theta
    #                        = -grad_ll + grad_prior.
    # (Note the signs: grad_ll is the log-LIKELIHOOD gradient, grad_prior is
    # already the gradient of the prior NLL. Getting this wrong turns the
    # prior into anti-regularisation -- verified against finite differences.)
    return nll + nll_prior, grad_prior - grad_ll


def fit(df, prior_sigma=None, half_life_days=None, verbose=False):
    """Fit a DixonColesModel by maximum likelihood on a matches DataFrame.

    Expects columns: home_team, away_team, home_goals, away_goals.
    Optional column 'date' is required when half_life_days is given.

    Args:
        prior_sigma: std of the Gaussian prior on team strengths (L2
            shrinkage toward league average). None = pure MLE (v1).
        half_life_days: matches lose half their likelihood weight every
            this many days back from the most recent training game.
            None = all matches weighted equally (v1).
    Returns a DixonColesModel.
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    h = df["home_team"].map(idx).to_numpy()
    a = df["away_team"].map(idx).to_numpy()
    x = df["home_goals"].to_numpy(dtype=float)
    y = df["away_goals"].to_numpy(dtype=float)

    prior_prec = 0.0 if prior_sigma is None else 1.0 / prior_sigma ** 2

    w = None
    if half_life_days is not None:
        # Exponential decay anchored at the most recent training game:
        # a game `d` days old gets weight 0.5 ** (d / half_life_days).
        days_ago = (df["date"].max() - df["date"]).dt.days.to_numpy(
            dtype=float)
        w = 0.5 ** (days_ago / half_life_days)

    # Start from "everyone is average": attacks/defenses 0, mild home edge.
    theta0 = np.zeros(2 * (n - 1) + 2)
    theta0[-2] = 0.25
    # rho must stay in a range where tau() stays positive for realistic
    # scoring rates; the penalty inside _nll_and_grad guards the boundary.
    bounds = [(None, None)] * (2 * (n - 1) + 1) + [(-0.5, 0.5)]

    res = minimize(_nll_and_grad, theta0, args=(h, a, x, y, n, w, prior_prec),
                   method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": 5000})
    if not res.success:
        raise RuntimeError(f"MLE did not converge: {res.message}")

    attack, defense, home_adv, rho = _unpack_params(res.x, n)
    if verbose:
        print(f"converged in {res.nit} iterations, "
              f"neg-loglik={res.fun:.1f}, home_adv={home_adv:.3f}, "
              f"rho={rho:.4f}")
    return DixonColesModel(teams, attack, defense, home_adv, rho)
