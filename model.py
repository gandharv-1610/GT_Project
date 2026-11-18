"""model.py - DDoS Stackelberg budget-allocation model (Eqs. 1-15, Props. 1-2, Thm 2)."""
import numpy as np
from scipy.optimize import minimize

TECH = dict(kappa=(0.4, 0.3, 0.25), pbar=0.95, wf=0.7, wg=0.6, beta=0.3)
BASE = dict(V=10.0, ca=5.0, c0=0.0, amax=2.0, eta=0.5, B=10.0)   # Lambda = V^2/(2 ca) = 10


def theta(Lam=10.0, c0=0.0, **kw):
    """Attacker/price parameter set from threat level Lambda=V^2/(2 ca)."""
    th = dict(BASE); th.update(kw)
    th["ca"] = th["V"] ** 2 / (2 * Lam); th["c0"] = c0
    return th


# ---------- defense technology: Eqs. (1)-(5) ----------
def rho(x, tech=TECH):
    x = np.asarray(x, float)
    xm, xf, xg = x[..., 0], x[..., 1], x[..., 2]
    k = tech["kappa"]
    p = tech["pbar"] * (1 - np.exp(-k[0] * xm))
    E = tech["beta"] + (1 - tech["beta"]) * p
    cf = tech["wf"] * (1 - np.exp(-k[1] * xf))
    cg = tech["wg"] * (1 - np.exp(-k[2] * xg))
    return (1 - E * cf) * (1 - E * cg)


# ---------- attacker best response: Prop. 1 ----------
def a_c(r, th):  return np.minimum(th["amax"], th["V"] * r / (2 * th["ca"]))                    # Eq. 8
def profit(r, th):
    a = a_c(r, th); return th["V"] * r * a - th["ca"] * a ** 2                                   # Eq. 9
def damage(r, th): return th["V"] * r * a_c(r, th)
def rho_det(th):                                                                                 # Eq. 11
    Lam = th["V"] ** 2 / (2 * th["ca"]); c0 = th["c0"]
    if c0 <= th["ca"] * th["amax"] ** 2: return np.sqrt(2 * c0 / Lam)
    return (c0 + th["ca"] * th["amax"] ** 2) / (th["V"] * th["amax"])
def attacks(r, th):  return profit(r, th) > th["c0"] + 1e-12     # tie -> abstain (SSE)


def loss_rho(s, r, th):
    """Defender loss, Eq. (12), as function of spend s and residual fraction r."""
    return th["eta"] * s + damage(r, th) * attacks(r, th)


def loss_x(x, th, tech=TECH):
    x = np.asarray(x, float); return loss_rho(x.sum(-1), rho(x, tech), th)


# ---------- efficient frontier rho*(s) ----------
def _simplex(n):
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    m = i + j <= n; i, j = i[m], j[m]
    return np.stack([i, j, n - i - j], 1) / n


_SIMP = _simplex(60)


def frontier_point(s, tech=TECH, polish=True):
    """Best split of spend s: lattice search (n=60) + SLSQP polish."""
    if s <= 1e-12: return np.zeros(3), 1.0
    cand = _SIMP * s; r = rho(cand, tech); x0 = cand[np.argmin(r)]
    best_x, best_r = x0, r.min()
    if polish:
        res = minimize(lambda x: rho(x, tech), x0, method="SLSQP", bounds=[(0, s)] * 3,
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - s}])
        if res.success and res.fun < best_r: best_x, best_r = res.x, res.fun
    return best_x, best_r


class Frontier:
    def __init__(self, B=10.0, npts=401, tech=TECH):
        self.B, self.tech = B, tech
        self.s = np.linspace(0, B, npts)
        pts = [frontier_point(s, tech) for s in self.s]
        self.x = np.array([p[0] for p in pts])
        self.r = np.minimum.accumulate(np.array([p[1] for p in pts]))   # nonincreasing

    def rho_at(self, s): return np.interp(s, self.s, self.r)

    def s_det(self, rd):
        """Smallest spend with rho*(s) <= rd (inf if none)."""
        if self.r[-1] > rd: return np.inf
        if self.r[0] <= rd: return 0.0
        lo, hi = 0.0, self.B
        for _ in range(60):
            mid = (lo + hi) / 2
            if self.rho_at(mid) <= rd: hi = mid
            else: lo = mid
        return hi

    def x_at(self, s):
        return frontier_point(s, self.tech)[0]

    def losses(self, th, s=None):
        s = self.s if s is None else s
        return loss_rho(s, self.rho_at(s), th)

    def candidates(self, th):
        """Grid plus the deterrence corner for this theta."""
        sd = self.s_det(rho_det(th)) if th["c0"] > 0 else np.inf
        c = np.append(self.s, sd) if np.isfinite(sd) and sd <= self.B else self.s
        return np.sort(c)

    def optimum(self, th):
        c = self.candidates(th); L = self.losses(th, c); i = int(np.argmin(L))
        return L[i], c[i]


# ---------- Theorem 2 thresholds ----------
def thresholds(F, th):
    """Return (c_feas, c_cost, c_star, T, s_T, pi_T) for launch cost th['c0'] ignored."""
    t0 = dict(th); t0["c0"] = 0.0
    L = F.losses(t0); i = int(np.argmin(L)); T, sT = L[i], F.s[i]    # tolerate optimum
    c_feas = float(profit(F.r[-1], th))
    Teta = T / th["eta"]
    c_cost = float(profit(F.rho_at(Teta), th)) if Teta <= F.B else -np.inf
    return dict(c_feas=c_feas, c_cost=c_cost, c_star=max(c_feas, c_cost),
                T=T, sT=sT, piT=float(profit(F.rho_at(sT), th)))
