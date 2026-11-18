"""robust.py - minimax-regret solvers (Algorithm 2, Prop. 3) and experiments U1-U4."""
import itertools, numpy as np
from model import *

def regret_table(F, designs, scen, relative=True):
    """designs: dict name -> x ; scen: list of theta. Returns dict name -> (max regret, argmax theta idx)."""
    Ls = np.array([F.optimum(t)[0] for t in scen])
    out = {}
    for n, x in designs.items():
        L = np.array([loss_x(x, t) for t in scen]); r = L - Ls
        if relative: r = r / np.maximum(Ls, 1e-9)
        out[n] = (r.max(), int(r.argmax()))
    return out

def robust_frontier(F, scen, relative=True):
    """Algorithm 2: minimax regret over spend s on the frontier (scenario-defined candidates)."""
    cand = np.unique(np.concatenate([F.candidates(t) for t in scen]))
    Ls = np.array([F.optimum(t)[0] for t in scen])
    M = np.array([F.losses(t, cand) for t in scen])           # scen x cand
    R = M - Ls[:, None]
    if relative: R = R / np.maximum(Ls[:, None], 1e-9)
    i = int(np.argmin(R.max(0))); s = cand[i]
    return F.x_at(s), s, R.max(0)[i]

def nominal_design(F, th):
    s = F.optimum(th)[1]; return F.x_at(s)

def equal_split(B=10.0): return np.ones(3) * B / 3

def lattice(step, B=10.0):
    g = np.arange(0, B + 1e-9, step)
    X = np.array(list(itertools.product(g, g, g))); return X[X.sum(1) <= B + 1e-9]

def lattice_minimax(scen, tech_list=None, step=0.2, relative=True):
    """3-D lattice minimax regret; tech_list lets defense parameters vary per scenario (U4)."""
    X = lattice(step); s = X.sum(1)
    cols, Ls = [], []
    for k, t in enumerate(scen):
        tech = TECH if tech_list is None else tech_list[k]
        L = loss_rho(s, rho(X, tech), t); cols.append(L); Ls.append(L.min())
    R = np.array(cols) - np.array(Ls)[:, None]
    if relative: R = R / np.maximum(np.array(Ls)[:, None], 1e-9)
    i = int(R.max(0).argmin()); return X[i], R.max(0)[i], R, np.array(Ls)

if __name__ == "__main__":
    F = Frontier(); th0 = theta(10, 0.0)
    pct = lambda v: f"{100*v:6.1f}%"

    print("== H1: thresholds (base) ==")
    th = theta(10, 1.0); T = thresholds(F, th)
    print({k: round(float(v), 4) for k, v in T.items()})
    cs = np.linspace(0.9, 1.4, 501)
    sw = None
    # numeric switch: smallest c0 at which deterrence (L = eta*s_det) is optimal
    prev = None
    for c in cs:
        t = theta(10, c); L, s = F.optimum(t)
        det = not attacks(F.rho_at(s), t)
        if det: sw = c; break
    print("numerical switch ~", round(sw, 4))

    print("\n== Table III variants ==")
    for B, eta, Lam in [(10, .5, 10), (20, .5, 10), (20, 1.2, 10), (10, 1.2, 10), (20, .5, 20)]:
        Fb = Frontier(B=B); t = theta(Lam, 0.0, eta=eta, B=B); T = thresholds(Fb, t)
        num = None
        for c in np.arange(0.05, 3, 0.005):
            tc = theta(Lam, c, eta=eta, B=B); L, s = Fb.optimum(tc)
            if not attacks(Fb.rho_at(s), tc): num = c; break
        print(f"B={B} eta={eta} Lam={Lam}: feas={T['c_feas']:.3f} cost={T['c_cost']:.3f} c*={T['c_star']:.3f} numeric={num}")

    print("\n== U1: Lambda in [5,20], c0=0 ==")
    scen = [theta(L, 0.0) for L in np.logspace(np.log10(5), np.log10(20), 41)]
    xr, sr, _ = robust_frontier(F, scen)
    des = {"Nominal(L=10)": nominal_design(F, theta(10, 0)), "Low(L=5)": nominal_design(F, theta(5, 0)),
           "High(L=20)": nominal_design(F, theta(20, 0)), "Equal split": equal_split(), "Robust MMR": xr}
    for n, (m, _) in regret_table(F, des, scen).items():
        print(f"{n:15s} maxregret {pct(m)} spend {des[n].sum():5.2f} nominal loss {loss_x(des[n], th0):.2f}")

    print("\n== U2: c0 in [1,2], Lambda=10 ==")
    scen = [theta(10, c) for c in np.linspace(1, 2, 41)]
    xr, sr, _ = robust_frontier(F, scen)
    des = {f"Nominal c0={c}": nominal_design(F, theta(10, c)) for c in (1.2, 1.5, 2.0)}
    des["Ignore c0 (tolerate)"] = nominal_design(F, theta(10, 0.0))
    des["Equal split"] = equal_split(); des["Robust MMR"] = xr
    for n, (m, i) in regret_table(F, des, scen).items():
        print(f"{n:22s} maxregret {pct(m)} at c0={scen[i]['c0']:.2f}")

    print("\n== U3: Lambda x c0 (absolute regret) + 3-D lattice check ==")
    scen = [theta(L, c) for L in np.logspace(np.log10(5), np.log10(20), 7) for c in np.linspace(.5, 3, 11)]
    xr, sr, _ = robust_frontier(F, scen, relative=False)
    des = {"Nominal": nominal_design(F, theta(10, 1.5)), "Equal split": equal_split(), "Robust MMR": xr}
    for n, (m, _) in regret_table(F, des, scen, relative=False).items():
        print(f"{n:12s} worst abs regret {m:.2f}")
    _, v, _, _ = lattice_minimax(scen, step=0.25, relative=False)
    print("3-D lattice minimax (step .25):", round(v, 2))

    print("\n== U4: defense-effectiveness uncertainty (3-D lattice, step 0.2) ==")
    combos = [(b, p, L) for b in (.1, .3, .5) for p in (.75, .95) for L in (5, 10, 20)]
    techs = [dict(TECH, beta=b, pbar=p) for b, p, _ in combos]
    scen = [theta(L, 0.0) for _, _, L in combos]
    x_rob, v, R, Ls = lattice_minimax(scen, techs, step=0.2)
    X = lattice(0.2)
    def stats(x):
        L = np.array([loss_x(x, t, tc) for t, tc in zip(scen, techs)]); r = (L - Ls) / Ls; return r.max(), r.mean()
    xn = X[np.argmin(loss_rho(X.sum(1), rho(X, TECH), th0))]
    for n, x in {"Nominal": xn, "Equal split": equal_split(), "Robust": x_rob}.items():
        mx, mn = stats(x); print(f"{n:12s} max {pct(mx)} mean {pct(mn)}  x={np.round(x,2)}")
