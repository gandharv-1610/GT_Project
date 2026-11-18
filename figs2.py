"""figs2.py - Figures 1-3."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import *
from robust import *

F = Frontier(); th0 = theta(10, 0.0)

# Fig 1: optimal loss and spend vs launch cost
cs = np.linspace(0.3, 2.5, 221); Lo, So = [], []
for c in cs:
    L, s = F.optimum(theta(10, c)); Lo.append(L); So.append(s)
T = thresholds(F, theta(10, 1.0))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for a, y, lab in zip(ax, (Lo, So), ("optimal loss", "optimal spend")):
    a.plot(cs, y, "k"); a.axvline(T["c_feas"], ls="--", c="C1"); a.axvline(T["piT"], ls="--", c="C0")
    a.set_xlabel("launch cost $c_0$"); a.set_ylabel(lab)
ax[0].axhline(T["T"], ls=":", c="gray"); ax[1].axhline(T["sT"], ls=":", c="gray")
plt.tight_layout(); plt.savefig("fig1_loss_spend_vs_c0.png", dpi=200); plt.close()

# Fig 2: regret vs true threat (U1)
scen = [theta(L, 0.0) for L in np.logspace(np.log10(5), np.log10(20), 41)]
Lams = np.array([V ** 2 / (2 * t["ca"]) for t in scen for V in [t["V"]]])
xr, _, _ = robust_frontier(F, scen)
des = {"Nominal": nominal_design(F, theta(10, 0)), "Low-estimate design (L=5)": nominal_design(F, theta(5, 0)),
       "Equal split": equal_split(), "Robust MMR": xr}
Ls = np.array([F.optimum(t)[0] for t in scen])
plt.figure(figsize=(5, 3.6))
for (n, x), ls in zip(des.items(), ("-", ":", "--", "-")):
    r = 100 * (np.array([loss_x(x, t) for t in scen]) - Ls) / Ls
    plt.plot(Lams, r, ls, label=n)
plt.xscale("log"); plt.ylim(0, 40); plt.xlabel("true threat $\\Lambda$"); plt.ylabel("regret (%)"); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig2_regret_vs_threat.png", dpi=200); plt.close()

# Fig 3: regret vs true launch cost (U2)
cgrid = np.linspace(1, 2, 201); sc = [theta(10, c) for c in cgrid]
xr, _, _ = robust_frontier(F, [theta(10, c) for c in np.linspace(1, 2, 41)])
Ls = np.array([F.optimum(t)[0] for t in sc]); plt.figure(figsize=(5, 3.6))
for n, x in {"Nominal c0=1.2": nominal_design(F, theta(10, 1.2)), "Nominal c0=1.5": nominal_design(F, theta(10, 1.5)),
             "Nominal c0=2.0": nominal_design(F, theta(10, 2.0)), "Robust MMR": xr}.items():
    plt.plot(cgrid, 100 * (np.array([loss_x(x, t) for t in sc]) - Ls) / Ls, label=n)
plt.xlabel("true launch cost $c_0$"); plt.ylabel("regret (%)"); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig3_regret_vs_c0.png", dpi=200); plt.close()
print("figures saved")
