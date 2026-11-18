# Optimal Budget Allocation for DDoS Defence System: Stackelberg Approach

Code accompanying the paper by Ayush Tiwari, Gandharv Bhalekar, Milan Singh Bhadoria and Shubh Srivastava (SVNIT Surat).

A defender splits a limited budget among **monitoring**, **filtering** and **mitigation** against a DDoS attacker who has a convex running cost, a fixed launch cost and an intensity cap. The paper does two things:

1. **Deterrence structure.** Closed-form thresholds (affordability `c_feas` and cost-effectiveness `c_cost`) for when the defender should deter the attack rather than tolerate it, including the region where deterrence needs *over-investment* (Theorem 2).
2. **Robust design.** For any uncertainty in attacker parameters or defense price, a minimax-regret allocation lies on the efficient frontier, so the robust problem reduces to a 1-D search over total spend (Proposition 3).

> **Scope.** All parameters are synthetic and illustrative. Nothing here is validated against real traffic. Minimax regret in security games and deterrence by a defender effort level already exist in the literature, so the claimed novelty is at most incremental (see the paper's Sections III and XIII).

## Repository contents

| File | Purpose |
|---|---|
| `model.py` | Defense model (Eqs. 1–5), attacker best response (Prop. 1), efficient frontier ρ*(s), Stackelberg loss (Eq. 12), Theorem 2 thresholds |
| `robust.py` | Minimax-regret solvers (Algorithm 2 and the 3-D lattice search) and experiments H1, U1–U4. Run directly to print all results |
| `figs2.py` | Generates Figs. 1–3 as PNG files |

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
python robust.py    # threshold check, Tables III–V, U3 and U4 results
python figs2.py     # fig1_loss_spend_vs_c0.png, fig2_regret_vs_threat.png, fig3_regret_vs_c0.png
```

Building the frontier (401 spend points, each a lattice search plus SLSQP polish) takes the bulk of the run time. Both scripts rebuild it, so expect a short wait.

### Minimal example

```python
from model import *

F  = Frontier(B=10.0)             # efficient frontier rho*(s) on s in [0, 10]
th = theta(Lam=10, c0=1.3)        # threat level Lambda = V^2/(2 ca), launch cost c0

loss, spend = F.optimum(th)       # optimal loss and spend against this attacker
x = F.x_at(spend)                 # allocation (x_m, x_f, x_g)
print(thresholds(F, th))          # c_feas, c_cost, c_star, T, s_T, pi_T
```

## Model summary

| Quantity | Definition |
|---|---|
| Detection | `p(x_m) = p̄ (1 − exp(−κ_m x_m))` |
| Stop rates | `c_f = w_f (1 − exp(−κ_f x_f))`, `c_g = w_g (1 − exp(−κ_g x_g))` |
| Enhancement | `E = β + (1 − β) p(x_m)` |
| Residual traffic | `ρ(x) = (1 − E c_f)(1 − E c_g)` |
| Attacker payoff | `U_a = V a ρ − c_a a² − c_0·1[a>0]`, with `a ∈ [0, a_max]` |
| Defender loss | `L = η Σx_i + V a ρ·1[attack]`; ties resolve to abstention (SSE) |
| Threat level | `Λ = V² / (2 c_a)` |

Default parameters (from the paper): `V=10, c_a=5 (Λ=10), η=0.5, B=10, κ=(0.4, 0.3, 0.25), p̄=0.95, w_f=0.7, w_g=0.6, β=0.3, a_max=2`.

## Experiments

| ID | Setting | What it tests |
|---|---|---|
| H1 | Closed-form thresholds vs numerical switch points | Formulas match numerics (Table III) |
| U1 | Λ ∈ [5, 20], 41 log-spaced points, c₀ = 0 | Threat-underestimation cliff; robust vs nominal (Table IV, Fig. 2) |
| U2 | c₀ ∈ [1, 2], 41 points, Λ = 10 | Deterrence regret cliffs (Table V, Fig. 3) |
| U3 | Λ (7 points) × c₀ (11 points), absolute regret | Joint uncertainty; frontier solution vs 3-D lattice (H4) |
| U4 | β ∈ {0.1, 0.3, 0.5}, p̄ ∈ {0.75, 0.95}, Λ ∈ {5, 10, 20} | Defense-effectiveness uncertainty; no frontier dominance (H5) |

## Reproduced results

| Result | Paper | This code |
|---|---|---|
| Base c_feas | 1.11458 | 1.1146 |
| U1 max regret: nominal / low / high / equal / robust | 13.0 / 99.5 / 23.3 / 23.7 / 10.2 % | 13.0 / 99.4 / 23.3 / 23.7 / 10.2 % |
| U2 max regret: robust / equal split | 47.9 / 53.8 % | 48.3 / 54.1 % |
| U3 worst absolute regret: nominal / equal / robust | 5.74 / 5.00 / 4.66 | 5.74 / 5.00 / 4.67 |
| U3 3-D lattice check | 4.75 | 4.75 |
| U4 max regret: nominal / equal / robust | 24.8 / 35.9 / 19.0 % | 24.7 / 35.9 / 19.0 % |

## Notes and limitations

- **Reconstruction.** The original `model.py`, `robust.py` and `figs2.py` are not included in the paper PDF. These files were rebuilt from the paper's equations and stated parameters. Small differences (for example U2: 48.3 vs 47.9 %) likely come from grid and candidate-point choices.
- **Assumptions not stated in the paper.** U4 scenarios use `c₀ = 0`. Each scenario's deterrence-corner spend is added as an extra candidate point in the robust search.
- **Discretization.** Robust values are exact only on the discretized scenario sets. Thresholds match numerical switch points only to grid resolution.
- **Proposition 3 scope.** Frontier dominance covers attacker-side and price uncertainty only. For uncertainty in β, κ, w or p̄ the frontier itself changes, so U4 uses a 3-D lattice without a global-optimality proof.
- **Synthetic parameters.** Results show model behavior, not real DDoS performance.

## Citation

If you use this code, please cite the paper:

> A. Tiwari, G. Bhalekar, M. S. Bhadoria, S. Srivastava, "Optimal Budget Allocation for DDoS Defence System: Stackelberg Approach," Sardar Vallabhbhai National Institute of Technology, Surat.
