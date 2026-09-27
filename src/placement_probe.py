"""Step 2e: last-mile placement probe - R2 AND ST-RAE vs the spread factor f.

sec 14 read: neighbors at our rho carry R2/rho^2 = 1.02-1.15, we are 0.988;
the R2 decomposition R2 = 2*rho*k*c - c^2 - d^2 (c = placed sd / sd_blind,
d = mean offset in sd units) says R2 is maximized at c = rho. Our shipped
F_SPREAD shrink was tuned for ST-RAE only. This script sweeps the GLOBAL
multiplier on F_SPREAD (0.8..1.6) and reports simulated blind MACRO R2 and
MA-ST-RAE of the placement, so any spread bump is judged on both scored
components at once (they fight on 2D6 - briford flags the same effect).

Model (same as shrink_placement.py): truth y ~ train labeled distribution with
its DRC bands; predictor z with corr(z,y)=rho_oof*k, k=1.05 (the transfer
ratio, sec 13); placement p = mu_t + f * F_SPREAD * rho_a * sd_t * z_hat with
rho_a = rho_oof * OOF_TO_BLIND (what regression_final.py ships).

Usage (cyp env): python src/placement_probe.py [rho_macro]
Prints per-f macro R2 + ST-RAE for k in {0.95, 1.05} (blind-corr uncertainty).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
import run_regression as R  # noqa: E402

sys.path.insert(0, os.path.join(HERE, ".."))
from src.regression_final import BLIND_MOMENTS, F_SPREAD, OOF_TO_BLIND, STRAE_MOMENTS  # noqa: E402

NSIM = 4000
RNG = np.random.default_rng(7)

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()


def st_rae_pop(y, lo, hi, p):
    err = np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)
    base = np.clip(np.mean(y) - hi, 0, None) + np.clip(lo - np.mean(y), 0, None)
    return err.sum() / base.sum()


def moments(iso):
    if iso in STRAE_MOMENTS:
        return STRAE_MOMENTS[iso]
    return BLIND_MOMENTS[iso]


# blend per-iso Pearson: default = cp7 nested-honest; override via argv
rho_file = sys.argv[1] if len(sys.argv) > 1 else None
if rho_file:
    import json
    rho_oof = json.load(open(rho_file))
else:
    rho_oof = {"CYP1A2": 0.598, "CYP2C9": 0.715, "CYP2D6": 0.466, "CYP3A4": 0.818}
    print("using default cp7-honest rho:", rho_oof)

fgrid = np.arange(0.8, 1.65, 0.05).round(2)
print(f"{'f':>5s} | {'k=0.95 R2  ST-RAE':>20s} | {'k=1.05 R2  ST-RAE':>20s}")
rows = {}
for f in fgrid:
    line = [f"{f:5.2f}"]
    for k in (0.95, 1.05):
        r2s, strs = [], []
        for iso in R.ISOFORMS:
            y = targets[R.DIRECT[iso]].values
            lab = ~np.isnan(y)
            yl = y[lab]
            lo0 = bounds[f"{iso}_lo"].values[lab]
            hi0 = bounds[f"{iso}_hi"].values[lab]
            mu_b, sd_b = BLIND_MOMENTS[iso]
            mu_t, sd_t = moments(iso)
            rho_a = float(np.clip(rho_oof[iso] * OOF_TO_BLIND[iso], 0.05, 0.95))
            rho = min(0.99, rho_oof[iso] * k)
            nrep = max(1, NSIM // lab.sum())
            ys, los, his = np.tile(yl, nrep), np.tile(lo0, nrep), np.tile(hi0, nrep)
            eps = RNG.standard_normal(len(ys))
            z = rho * (ys - ys.mean()) / ys.std() + np.sqrt(max(0.0, 1 - rho * rho)) * eps
            # ST-RAE side: placed at target moments like production (2D6 STRAE moments)
            p = mu_t + f * F_SPREAD[iso] * rho_a * sd_t * z
            p = np.clip(p, 1.0, None)
            strs.append(st_rae_pop(ys, los, his, p))
            # R2 side: scored against the true blind population (mean/sd moments):
            # standardize z-space, then R2 = 1 - E[(y - p_scaled)^2]/var(y) where
            # p_scaled maps p onto blind moments as production would have scored:
            # production places ONTO (mu_t, sd_t*rho_a*f); R2 uses the true blind
            # distribution as truth -> simulate y_blind ~ N(mu_b, sd_b)-consistent:
            # keep the same rank structure: y_bl = mu_b + sd_b * Phi_inv-like of z...
            # Simpler exact form: R2 = 2*rho*c - c^2 - d^2 with
            # c = f*F_SPREAD*rho_a*sd_t/sd_b, d = (mu_t - mu_b)/sd_b (2D6 only).
            c = f * F_SPREAD[iso] * rho_a * sd_t / sd_b
            d = (mu_t - mu_b) / sd_b
            r2 = 2 * rho * c - c * c - d * d
            r2s.append(r2)
        line.append(f"{np.mean(r2s):10.4f} {np.mean(strs):9.4f}")
        rows[(f, k)] = (float(np.mean(r2s)), float(np.mean(strs)))
    print(" |".join(line))

for k in (0.95, 1.05):
    bestr2 = max(rows, key=lambda x: rows[x][0] if x[1] == k else -9)
    bestsr = min(rows, key=lambda x: rows[x][1] if x[1] == k else 9)
    print(f"k={k}: R2 best f={bestr2[0]} ({rows[bestr2][0]:.4f}; at f=1.0 {rows.get((1.0,k),('?',0))[0]:.4f}) | "
          f"ST-RAE best f={bestsr[0]} ({rows[bestsr][1]:.4f}; at f=1.0 {rows.get((1.0,k),('?',0))[1]:.4f})")
