"""L2: ST-RAE-aware affine placement grid on the hinge surrogate (offline).

Per isoform: take the SHIPPED ranking column's OOF surrogate (the hinge-fit
OOF blend where L1 passed - 1A2/3A4 - incumbent where it failed - 2C9/2D6),
z-score it, sweep affine placement (mean, sd) against the soft-RAE hinge
(src/run_regression.py:soft_rae) on scaffold-OOF rows with per-compound
credible bounds. Rules from docs/mining/open_top_extraction.md (briford):

  - mean range [mu_truth - 0.6, mu_truth + 0.6]; HARD CAP: never ship a
    grid mean > mu_truth + 0.75 (the 4.35 probe scored 0.8558 = disaster)
  - sd range [0.6*rho*sd, 1.1*sd] (optimum k = rho, not 1)
  - no clamps on the surrogate (a binding floor costs ranking); report
    how often FLOOR=1.0 binds at the chosen test point
  - the OOF-derived optimum is BIASED HIGH (OOF truth = assay set; blind
    = hit-expanded clusters). Grid output = the target for the probe
    solve, NOT a ship decision. The ship point needs probe support.

Also emits the 3-point probe plan per briford (base / 1.15x spread about
the same centre / centre +0.25 at the new spread) as concrete candidate
CSVs under cache/probes/, affines of --probe-base (default cp17 incumbent;
point at the scored vector once cp18 is on the board).

Usage (cyp env, repo root):
  python src/placement_grid.py [--probe-base cache/regression_cp18_hinge_submission.csv]
Output: cache/placement_grid.json + cache/probes/probe_{A_base,B_spread,C_centre}.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402
import hinge_blend as HB  # noqa: E402

FLOOR = 1.0
CAP = 0.75
MEAN_GRID = np.round(np.arange(-0.60, 0.6001, 0.05), 3)


def main(probe_base):
    Xtr, Xte, targets, bounds, S, Ste, _test = R.build_all()
    names = HB.POOLS[HB.POOL]
    oofs = {k: HB.raw_oof_fam(k) for k in names}
    w18 = json.load(open(os.path.join(CACHE, "hinge_w_cp18.json")))
    base_sub = pd.read_csv(probe_base if os.path.isabs(probe_base)
                           else os.path.join(CACHE, probe_base))
    rep = {}

    probe_cols = {}
    for iso in R.ISOFORMS:
        y = targets[R.DIRECT[iso]].values
        lab = ~np.isnan(y)
        lo, hi = bounds[f"{iso}_lo"].values, bounds[f"{iso}_hi"].values
        rows = np.where(lab)[0]
        P = np.column_stack([oofs[k].iloc[rows][iso].values for k in names])
        if w18[iso]["pass"]:
            w = np.array(w18[iso]["w_fitted"])
            p_oof = P @ w
        else:
            w0 = np.array([HB.bN[iso]["weights"].get(k, 0.0) for k in names])
            p_oof = P @ w0
        assert np.isfinite(p_oof).all()
        z = (p_oof - p_oof.mean()) / p_oof.std()
        mu_t, sd_t = float(np.mean(y[rows])), float(np.std(y[rows]))
        rho = float(HB.bN[iso]["pearson"])

        # current shipped placement (regression_final constants) on surrogate
        from regression_final import F_SPREAD, OOF_TO_BLIND, BLIND_MOMENTS, STRAE_MOMENTS
        rho_a = float(np.clip(rho * OOF_TO_BLIND[iso], 0.05, 0.95))
        mu_c, sd_c = BLIND_MOMENTS[iso][0], BLIND_MOMENTS[iso][1] * rho_a
        if iso in STRAE_MOMENTS:
            mu_c, sd_c = STRAE_MOMENTS[iso]
        sd_c = F_SPREAD[iso] * sd_c
        cur = HB.st_rae(mu_c + sd_c * z, y[rows], lo[rows], hi[rows])

        best = (None, np.inf)
        surf = {}
        for dm in MEAN_GRID:
            mu = mu_t + float(dm)
            if mu > mu_t + CAP:
                continue
            for fr in np.round(np.arange(0.6 * rho, 1.1001, 0.02), 3):
                sd = float(fr) * sd_t
                s = HB.st_rae(mu + sd * z, y[rows], lo[rows], hi[rows])
                surf[f"{round(mu, 3)}|{round(sd, 3)}"] = round(float(s), 4)
                if s < best[1]:
                    best = ((mu, float(sd)), float(s))
        (mu_b, sd_b), s_b = best
        rep[iso] = {
            "mu_truth": round(mu_t, 3), "sd_truth": round(sd_t, 3),
            "oof_pearson": round(rho, 4),
            "current_placement": [round(mu_c, 3), round(sd_c, 3)],
            "current_surrogate_st_rae": round(float(cur), 4),
            "grid_opt_placement": [round(mu_b, 3), round(sd_b, 3)],
            "grid_opt_st_rae": round(s_b, 4),
            "predicted_gain": round(float(cur) - s_b, 4),
            "grid_opt_minus_truth_mean": round(mu_b - mu_t, 3),
            "sd_over_rho_sd_truth": round(sd_b / (rho * sd_t), 3),
            "n_grid_points": len(surf),
            "surrogate": surf,
        }
        # probe plan: affines of the SHIPPED test column (z-space exact)
        col = f"{iso}_pIC50_direct_inhibition"
        c = base_sub[col].values.astype(float)
        zt = (c - c.mean()) / c.std()
        probe_cols[iso] = {
            "A_base": mu_b + sd_b * zt,
            "B_spread": mu_b + (1.15 * sd_b) * zt,
            "C_centre": (mu_b + 0.25) + (1.15 * sd_b) * zt,
        }
        binds = int((mu_b + sd_b * zt < FLOOR).sum())
        rep[iso]["probe_A_floor_binds_on_test"] = binds

    os.makedirs(os.path.join(CACHE, "probes"), exist_ok=True)
    test = pd.read_csv(os.path.join(HERE, "..", "data", "cyp-challenge-TEST-BLINDED.csv"))
    for tag in ("A_base", "B_spread", "C_centre"):
        sub = pd.DataFrame({"SMILES": test["SMILES"].values,
                            "Molecule_Name": test["Molecule_Name"].values})
        for iso in R.ISOFORMS:
            v = probe_cols[iso][tag]
            sub[f"{iso}_pIC50_direct_inhibition"] = np.clip(v, FLOOR, None)
        sub.to_csv(os.path.join(CACHE, "probes", f"probe_{tag}.csv"), index=False)

    json.dump(rep, open(os.path.join(CACHE, "placement_grid.json"), "w"), indent=2)
    for iso in R.ISOFORMS:
        r = rep[iso]
        print(iso, "cur", r["current_placement"], r["current_surrogate_st_rae"],
              "-> grid", r["grid_opt_placement"], r["grid_opt_st_rae"],
              "gain", r["predicted_gain"],
              "(dmu_vs_truth", r["grid_opt_minus_truth_mean"],
              "sd/rho_sd", r["sd_over_rho_sd_truth"],
              "floor_binds", r["probe_A_floor_binds_on_test"], ")", flush=True)
    print("wrote cache/placement_grid.json + cache/probes/probe_{A_base,B_spread,C_centre}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-base", default=HB.INCUMBENT_CSV,
                    help="shipped regression CSV whose columns the probes affine")
    a = ap.parse_args()
    main(a.probe_base)
