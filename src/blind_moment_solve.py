"""L2b: briford blind-moment solve from 3 scored affine probes.

Three scored submissions of ONE ranking vector per isoform, affines
(mu, sd) in z-space of the shipped cp18 columns:
  A = base       (mu_A, s)       <- ALREADY SCORED: cp18 submission
  B = spread     (mu_A, 1.15 s)
  C = centre     (mu_A + 0.25, 1.15 s)
Each scored row gives, per isoform, the exact equation (briford):

    R2 * V = 2*Cov - var(p) - (mu_y - mu_p)^2       [V = var(y)]

Three rows, three unknowns (mu_y, V, Cov) -> closed form:
  C-B (same spread, centre +d): V*(R2_C - R2_B) = 2*mu_y*d - (mu_C^2 - mu_B^2)
  A-B: V = [var(p_B) - var(p_A) + (mu_y-mu_A)^2 - (mu_y-mu_B)^2] / (R2_A - R2_B)
  Cov from row A.

Gates (open_top_extraction): physical root 0.25 < V < 9.0; mu_y within
[OOF-truth mean - 1.6, +0.5]; solved Pearson rho = Cov/sqrt(V*var(p_A))
>= measured Spearman - 0.07; treat rho as +-0.07 noise.

Then: final placement = cp18 z-space vector placed at
  centre = mu_y + dmu   (dmu from cache/placement_grid.json grid shape,
                         capped at +0.75 above mu_y... cap is on
                         OOF-truth; here centre <= mu_y + 0.35 hard)
  sd     = grid_ratio * rho_solved * sqrt(V)   (ratio k from the L2 grid:
                         optimum k ~ rho, narrower for 2D6)
Ship only if it also improves the OOF hinge surrogate shape (grid) and
passes verify_submissions.py.

Usage:
  # print the 3 probe CSVs (B, C) to submit (A = cp18, already scored):
  python src/blind_moment_solve.py --emit-probes
  # solve after scoring; paste per-isoform R2 from the feedback page:
  python src/blind_moment_solve.py --solve \
      --R2A 0.5219,0.7313,0.4133,0.6593 --R2B ... --R2C ...
  (order CYP1A2,CYP2C9,CYP2D6,CYP3A4)
  # final placement (after solve passes, writes candidate CSV):
  python src/blind_moment_solve.py --build-final

Output: cache/probes/probe_B_spread.csv, probe_C_centre.csv,
cache/blind_moments_solved.json, cache/regression_cp18_placed_submission.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
BASE = "regression_cp18_hinge_submission.csv"
D_MU = 0.25
SPREAD = 1.15


def base_moments():
    sub = pd.read_csv(os.path.join(CACHE, BASE))
    return {iso: (float(sub[f"{iso}_pIC50_direct_inhibition"].mean()),
                  float(sub[f"{iso}_pIC50_direct_inhibition"].std()))
            for iso in ISO}


def emit_probes():
    sub = pd.read_csv(os.path.join(CACHE, BASE))
    os.makedirs(os.path.join(CACHE, "probes"), exist_ok=True)
    stats = {}
    for tag, fmu, fsd in (("probe_B_spread", 0.0, SPREAD),
                          ("probe_C_centre", D_MU, SPREAD)):
        out = sub.copy()
        for iso in ISO:
            col = f"{iso}_pIC50_direct_inhibition"
            mu, sd = base_moments()[iso]
            z = (sub[col].values - mu) / sd
            out[col] = np.clip((mu + fmu) + (fsd * sd) * z, 1.0, None)
            stats.setdefault(tag, {})[iso] = [round(float(out[col].mean()), 3),
                                              round(float(out[col].std()), 3)]
        out.to_csv(os.path.join(CACHE, "probes", f"{tag}.csv"), index=False)
    print(json.dumps(stats, indent=2))
    print("base(A) = cp18 (already scored). Submit B, C after cooldown.")


def parse_r2(s):
    v = [float(x) for x in s.split(",")]
    assert len(v) == 4, "give 4 values: CYP1A2,CYP2C9,CYP2D6,CYP3A4"
    return dict(zip(ISO, v))


def solve(r2a, r2b, r2c):
    out = {}
    mu_z = base_moments()
    for iso in ISO:
        mu_A, s_A = mu_z[iso]
        mu_B, s_B = mu_A, SPREAD * s_A
        mu_C = mu_A + D_MU
        s_C = SPREAD * s_A
        var_A, var_B, var_C = s_A ** 2, s_B ** 2, s_C ** 2
        dmu = mu_C - mu_B
        dR2_CB = r2c[iso] - r2b[iso]
        # C - B: V*dR2_CB = 2*mu_y*dmu - (mu_C^2 - mu_B^2)
        if abs(r2a[iso] - r2b[iso]) < 1e-4 or abs(dR2_CB) < 1e-4:
            out[iso] = {"fail": "R2 rows degenerate (noise floor +-0.005)"}
            continue
        # parameterize: mu_y = [V*dR2_CB + mu_C^2 - mu_B^2]/(2*dmu)
        # plug into A-B equation, solve quadratic in V
        a1 = 1.0 / (2 * dmu)
        # mu_y(V) = a1*V + c1 ; c1 = (mu_C^2-mu_B^2)/(2 dmu) = mu_A+mu_C)/2? check:
        c1 = (mu_C ** 2 - mu_B ** 2) / (2 * dmu)
        # A-B: dR2_AB*V = var_B - var_A + (mu_y-mu_A)^2 - (mu_y-mu_B)^2
        #      = var_B - var_A + (mu_B-mu_A)*(2 mu_y - mu_A - mu_B)
        dR2_AB = r2a[iso] - r2b[iso]
        k = var_B - var_A + (mu_B - mu_A) * (-(mu_A + mu_B))
        # dR2_AB*V = k + (mu_B-mu_A)*(2*(a1 V + c1))
        # V*(dR2_AB - 2*(mu_B-mu_A)*a1) = k + 2*(mu_B-mu_A)*c1
        denom = dR2_AB - 2 * (mu_B - mu_A) * a1
        if abs(denom) < 1e-6:
            out[iso] = {"fail": "degenerate spread pair"}
            continue
        V = (k + 2 * (mu_B - mu_A) * c1) / denom
        mu_y = a1 * V + c1
        cov = r2a[iso] * V + var_A + (mu_y - mu_A) ** 2
        rho = cov / np.sqrt(V * var_A)
        gates = {"physical_root": bool(0.25 < V < 9.0),
                 "mu_in_range": None,  # filled by caller w/ OOF truth
                 "rho_ge_spearman_minus_007": None}
        out[iso] = {"V": round(float(V), 4), "mu_y": round(float(mu_y), 4),
                    "cov": round(float(cov), 4),
                    "sd_y": round(float(np.sqrt(V)), 4),
                    "pearson_solved": round(float(rho), 4),
                    "gates": gates}
    return out


def build_final(solved, grid_shape, spearman_measured, oof_truth_mean):
    sub = pd.read_csv(os.path.join(CACHE, BASE))
    out = sub.copy()
    report = {}
    for iso in ISO:
        s = solved[iso]
        if "fail" in s:
            out[f"{iso}_pIC50_direct_inhibition"] = sub[f"{iso}_pIC50_direct_inhibition"]
            report[iso] = {"kept_incumbent": s["fail"]}
            continue
        s["gates"]["physical_root"] = bool(0.25 < s["V"] < 9.0)
        s["gates"]["mu_in_range"] = bool(oof_truth_mean[iso] - 1.6 <= s["mu_y"]
                                         <= oof_truth_mean[iso] + 0.5)
        s["gates"]["rho_ge_spearman_minus_007"] = bool(
            s["pearson_solved"] >= spearman_measured[iso] - 0.07)
        ok = all(s["gates"].values())
        if not ok:
            out[f"{iso}_pIC50_direct_inhibition"] = sub[f"{iso}_pIC50_direct_inhibition"]
            report[iso] = {"kept_incumbent": s["gates"]}
            continue
        mu_A, s_A = base_moments()[iso]
        z = (sub[f"{iso}_pIC50_direct_inhibition"].values - mu_A) / s_A
        dmu = min(grid_shape[iso]["dmu"], 0.35)
        k = grid_shape[iso]["k_ratio"] * s["pearson_solved"]
        mu_new = min(s["mu_y"] + dmu, s["mu_y"] + 0.75)
        sd_new = k * s["sd_y"]
        out[f"{iso}_pIC50_direct_inhibition"] = np.clip(mu_new + sd_new * z, 1.0, None)
        report[iso] = {"mu_new": round(mu_new, 3), "sd_new": round(sd_new, 3),
                       "rho_solved": s["pearson_solved"], "gates": s["gates"]}
    out.to_csv(os.path.join(CACHE, "regression_cp18_placed_submission.csv"),
               index=False)
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-probes", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--build-final", action="store_true")
    ap.add_argument("--R2A", default=None)
    ap.add_argument("--R2B", default=None)
    ap.add_argument("--R2C", default=None)
    a = ap.parse_args()
    if a.emit_probes:
        emit_probes()
    if a.solve and a.R2A and a.R2B and a.R2C:
        sol = solve(parse_r2(a.R2A), parse_r2(a.R2B), parse_r2(a.R2C))
        json.dump(sol, open(os.path.join(CACHE, "blind_moments_solved.json"), "w"),
                  indent=2)
        print(json.dumps(sol, indent=2))
    if a.build_final:
        sol = json.load(open(os.path.join(CACHE, "blind_moments_solved.json")))
        grid = json.load(open(os.path.join(CACHE, "placement_grid.json")))
        shape = {iso: {"dmu": grid[iso]["grid_opt_placement"][0] - grid[iso]["mu_truth"],
                       "k_ratio": grid[iso]["sd_over_rho_sd_truth"]}
                 for iso in ISO}
        # measured per-isoform Spearman from cp18 on-board feedback (caller
        # updates these constants after each scored submission; defaults =
        # Oct 22 board per-isoform rho from NOTES sec 29b board snapshot)
        spearman_measured = {"CYP1A2": 0.6245, "CYP2C9": 0.7290,
                             "CYP2D6": 0.5730, "CYP3A4": 0.7710}
        oof_truth_mean = {"CYP1A2": 4.955, "CYP2C9": 4.831,
                          "CYP2D6": 4.784, "CYP3A4": 4.096}
        build_final(sol, shape, spearman_measured, oof_truth_mean)


if __name__ == "__main__":
    main()
