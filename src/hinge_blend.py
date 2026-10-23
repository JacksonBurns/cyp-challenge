"""L1 + L4: ST-RAE hinge-fit combiner over the cp17 pool (per isoform).

L1 (plan docs/IMPROVEMENT_PLAN.md sec 2, Scigantic recipe): per isoform,
Powell-fit member-delta weights w = w_incumbent + d minimizing

    sum_i hinge(p_i, lo_i, hi_i) + LAMBDA * ||d||^2

over scaffold-OOF rows (raw pIC50-scale member preds; hinge =
clip(p-hi,0)+clip(lo-p,0) per src/run_regression.py:soft_rae, mean-centred
normalisation).  Bounds come from build_all() -> {iso}_lo/_hi (verified:
finite on every labeled row).  Raw preds, NOT z-scored (z-scoring destroys
the affine information the hinge metric scores - briford lesson).

Gate (per isoform, 2D6 strictest): rotation CV - fit on 4/5 folds of labeled
rows, score ST-RAE on the held-out fold; positive gain in >=4/5 folds AND
mean gain > 0.004.  Report every fold (per-fold means; quarantine on any
fold >0.085 off pooled).

L4 (--l4): confidence-weighted, non-negative delta variant.  Confidence
proxy = per-compound disagreement std across incumbent members' z-aligned
OOF preds (1/std weights, damped).  L-BFGS-B with d >= 0.  Gate: paired
bootstrap (500 resamples, shared rows) 95% CI on hinge-sum delta excludes 0
AND direction agrees with L1.

Shipped column build (only if gate passes, --emit): test preds p = sum w_k
raw_test_k, then moment-restore affine to the incumbent placed column's
mean/sd from cache/regression_final_cp17_submission.csv, so ONLY the
reordering ships.  Spearman(shipped, fitted) is 1 by construction (affine
integrity); Spearman vs incumbent recorded.

Usage (cyp env, repo root):
  python src/hinge_blend.py                 # L1 fit + rotation CV gate
  python src/hinge_blend.py --l4            # L1 + L4 + bootstrap gate
  python src/hinge_blend.py --emit          # build cp18 z-blend test preds
Output: cache/hinge_blend_l1.json (report), cache/hinge_w_cp18.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402

# Family/pool definitions duplicated from src/blendN_all.py (that module
# runs a blend at import time and cannot be imported). Keep in sync.
def g(tag):
    return sorted(f for f in glob.glob(os.path.join(CACHE, f"ft_oof_{tag}*.csv"))
                  if "smoke" not in os.path.basename(f))


FAMS = {
    "gbm": [os.path.join(CACHE, "oof_blend.csv")],
    "emb": [os.path.join(CACHE, "oof_emb_concat.csv")],
    "ft_frozen": [os.path.join(CACHE, "ft_oof_frozen.csv")],
    "ft_fullft": g("fullft"),
    "ft_ext": g("ext"),
    "ft_pre": g("pre"),
    "ft_dmpnn": g("dmpnn"),
    "ft_cpmed": g("cpmed"),
    "ft_cpchm": g("cpchm"),
    "ft_chmridge": g("chmridge"),
    "ft_d2d6": g("d2d6"),
    "ft_near": g("near"),
    "ft_admchm": g("admchm"),
    "ft_admmed": g("admmed"),
    "ft_unimol": g("unimol"),
    "ft_tabpfn_cpmed": g("tabpfn_cpmed"),
    "ft_intarm": g("intarm"),
}
for _l in ("L1", "L2", "L3", "L4"):
    FAMS[f"ft_umtm{_l}"] = g(f"umtm{_l}")
POOLS = {
    "cp17": ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre",
             "ft_dmpnn", "ft_cpmed", "ft_cpchm", "ft_chmridge", "ft_d2d6",
             "ft_admchm", "ft_admmed", "ft_unimol", "ft_umtmL2", "ft_umtmL1",
             "ft_umtmL3", "ft_umtmL4", "ft_tabpfn_cpmed"],
    "cp18i": [n for n in
              ["gbm", "emb", "ft_frozen", "ft_fullft", "ft_ext", "ft_pre",
               "ft_dmpnn", "ft_cpmed", "ft_cpchm", "ft_chmridge", "ft_d2d6",
               "ft_admchm", "ft_admmed", "ft_unimol", "ft_umtmL2", "ft_umtmL1",
               "ft_umtmL3", "ft_umtmL4", "ft_tabpfn_cpmed", "ft_intarm"]
              ],
}

ISO = R.ISOFORMS
POOL = "cp17"
BLENDF = "blendN_cp17.json"
INCUMBENT_CSV = "regression_final_cp17_submission.csv"
LAMBDAS = [0.03, 0.1, 0.3]
MAXEV = 4000
Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
FOLDS = np.load(os.path.join(CACHE, "folds.npy"))
bN = json.load(open(os.path.join(CACHE, BLENDF)))


def raw_oof_fam(name):
    """Raw-scale family OOF: mean across seed files (raw pIC50 preds).

    Files with a SMILES column are aligned by SMILES; gbm/emb OOF files
    (oof_blend.csv, oof_emb_concat.csv) are positional to Xtr order, same
    convention as blendN_all.zavg.
    """
    files = [f if os.path.isabs(f) else os.path.join(CACHE, f) for f in FAMS[name]]
    smi_index = Xtr["SMILES"].values
    out = pd.DataFrame(index=smi_index, columns=ISO, dtype=float)
    acc = np.zeros((len(smi_index), len(ISO)))
    cnt = np.zeros((len(smi_index), len(ISO)))
    for f in files:
        d = pd.read_csv(f)
        if "SMILES" in d.columns:
            v = d.set_index("SMILES")[ISO].reindex(smi_index).values
        else:
            assert len(d) == len(smi_index), f"{f}: {len(d)} rows, positional?"
            v = d[ISO].values
        ok = np.isfinite(v)
        acc = np.where(ok, np.nan_to_num(v) + acc, acc)
        cnt += ok
    out[:] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return out


def raw_test_matrix(name, test_smiles):
    """Full (n_test, 4) raw test preds for one family."""
    if name == "gbm":
        blend = np.load(os.path.join(CACHE, "blend_test_preds.npz"))
        return np.column_stack([blend[f"blend_{i}"] for i in ISO])
    if name == "emb":
        blend = np.load(os.path.join(CACHE, "blend_test_preds.npz"))
        return np.column_stack([blend[f"concat_{i}"] for i in ISO])
    tag = name[3:]
    files = sorted(f for f in glob.glob(os.path.join(CACHE, f"ft_test_preds_{tag}*.csv"))
                   if "smoke" not in os.path.basename(f))
    assert files, f"no test preds for {tag}"
    acc = np.full((len(test_smiles), len(ISO)), np.nan)
    cnt = np.zeros((len(test_smiles), len(ISO)))
    for f in files:
        p = pd.read_csv(f).set_index("SMILES")
        v = p[[c for c in ISO if c in p.columns]].reindex(test_smiles).values
        ok = np.isfinite(v)
        acc = np.where(ok, np.nan_to_num(v) + np.nan_to_num(acc), acc)
        cnt += ok
    out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    assert not np.isnan(out).any(), f"{tag}: NaN test preds"
    return out


def hinge_sum(p, lo, hi):
    return float(np.clip(p - hi, 0, None).sum() + np.clip(lo - p, 0, None).sum())


def st_rae(p, y, lo, hi):
    base = np.clip(np.mean(y) - hi, 0, None) + np.clip(lo - np.mean(y), 0, None)
    return hinge_sum(p, lo, hi) / base.sum()


def obj(w, P, lo, hi, w0, lam):
    p = P @ w
    return hinge_sum(p, lo, hi) + lam * float(np.sum((w - w0) ** 2))


def fit(P, lo, hi, w0, lam, nonneg=False):
    if nonneg:
        res = minimize(obj, w0 + np.ones(len(w0)) * 0.02, args=(P, lo, hi, w0, lam),
                       method="L-BFGS-B", bounds=[(0.0, None)] * len(w0),
                       options={"maxiter": 2000})
        return res.x, float(res.fun)
    res = minimize(obj, w0, args=(P, lo, hi, w0, lam), method="Powell",
                   options={"maxiter": MAXEV, "xtol": 1e-6, "ftol": 1e-8})
    return res.x, float(res.fun)


def main(emit, do_l4, pool=POOL, wout='hinge_w_cp18.json', report_name='hinge_blend_l1.json',
         incumbent_csv=INCUMBENT_CSV, out_csv="regression_cp18_hinge_submission.csv",
         ship_list=""):
    names = [n for n in POOLS[pool] if FAMS.get(n)]
    oofs = {k: raw_oof_fam(k) for k in names}
    inc = pd.read_csv(os.path.join(CACHE, incumbent_csv))
    global WOUT, REPORT
    WOUT, REPORT = wout, report_name
    ship_only = set(ship_list.split(",")) if ship_list else None
    report = {}
    new_w = {}
    for iso in ISO:
        y = targets[R.DIRECT[iso]].values
        lab = ~np.isnan(y)
        lo, hi = bounds[f"{iso}_lo"].values, bounds[f"{iso}_hi"].values
        rows = np.where(lab)[0]
        P = np.column_stack([oofs[k].iloc[rows][iso].values for k in names])
        w0 = np.array([bN[iso]["weights"].get(k, 0.0) for k in names])
        lo_r, hi_r, y_r = lo[rows], hi[rows], y[rows]
        base_w0 = st_rae(P @ w0, y_r, lo_r, hi_r)

        # ---- L1: Powell hinge fit, lambda sweep on rotation CV ----
        fold_of = FOLDS[rows]
        best_lam, best_mean, best_folds = None, -9, None
        per_lam = {}
        for lam in LAMBDAS:
            gains = []
            for f in range(5):
                tr = fold_of != f
                va = fold_of == f
                w, _ = fit(P[tr], lo_r[tr], hi_r[tr], w0, lam)
                g = (st_rae(P[tr] @ w0, y_r[tr], lo_r[tr], hi_r[tr])
                     - st_rae(P[tr] @ w, y_r[tr], lo_r[tr], hi_r[tr]))
                gv = (st_rae(P[va] @ w0, y_r[va], lo_r[va], hi_r[va])
                      - st_rae(P[va] @ w, y_r[va], lo_r[va], hi_r[va]))
                gains.append(gv)
            mean_g = float(np.mean(gains))
            per_lam[lam] = {"folds": [round(g, 4) for g in gains], "mean": round(mean_g, 4),
                            "pos_folds": int(sum(g > 0 for g in gains))}
            if mean_g > best_mean:
                best_lam, best_mean, best_folds = lam, mean_g, gains
        w1, f1 = fit(P, lo_r, hi_r, w0, best_lam)
        insample = st_rae(P @ w1, y_r, lo_r, hi_r)
        n_pos = int(sum(g > 0 for g in best_folds))
        l1_pass = n_pos >= 4 and best_mean > 0.004
        new_w[iso] = {"w1": w1, "lam": best_lam,
                      "pass": bool(l1_pass and (ship_only is None
                                                or iso in ship_only))}

        report[iso] = {
            "n_rows": int(len(rows)),
            "incumbent_oof_st_rae": round(base_w0, 4),
            "fitted_oof_st_rae": round(insample, 4),
            "lam": best_lam,
            "rotation_cv": per_lam,
            "cv_mean_gain": round(best_mean, 4),
            "cv_pos_folds": n_pos,
            "gate_pass_L1": bool(l1_pass),
            "spearman_w_vs_incumbent_blend_oof": round(float(spearmanr(
                P @ w1, P @ w0).statistic), 4),
            "weights_fitted_top": {k: round(float(v), 3) for k, v in
                                   sorted(zip(names, w1), key=lambda t: -abs(t[1]))
                                   if abs(v) > 0.01},
        }

        # ---- L4: confidence-weighted non-negative deltas ----
        if do_l4:
            Zs = (P - P.mean(0)) / P.std(0)
            act = np.abs(w0) > 1e-6
            sd = Zs[:, act].std(1)
            conf = 1.0 / np.maximum(sd, np.quantile(sd, 0.1))
            cw = conf / conf.mean()
            d0 = np.where(w0 > 0, 0.02, 0.0)
            res = minimize(lambda d: obj(w0 + d, P, lo_r, hi_r, w0, best_lam),
                           d0, method="L-BFGS-B",
                           bounds=[(0.0, None)] * len(w0),
                           options={"maxiter": 2000})
            # confidence-weighted: reweight rows via sqrt scaling in obj is not
            # separable with hinge; use weights as multipliers on row terms
            def objw(d):
                w = w0 + d
                p = P @ w
                h = np.clip(p - hi_r, 0, None) + np.clip(lo_r - p, 0, None)
                return float((cw * h).sum()) + best_lam * float(np.sum(d ** 2))
            resw = minimize(objw, d0, method="L-BFGS-B",
                            bounds=[(0.0, None)] * len(w0),
                            options={"maxiter": 2000})
            w4 = w0 + resw.x
            delta_oof = st_rae(P @ w4, y_r, lo_r, hi_r)
            # paired bootstrap on shared rows
            rng = np.random.default_rng(0)
            h0 = np.clip(P @ w0 - hi_r, 0, None) + np.clip(lo_r - P @ w0, 0, None)
            h4 = np.clip(P @ w4 - hi_r, 0, None) + np.clip(lo_r - P @ w4, 0, None)
            hd = h0 - h4
            boots = []
            for _ in range(500):
                idx = rng.integers(0, len(rows), len(rows))
                base_b = np.clip(np.mean(y_r[idx]) - hi_r[idx], 0, None) \
                    + np.clip(lo_r[idx] - np.mean(y_r[idx]), 0, None)
                boots.append(hd[idx].sum() / base_b[idx].sum())
            lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
            report[iso]["L4"] = {
                "st_rae_L4": round(delta_oof, 4),
                "delta_vs_incumbent": round(base_w0 - delta_oof, 4),
                "bootstrap_mean": round(float(np.mean(boots)), 4),
                "ci95": [round(float(lo_ci), 4), round(float(hi_ci), 4)],
                "ci_excludes_0": bool(lo_ci > 0 or hi_ci < 0),
                "direction_agrees_L1": bool((base_w0 - delta_oof > 0) and (best_mean > 0)),
            }
        report[iso]["w_fitted"] = [round(float(v), 4) for v in w1]

    json.dump(report, open(os.path.join(CACHE, REPORT), "w"),
              indent=2, default=float)
    for iso in ISO:
        print(iso, json.dumps({k: v for k, v in report[iso].items()
                               if k not in ("rotation_cv", "w_fitted")}), flush=True)
        print("   cv:", report[iso]["rotation_cv"][report[iso]["lam"]], flush=True)

    # save fitted weights for emit/step 3 (cp18 build), keyed by iso
    wsave = {iso: {"lam": new_w[iso]["lam"], "pass": new_w[iso]["pass"],
                   "w": {k: float(v) for k, v in zip(names, new_w[iso]["w1"])},
                   "w_fitted": [float(v) for v in new_w[iso]["w1"]],
                   "members": names}
             for iso in ISO}
    json.dump(wsave, open(os.path.join(CACHE, WOUT), "w"), indent=2)

    if emit:
        test_smiles = test["SMILES"].values
        out = pd.DataFrame({"SMILES": test_smiles,
                            "Molecule_Name": test["Molecule_Name"].values})
        for iso in ISO:
            if not new_w[iso]["pass"]:
                print(f"{iso} gate FAILED - keeping incumbent column", flush=True)
                out[f"{iso}_pIC50_direct_inhibition"] = inc[f"{iso}_pIC50_direct_inhibition"].values
                continue
            M = np.column_stack([raw_test_matrix(k, test_smiles)[:, ISO.index(iso)]
                                 for k in names])
            w = new_w[iso]["w1"]
            p = M @ w
            mu_i = inc[f"{iso}_pIC50_direct_inhibition"].mean()
            sd_i = inc[f"{iso}_pIC50_direct_inhibition"].std()
            z = (p - p.mean()) / p.std()
            placed = np.clip(mu_i + sd_i * z, R_FLOOR, None)
            out[f"{iso}_pIC50_direct_inhibition"] = placed
            print(iso, "emit rho_vs_incumbent",
                  round(float(spearmanr(placed, inc[f"{iso}_pIC50_direct_inhibition"]).statistic), 4),
                  "mean/sd", round(float(placed.mean()), 3), round(float(placed.std()), 3),
                  flush=True)
        out.to_csv(os.path.join(CACHE, out_csv), index=False)
        print(f"wrote cache/{out_csv}", flush=True)


R_FLOOR = 1.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--l4", action="store_true")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--wout", default="hinge_w_cp18.json")
    ap.add_argument("--report", default="hinge_blend_l1.json")
    ap.add_argument("--incumbent", default=INCUMBENT_CSV)
    ap.add_argument("--out", default="regression_cp18_hinge_submission.csv")
    ap.add_argument("--ship", default="",
                    help="comma list of isoforms to ship (gate AND ship-set)")
    a = ap.parse_args()
    main(a.emit, a.l4, a.pool, a.wout, a.report, a.incumbent, a.out, a.ship)
