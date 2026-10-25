"""Q2: fit the CV->blind ST-RAE (and R2, rho) discount for shipped submissions.

For each scored leaderboard point (team JacksonBurns), pair the honest
CV-side score of the SHIPPED submission with what the blind board scored:

  cv_STRAE  = soft-threshold RAE surrogate (src/run_regression.py:soft_rae ==
              hinge_blend:st_rae; per isoform over labeled train rows with
              assay confidence bounds from build_all) of the exact shipped
              blend column: for hinge-refit candidates (cp18/21/22) the
              fitted-OOF ST-RAE of the SHIPPED isoform columns (ship cols =
              hinge-fitted where the gate passed, incumbent cp17-blend bytes
              elsewhere), for incumbent blend columns blend-weights @ raw
              family OOF. Macro = mean over the 4 isoforms.
  cv_R2     = nested-honest macro R2 (cache/blend_nested_*.json / NOTES);
              point 1 (raw GBM, no nested audit) uses the documented OOF
              Pearson^2 proxy 0.349 (NOTES sec 9).
  cv_rho    = per-iso Spearman of the same blend column vs labels, macro mean.

  blind_*   = MA-ST-RAE / MA-R2 / MA-Spearman from leaderboard/regression_*.csv;
              cp18's 0.5119 comes from NOTES sec 30b (re-score, no CSV).

Boards: the blind set CHANGED between the Sep 23 boards (era A) and the
Sep 26 / Oct 3 / Oct 22 boards (era B: same cp7 file scored 0.5354 on Sep26
vs 0.5366 on Oct3 -> board-side noise ~0.013, NOTES sec 20). Era-A blind
scores are drift-adjusted to the era-B scale with the PAIRED field drift
(NOTES sec 14: median new-minus-old +0.073 R2 / -0.058 ST-RAE / +0.017 rho);
an unadjusted sensitivity fit is also reported.

Fits (n=5 -> simplest defensible model, OLS with intercept):
  blind = a + b * cv per metric, adjusted + unadjusted, bootstrap CI (10k
  resamples of the 5 points); mean signed delta (blind - cv) over era-B pts;
  era-B-only anchor OLS slope b_anchor for gain propagation.

Practical output: predicted blind MA-ST-RAE for cp21/cp22 from their
ship-assembly CV surrogate via (a) the OLS discount line, (b) cp17/cp18-anchored
gain propagation with b_anchor, (c) the cp18 observed 13% transfer rate
(NOTES sec 30b).

Run (cyp env, repo root):  python src/q2_discount_fit.py
Output: cache/q2_discount.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CACHE = os.path.join(ROOT, "cache")
sys.path.insert(0, HERE)
import run_regression as R  # noqa: E402
import hinge_blend as HB    # noqa: E402

ISO = R.ISOFORMS
targets, bounds = HB.targets, HB.bounds
N_BOOT = 10000
RNG = np.random.default_rng(20261025)

# Paired field drift Sep23(eraA) -> Sep26(eraB) blind, new-minus-old (NOTES sec 14)
DRIFT = {"st_rae": -0.058, "r2": 0.073, "rho": 0.017}


def cv_metrics(blend):
    """Per-iso ST-RAE / R2 / rho of a blend OOF frame -> (macro, per-iso)."""
    per = {}
    for iso in ISO:
        y = targets[R.DIRECT[iso]].values
        lab = ~np.isnan(y)
        lo, hi = bounds[f"{iso}_lo"].values, bounds[f"{iso}_hi"].values
        rows = np.where(lab)[0]
        p = blend[iso].values[rows]
        ok = np.isfinite(p)
        yy, ll, hh = y[rows][ok], lo[rows][ok], hi[rows][ok]
        per[iso] = {
            "st_rae": round(float(HB.st_rae(p[ok], yy, ll, hh)), 4),
            "r2": round(float(1 - np.sum((yy - p[ok]) ** 2)
                              / np.sum((yy - yy.mean()) ** 2)), 4),
            "rho": round(float(spearmanr(yy, p[ok]).statistic), 4),
        }
    macro = {m: round(float(np.mean([per[i][m] for i in ISO])), 4)
             for m in ("st_rae", "r2", "rho")}
    return macro, per


def blend_frame(blend_json):
    """blend_json: parsed cache blendN_*.json (per-iso 'weights') -> blended raw-OOF frame."""
    weights_by_iso = {iso: blend_json[iso]["weights"] for iso in ISO}
    fams = {}
    df = pd.DataFrame(index=HB.Xtr["SMILES"].values)
    for iso in ISO:
        col = None
        for k, w in weights_by_iso[iso].items():
            fam = k.replace("_avg", "")  # 'ft_fullft_avg' == ft_fullft family
            if fam not in fams:
                fams[fam] = HB.raw_oof_fam(fam)
            c = fams[fam][iso].values * float(w)
            col = c if col is None else col + c
        df[iso] = col
    return df


def nested_macro(fn):
    """cache/blend_nested_*.json stores per-iso nested-honest PEARSON;
    macro R2 = mean of squares (NOTES sec 13: cp7 -> 0.4385)."""
    d = json.load(open(os.path.join(CACHE, fn)))
    return round(float(np.mean([v ** 2 for v in d.values()])), 4)


def load_board(csv_name):
    df = pd.read_csv(os.path.join(ROOT, "leaderboard", csv_name))
    row = df[df["Username"] == "JacksonBurns"].iloc[0]
    return {"st_rae": round(float(row["MA-ST-RAE"]), 4),
            "r2": round(float(row["MA-R²"]), 4),
            "rho": round(float(row["MA-Spearman's ρ"]), 4)}


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) <= 0:
        return None
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def boot_fit(x, y, targets_x, n=N_BOOT):
    xs, ys = np.asarray(x, float), np.asarray(y, float)
    A, B = [], []
    preds = {t: [] for t in targets_x}
    for _ in range(n):
        idx = RNG.integers(0, len(xs), len(xs))
        f = ols(xs[idx], ys[idx])
        if f is None:
            continue
        a, b = f
        A.append(a); B.append(b)
        for t in targets_x:
            preds[t].append(a + b * t)

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)]
    return {"a_CI95": ci(A), "b_CI95": ci(B), "n_valid": len(A),
            "pred": {str(t): [round(float(np.mean(v)), 4)] + ci(v)
                     for t, v in preds.items()}}


def main():
    # ---- CV surrogates of shipped columns ----
    m1, _ = cv_metrics(blend_frame(
        json.load(open(os.path.join(CACHE, "blendN.json")))))
    m1 = dict(m1, r2=0.349)  # documented Pearson^2 proxy (NOTES sec 9)

    m2, _ = cv_metrics(blend_frame(
        json.load(open(os.path.join(CACHE, "blendN_ext.json")))))
    m2 = dict(m2, r2=0.4127)  # documented nested-honest macro R2 (NOTES sec 12)

    m3, _ = cv_metrics(blend_frame(
        json.load(open(os.path.join(CACHE, "blendN_cp7.json")))))
    m3 = dict(m3, r2=0.4385)  # documented nested-honest macro R2 (NOTES sec 13)

    m4, p4 = cv_metrics(blend_frame(
        json.load(open(os.path.join(CACHE, "blendN_cp17.json")))))
    m4 = dict(m4, r2=0.4682)  # cp17 nested macro (NOTES sec 23)

    # shipped-column assembly for hinge candidates
    h18 = json.load(open(os.path.join(CACHE, "hinge_blend_l1.json")))
    h21 = json.load(open(os.path.join(CACHE, "hinge_blend_cp21.json")))
    h22 = json.load(open(os.path.join(CACHE, "hinge_blend_cp22.json")))

    def ship(fits, pass_map):
        cols = {}
        for iso in ISO:
            cols[iso] = (fits[iso]["fitted_oof_st_rae"] if pass_map[iso]
                         else p4[iso]["st_rae"])
        return cols

    ship18 = ship(h18, {iso: h18[iso]["gate_pass_L1"] for iso in ISO})
    ship21 = ship(h21, {iso: True for iso in ISO})       # cp21: all-4 zoo4 fit
    ship22 = ship(h22, {iso: h22[iso]["gate_pass_L1"] for iso in ISO})  # 2C9 fail -> cp18

    cv18 = round(float(np.mean(list(ship18.values()))), 4)
    cv21 = round(float(np.mean(list(ship21.values()))), 4)
    cv22 = round(float(np.mean(list(ship22.values()))), 4)
    # cp18/21/22 are reordering-only moment-restored: rho ~= incumbent blend OOF
    m5 = {"st_rae": cv18, "r2": 0.4682, "rho": m4["rho"]}

    # ---- points table ----
    pts = {
        "P1_Sep23interim_rawGBM":  ("regression_2026-09-23_interim_reveal.csv", "A", m1),
        "P2_Sep23final_ext2v3":    ("regression_2026-09-23_final_reveal.csv", "A", m2),
        "P3_Sep26_cp7":            ("regression_2026-09-26_interim_reveal2.csv", "B", m3),
        "P4_Oct22_cp17":           ("regression_2026-10-22_board.csv", "B", m4),
        "P5_Oct23rescore_cp18":    (None, "B", m5),
    }
    points = {}
    for k, (f, era, m) in pts.items():
        b = (load_board(f) if f
             else {"st_rae": 0.5119, "r2": None, "rho": None})  # NOTES sec 30b
        points[k] = {"era": era, "cv": m, "blind": b}

    def series(metric, adjust):
        xs, ys, keys = [], [], []
        for k, rec in points.items():
            cv, bl = rec["cv"].get(metric), rec["blind"].get(metric)
            if cv is None or bl is None:
                continue
            if adjust and rec["era"] == "A":
                bl = bl + DRIFT[metric]  # era-A blind -> era-B scale
            xs.append(cv); ys.append(bl); keys.append(k)
        return np.array(xs), np.array(ys), keys

    fits = {}
    for metric in ("st_rae", "r2", "rho"):
        f = {}
        for adjust in (True, False):
            x, y, keys = series(metric, adjust)
            a, b = ols(x, y)
            tgt = [round(v, 4) for v in (a + b * np.array([cv21, cv22]))]
            bs = boot_fit(x, y, tgt)
            resid = y - (a + b * x)
            f["adjusted" if adjust else "unadjusted"] = {
                "a": round(a, 4), "b": round(b, 4),
                "r2_of_fit": round(1 - np.sum(resid ** 2)
                                   / np.sum((y - y.mean()) ** 2), 4),
                "n": int(len(x)), "keys": keys,
                "boot_a_CI95": bs["a_CI95"], "boot_b_CI95": bs["b_CI95"],
                "boot_n_valid": bs["n_valid"], "boot_pred": bs["pred"],
                "pred_cp21": tgt[0], "pred_cp22": tgt[1],
            }
        xB, yB, kB = series(metric, True)
        fits[metric] = f
        eraB_mask = np.array([points[k]["era"] == "B" for k in kB])
        fits[metric + "_mean_signed_delta_eraB"] = {
            "keys": [k for k, msk in zip(kB, eraB_mask) if msk],
            "deltas": [round(float(bv - cvv), 4)
                       for bv, cvv in zip(yB[eraB_mask], xB[eraB_mask])],
            "mean": round(float(np.mean(yB[eraB_mask] - xB[eraB_mask])), 4),
        }

    # era-B anchor gain propagation (ST-RAE)
    xB, yB, kB = series("st_rae", True)
    b_anchor = ols(xB[np.array([points[k]["era"] == "B" for k in kB])],
                   yB[np.array([points[k]["era"] == "B" for k in kB])])[1]
    anchor = {"b_anchor_eraB": round(float(b_anchor), 4), "pred": {}}
    for name, cvx in (("cp21", cv21), ("cp22", cv22)):
        anchor["pred"][name] = {
            "from_cp17_blind_0.5133": round(0.5133 - b_anchor * (m4["st_rae"] - cvx), 4),
            "from_cp18_blind_0.5119_anchor_slope":
                round(0.5119 - b_anchor * (cv18 - cvx), 4),
            "from_cp18_blind_0.5119_rate_0.13":
                round(0.5119 - 0.13 * (cv18 - cvx), 4),
        }
    fits["st_rae_anchor"] = anchor

    out = {
        "method_doc": __doc__.strip().splitlines()[0],
        "points": points,
        "drift_adjust_eraA": DRIFT,
        "cv_ship_st_rae": {"cp18": cv18, "cp21": cv21, "cp22": cv22},
        "ship_cols_st_rae": {"cp18": ship18, "cp21": ship21, "cp22": ship22},
        "cp18_anchor": {"cv_gain_st_rae_vs_cp17": round(m4["st_rae"] - cv18, 4),
                        "blind_gain_st_rae": round(0.5133 - 0.5119, 4),
                        "transfer_rate": round((0.5133 - 0.5119) / (m4["st_rae"] - cv18), 3)},
        "per_iso_cp17_blend_cv": p4,
        "fits": fits,
    }
    with open(os.path.join(CACHE, "q2_discount.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("cv_ship_st_rae", "ship_cols_st_rae", "cp18_anchor")}, indent=1))
    print(json.dumps({m: {k: fits[m][k] for k in ("adjusted", "unadjusted")}
                      for m in ("st_rae", "r2", "rho")}, indent=1))
    print(json.dumps({m + "_delta": fits[m + "_mean_signed_delta_eraB"]
                      for m in ("st_rae", "r2", "rho")}, indent=1))
    print(json.dumps(fits["st_rae_anchor"], indent=1))


if __name__ == "__main__":
    main()
