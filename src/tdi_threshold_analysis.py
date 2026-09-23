"""Threshold-transfer analysis for TDI submission.

Compares OOF vs test probability distributions and evaluates three
threshold-selection strategies per isoform:
  A) OOF-argmax MCC threshold applied raw to test
  B) rate-matched: test threshold whose positive rate == train labeled pos rate
  C) quantile-matched: test threshold preserving the OOF operating point
     (same predicted-positive fraction on test as OOF had at the OOF argmax)

Also: calibration check (reliability of OOF probs) and an empirical estimate
of the expected test positive rate via direct-pIC50 conditioning.
Writes cache/tdi_threshold_analysis.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

ISO = ["CYP2D6", "CYP3A4"]


def mcc(tp, fp, tn, fn):
    n = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if n <= 0:
        return 0.0
    return (tp * tn - fp * fn) / np.sqrt(n + 1e-12)


def mcc_at(p, y):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return mcc(tp, fp, tn, fn)


def main():
    oof = np.load(os.path.join(CACHE, "tdi_oof.npz"))
    probs = pd.read_csv(os.path.join(CACHE, "tdi_test_probs.csv")).set_index("SMILES")
    cv = json.load(open(os.path.join(CACHE, "tdi_cv.json")))
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    # rebuild label frame EXACTLY as run_tdi.py did (oof npz is in Xtr-dedup order)
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates(
        "SMILES").reset_index(drop=True)
    lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                            emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates(
        "SMILES").set_index("SMILES")
    Y = lab_by_smi.reindex(Xtr["SMILES"])
    sub_reg = pd.read_csv(os.path.join(CACHE, "raw_regression_submission.csv"))

    report = {}
    grid = np.linspace(0.02, 0.98, 961)
    for iso in ISO:
        col = f"{iso}_is_TDI"
        y = Y[col].values
        lab = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[lab]], dtype=int)
        po = oof[iso][lab]  # OOF probs, labeled only
        pt = probs[f"{iso}_proba"].reindex(test["SMILES"].values).values
        assert len(pt) == 750 and not np.isnan(pt).any()
        base = cv[iso]["pos_rate"]
        t_star = cv[iso]["best_thr"]

        # distributions
        qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        dist = {
            "oof_mean": float(po.mean()), "test_mean": float(pt.mean()),
            "oof_q": [round(float(np.quantile(po, q)), 4) for q in qs],
            "test_q": [round(float(np.quantile(pt, q)), 4) for q in qs],
        }

        oof_pred_frac_at_star = float((po >= t_star).mean())
        test_pred_frac_at_star = float((pt >= t_star).mean())

        # OOF curve: for every threshold, OOF pred-positive rate and MCC
        oof_fr = np.array([(po >= t).mean() for t in grid])
        oof_mcc = np.array([mcc_at((po >= t).astype(int), yl) for t in grid])

        # B) rate-matched on train base rate
        # threshold on test giving positive rate == labeled train pos rate
        order = np.sort(pt)
        n_pos_target = int(round(base * 750))
        t_B = float(order[-n_pos_target]) if n_pos_target > 0 else float(order[-1])
        # what OOF MCC corresponds to that same threshold?
        iB = int(np.argmin(np.abs(grid - t_B)))
        mcc_B_oof = float(oof_mcc[iB])
        oof_fr_B = float(oof_fr[iB])

        # C) quantile-matched: preserve OOF operating point (pred fraction)
        t_C = float(np.quantile(pt, 1.0 - oof_pred_frac_at_star))
        iC = int(np.argmin(np.abs(grid - t_C)))
        mcc_C_oof = float(oof_mcc[iC])

        # calibration of OOF: 10 bins of OOF prob, mean prob vs observed rate
        bins = np.quantile(po, np.linspace(0, 1, 11))
        bins[0] = -1e-9
        rel = []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (po > a) & (po <= b)
            if m.sum() > 20:
                rel.append((round(float(po[m].mean()), 3), round(float(yl[m].mean()), 3), int(m.sum())))

        # empirical expected test pos rate: condition train TDI-positive rate on
        # direct pIC50 deciles, then reweight by OUR predicted test direct pIC50.
        pcol = f"{iso}_pIC50_direct_inhibition"
        dp_all = pd.to_numeric(tdi.set_index("SMILES")[pcol], errors="coerce").reindex(
            Xtr["SMILES"]).values if pcol in tdi.columns else None
        if dp_all is not None:
            ok_lab = (~pd.isna(dp_all)) & lab
            idx_lab = np.where(lab)[0]
            keep = ok_lab[lab]  # bool over labeled subset
            dp_ok = pd.Series(dp_all[lab][keep])
            yl_sel = yl[keep]
            dec = pd.qcut(dp_ok, 10, duplicates="drop")
            grp = pd.DataFrame({"r": np.asarray(yl_sel), "d": dec.values})
            agg = grp.groupby("d", observed=True)["r"].agg(["mean", "count"])
            edges = dp_ok.groupby(dec.values, observed=True).agg(["min", "max"])
            pred_dp = sub_reg[pcol].values
            mids = ((edges["min"] + edges["max"]) / 2).values
            idx = np.abs(pred_dp[:, None] - mids[None, :]).argmin(1)
            rates = agg["mean"].values
            est_pos_rate = float(np.mean(rates[idx]))
            report.setdefault("decile_rates_" + iso, [
                [str(i), round(float(m), 3), int(c)] for i, (m, c) in enumerate(zip(rates, agg["count"]))])
        else:
            est_pos_rate = None

        report[iso] = {
            "base_train_pos_rate": base, "t_star_oof_argmax": t_star,
            "oof_mcc_at_star": cv[iso]["oof_mcc"],
            "oof_pred_frac_at_star": oof_pred_frac_at_star,
            "test_pred_frac_at_star": test_pred_frac_at_star,
            "B_rate_matched_thr": t_B, "B_test_pos_rate": base,
            "B_oof_pred_frac_at_thr": oof_fr_B, "B_oof_mcc_at_thr": mcc_B_oof,
            "C_quantile_thr": t_C, "C_test_pred_frac": oof_pred_frac_at_star,
            "C_oof_mcc_at_thr": mcc_C_oof,
            "oof_mcc_at_0.5": cv[iso]["mcc_at_0.5"],
            "reliability_bins(meanprob,observed,n)": rel,
            "dist": dist,
            "est_test_pos_rate_via_direct_pIC50": est_pos_rate,
        }
        # also: OOF pos rate at thr where OOF MCC curve is within noise of max,
        # and MCC of predicting the base-rate fraction of positives by top-k proba
        k = int(round(base * len(yl)))
        topk_mcc = mcc_at((po >= np.sort(po)[-k]).astype(int), yl)
        report[iso]["oof_topk_by_base_rate_mcc"] = float(topk_mcc)

    with open(os.path.join(CACHE, "tdi_threshold_analysis.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
