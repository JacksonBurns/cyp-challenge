"""Q5 (CPU): interval supervision -> calibrated uncertainty, not point
accuracy. Quantify the measured dissociation using existing OOF artifacts:

For each family with OOF + pool weight (from hinge_blend_*):
 - standalone quality: RMSE of OOF vs labels (point accuracy),
 - pool weight w (decorrelation contribution),
 - interval calibration: empirical coverage of assay bounds by OOF preds
   (fraction of labeled rows with lo <= pred <= hi where bounds exist).
Prediction: hinge/intarm families have POOR RMSE, HIGH coverage, POSITIVE w;
MSE families good RMSE, coverage ~ nominal-low, w near zero.
Outputs a table for the methods report.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

CACHE = "/home/jackson/cyp-challenge/cache"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
DATA = "/home/jackson/cyp-challenge/data/"


def load_bounds():
    """assay bounds per (SMILES, iso) from inhibition file (conf_low/high)."""
    b = {}
    inh = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))
    inh = inh.drop_duplicates("SMILES").set_index("SMILES")
    inh = inh[~inh.index.duplicated(keep="first")]
    for iso in ISO:
        lo = inh[f"{iso}_pIC50_direct_inhibition_conf_low"].astype(float)
        hi = inh[f"{iso}_pIC50_direct_inhibition_conf_high"].astype(float)
        b[iso] = (lo, hi)
    return b


def main():
    bounds = load_bounds()
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    lab = ft.drop_duplicates("SMILES").set_index("SMILES")
    wsum = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "hinge_blend_*.json"))):
        P = json.load(open(f))
        for iso in ISO:
            wm = P.get(iso, {}).get("weights_fitted_top", {})
            for k, v in wm.items():
                wsum.setdefault(k, {}).setdefault(iso, []).append(v)
    rows = []
    for f in sorted(glob.glob(os.path.join(CACHE, "ft_oof_*.csv"))):
        stem = os.path.basename(f)[len("ft_oof_"):-len(".csv")]
        if "_s" in stem:
            continue
        key = f"ft_{stem}"
        ws = wsum.get(key, {})
        d = pd.read_csv(f).drop_duplicates("SMILES").set_index("SMILES")
        rec = {"family": stem,
               "mean_w": round(float(np.mean([np.mean(v) for v in ws.values()])),
                               3) if ws else np.nan}
        covs, rmses = [], []
        for iso in ISO:
            if iso not in d.columns:
                continue
            y = lab[iso].reindex(d.index).astype(float)
            p = d[iso].astype(float)
            m = y.notna() & p.notna()
            if m.sum() < 100:
                continue
            rmses.append(float(np.sqrt(((y[m] - p[m]) ** 2).mean())))
            if iso in bounds:
                lo, hi = bounds[iso]
                lo_m = lo.reindex(d.index); hi_m = hi.reindex(d.index)
                mb = m & lo_m.notna() & hi_m.notna()
                if mb.sum() >= 50:
                    inside = (p[mb] >= lo_m[mb]) & (p[mb] <= hi_m[mb])
                    covs.append(float(inside.mean()))
        rec["rmse"] = round(float(np.mean(rmses)), 4) if rmses else np.nan
        rec["coverage"] = round(float(np.mean(covs)), 3) if covs else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["mean_w"])
    df = df.sort_values("mean_w", ascending=False)
    df.to_csv(os.path.join(CACHE, "q5_interval.csv"), index=False)
    print(df.head(15).to_string(), flush=True)
    print("...", flush=True)
    print(df.tail(8).to_string(), flush=True)
    # dissociation test
    top = df[df["mean_w"] > 0.05]
    bot = df[df["mean_w"].abs() <= 0.02]
    print(f"POSITIVE-w families (n={len(top)}): rmse={top.rmse.mean():.3f} "
          f"coverage={top.coverage.mean():.3f}", flush=True)
    print(f"ZERO-w families (n={len(bot)}): rmse={bot.rmse.mean():.3f} "
          f"coverage={bot.coverage.mean():.3f}", flush=True)
    json.dump({"n_families": int(len(df))}, open(
        os.path.join(CACHE, "q5_interval.json"), "w"))


if __name__ == "__main__":
    main()
