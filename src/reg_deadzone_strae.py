"""R4 ST-RAE simulation: does the dead-zone member reduce the scored metric?

The R2 check (cp17 0.4583 vs cp16 0.4584) was the wrong metric - R4 is
ST-RAE-native. ST-RAE = rae_soft_threshold_absolute_error: a prediction inside
the DRC band [lo,hi] contributes ZERO error; outside, it contributes the
distance to the nearest edge. The dead-zone member (fitted to clip(oof, lo, hi))
is designed to push predictions toward/into the band.

This simulates ST-RAE on the TRAIN set (labeled rows with bands) for:
  - cp16 OOF (current pool)
  - cp17 OOF (cp16 + deadzone member)
  - deadzone member standalone
  - cp16 OOF directly clipped to the band (the oracle placement of cp16)
Under the official metric. Train-set sim is optimistic (in-distribution) but
shows the DIRECTION and RELATIVE size of the dead-zone effect.

CPU-only.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
import run_regression as R  # noqa: E402
from blend_nested_all import FAMS, POOLS  # noqa: E402
from evaluation.custom_scoring_functions import rae_soft_threshold_absolute_error as STRAE  # noqa: E402

ISO = R.ISOFORMS
K = 5


def zavg(files):
    dfs = [pd.read_csv(f) for f in files]
    df = dfs[0].copy()
    for iso in ISO:
        if len(dfs) > 1:
            z = np.zeros(len(df))
            for d in dfs:
                p = d[iso].values.astype(float)
                z += (p - p.mean()) / p.std()
            df[iso] = z / len(dfs)
    return df


def greedy(zs, y, m, rounds=12):
    from scipy.stats import pearsonr
    keys = list(zs)
    w = {k: 0.0 for k in keys}
    for _ in range(rounds):
        bestk, bestr = None, -2
        for k in keys:
            cand = sum((w[j] + (1 if j == k else 0)) * zs[j] for j in keys)
            tot = sum(w.values()) + 1
            r = float(pearsonr(y[m], (cand / tot)[m]).statistic)
            if r > bestr:
                bestr, bestk = r, k
        w[bestk] += 1
    return {k: v / sum(w.values()) for k, v in w.items()}


def nested_oof(oofs, names, folds, targets, smi):
    out = {iso: np.full(len(smi), np.nan) for iso in ISO}
    for iso in ISO:
        y = targets[R.DIRECT[iso]].values
        m = ~np.isnan(y)
        zs_all = {}
        for k, df in oofs.items():
            if k not in names:
                continue
            v = df[iso].values.astype(float)
            z = np.full(len(v), np.nan)
            z[m] = (v[m] - v[m].mean()) / v[m].std()
            zs_all[k] = z
        for f in range(K):
            trm = m & (folds != f)
            vam = m & (folds == f)
            zt = {}
            for k in zs_all:
                v = zs_all[k].copy()
                mu = v[trm].mean() if not np.isnan(v[trm]).all() else 0
                sd = v[trm].std()
                zt[k] = (v - mu) / (sd if sd > 0 else 1)
            w = greedy(zt, y, trm)
            zf = sum(w[k] * zt[k] for k in w)
            out[iso][vam] = zf[vam]
    return out


def main():
    Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
    smi = Xtr["SMILES"].values
    folds = np.load(os.path.join(CACHE, "folds.npy"))

    names16 = [n for n in POOLS["cp16"] if FAMS.get(n)]
    oofs = {k: zavg(FAMS[k]) for k in names16}
    oofs["ft_deadzone"] = pd.read_csv(os.path.join(CACHE, "ft_oof_deadzone.csv")).set_index("SMILES").reindex(smi)
    names17 = names16 + ["ft_deadzone"]

    cp16 = nested_oof(oofs, names16, folds, targets, smi)
    cp17 = nested_oof(oofs, names17, folds, targets, smi)
    dz = oofs["ft_deadzone"]

    # cp16/cp17 nested OOF are in z-space; place to pIC50 (train mean/std) for
    # the band-scale ST-RAE. dz is already pIC50-scale (trained on pIC50 target).
    def to_pic50(preds):
        out = {}
        for iso in ISO:
            y = targets[R.DIRECT[iso]].values
            ym = y[~np.isnan(y)]
            out[iso] = preds[iso] * float(ym.std()) + float(ym.mean())
        return out

    cp16p = to_pic50(cp16)
    cp17p = to_pic50(cp17)
    dzp = {iso: dz[iso].values for iso in ISO}

    def strae_macro(preds):
        rows = {}
        for iso in ISO:
            y = targets[R.DIRECT[iso]].values
            lo = bounds[f"{iso}_lo"].values.astype(float)
            hi = bounds[f"{iso}_hi"].values.astype(float)
            p = preds[iso]
            valid = ~np.isnan(y) & ~np.isnan(p) & ~np.isnan(lo) & ~np.isnan(hi)
            rows[iso] = float(STRAE(y[valid], p[valid], hi[valid], lo[valid]))
        return rows

    print("ST-RAE simulation on TRAIN (labeled rows with DRC bands, pIC50 scale)\n")
    print(f"  {'variant':<28} " + "  ".join(f"{i:>8}" for i in ISO) + f"  {'macro':>8}")
    variants = {}
    for label, preds in [("cp16 OOF", cp16p), ("cp17 OOF (cp16+dz)", cp17p), ("deadzone standalone", dzp)]:
        rows = strae_macro(preds)
        variants[label] = rows
        print(f"  {label:<28} " + "  ".join(f"{rows[i]:>8.4f}" for i in ISO) + f"  {np.mean(list(rows.values())):>8.4f}")

    # Oracle: cp16 OOF (pIC50) clipped to the band (best-case placement of the ranking)
    cp16_clip = {iso: np.clip(cp16p[iso],
                              bounds[f"{iso}_lo"].values.astype(float),
                              bounds[f"{iso}_hi"].values.astype(float)) for iso in ISO}
    rows = strae_macro(cp16_clip)
    variants["cp16 OOF clipped (oracle)"] = rows
    print(f"  {'cp16 OOF clipped (oracle)':<28} " + "  ".join(f"{rows[i]:>8.4f}" for i in ISO) + f"  {np.mean(list(rows.values())):>8.4f}")

    print("\n  Deltas (ST-RAE, lower=better):")
    base = float(np.mean(list(variants["cp16 OOF"].values())))
    for label in ["cp17 OOF (cp16+dz)", "deadzone standalone", "cp16 OOF clipped (oracle)"]:
        d = float(np.mean(list(variants[label].values()))) - base
        print(f"    {label:<28} {d:+.4f} macro")

    import json
    with open(os.path.join(CACHE, "r4_strae_sim.json"), "w") as f:
        json.dump(variants, f, indent=2, default=float)
    print(f"\nSaved r4_strae_sim.json")


if __name__ == "__main__":
    main()
