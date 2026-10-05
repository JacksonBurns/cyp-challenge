"""R4: Dead-zone ST-RAE refit (Lizard Wizard A6) - CPU-only, no GPU.

ST-RAE per-compound loss is L(p) = |p - clip(p, lo, hi)| where [lo,hi] is the
DRC band. A LightGBM member fitted under absolute_error (regression_l1) to the
CLIPPED cp16-blend OOF prediction is a majorise-minimise descent on the metric's
own gradient. The target is clip(oof_pred, lo, hi) - derived from the HONEST OOF
prediction (not the model's own in-fold train-row predictions = the self-target
trap Lizard Wizard documents), so it is a legitimate OOF target.

Steps:
1. Reconstruct the cp16 pool NESTED OOF predictions (per-fold greedy weights,
   applied to the held-out fold) - the honest blend OOF array.
2. Clip each to its DRC band [lo, hi].
3. Train a new LGBM member (regression_l1) OOF on the clipped target, same
   features (base + NN) and scaffold folds seed 0 (regression convention).
4. Evaluate the new member standalone (nested Pearson) and as cp17 = cp16 +
   deadzone, plus the family-block paranoid audit.

Gate: nested macro R2 > 0.4569 (cp16) with per-isoform non-regression and
paranoid-merge clean.
"""
import os
import sys
import glob
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from lightgbm import LGBMRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
import run_regression as R  # noqa: E402
from blend_nested_all import FAMS, POOLS  # noqa: E402

warnings.filterwarnings("ignore")
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


def main():
    Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
    smi = Xtr["SMILES"].values
    base = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    folds = np.load(os.path.join(CACHE, "folds.npy"))  # seed-0 regression folds
    print(f"folds: {np.bincount(folds)}", flush=True)

    # NN features (same recipe as run_regression cv_config)
    Btr = (Xtr[R.FP_COLS].values > 0).astype(np.float32)
    Bte = (Xte[R.FP_COLS].values > 0).astype(np.float32)
    S = S  # train x train sim (from build_all)
    nn_block = R.nn_block  # reuse the proven implementation

    # --- 1. cp16 nested OOF predictions ---
    names = [n for n in POOLS["cp16"] if FAMS.get(n)]
    oofs = {k: zavg(FAMS[k]) for k in names}
    print(f"cp16 members: {len(names)}", flush=True)

    cp16_oof = {iso: np.full(len(smi), np.nan) for iso in ISO}
    cp16_nested = {}
    for iso in ISO:
        y = targets[R.DIRECT[iso]].values
        m = ~np.isnan(y)
        zs_all = {}
        for k, df in oofs.items():
            v = df[iso].values.astype(float)
            z = np.full(len(v), np.nan)
            z[m] = (v[m] - v[m].mean()) / v[m].std()
            zs_all[k] = z
        rs = []
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
            cp16_oof[iso][vam] = zf[vam]
            rs.append(float(pearsonr(y[vam], zf[vam]).statistic))
        r_honest = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
        cp16_nested[iso] = r_honest
        print(f"  cp16 {iso}: nested Pearson {r_honest:.4f}", flush=True)
    cp16_macro = float(np.mean([cp16_nested[i] ** 2 for i in ISO]))
    print(f"  cp16 macro R2 (honest): {cp16_macro:.4f}", flush=True)

    # --- 2. De-standardize cp16 OOF to pIC50 scale, then clip to DRC bands ---
    # The nested blend is in z-space (per-fold standardized). The DRC bands are on
    # the pIC50 scale. Affine-place the z-blend to pIC50 using the per-isoform
    # train label mean/std (the same placement the pipeline uses), THEN clip.
    clipped = {}
    cp16_pic50 = {}
    for iso in ISO:
        y = targets[R.DIRECT[iso]].values
        ym = y[~np.isnan(y)]
        y_mean, y_std = float(ym.mean()), float(ym.std())
        p_z = cp16_oof[iso]
        p = p_z * y_std + y_mean  # z -> pIC50
        cp16_pic50[iso] = p
        lo = bounds[f"{iso}_lo"].values.astype(float)
        hi = bounds[f"{iso}_hi"].values.astype(float)
        c = p.copy()
        valid = ~np.isnan(p) & ~np.isnan(lo) & ~np.isnan(hi)
        c[valid] = np.clip(p[valid], lo[valid], hi[valid])
        clipped[iso] = c
        n_clip = (c[valid] != p[valid]).sum()
        print(f"  {iso}: {valid.sum()} rows with band, {n_clip} clipped "
              f"(pIC50 scale, mean {y_mean:.2f} std {y_std:.2f})", flush=True)

    # --- 3. Train dead-zone member (regression_l1) OOF on clipped target ---
    # Features: base + NN (same as the gbm member). Target: clipped cp16 OOF.
    dz_oof = {iso: np.full(len(smi), np.nan) for iso in ISO}
    dz_params = dict(
        objective="regression_l1", n_estimators=1000, learning_rate=0.03,
        num_leaves=63, min_child_samples=15, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=2.0,
        n_jobs=-1, random_state=42, verbosity=-1,
    )
    for iso in ISO:
        ytarget = clipped[iso]
        # candidates: rows with a valid clipped target (band + cp16 pred)
        lab = ~np.isnan(ytarget)
        for f in range(K):
            qtr = np.where(folds != f)[0]
            qval = np.where(folds == f)[0]
            cand = np.where(lab & (folds != f))[0]
            qb = np.vstack([nn_block(S[qtr], cand, ytarget, q_idx=qtr),
                            nn_block(S[qval], cand, ytarget, q_idx=qval)])
            qb = qb.astype(np.float32)
            Xf_tr = np.hstack([base[qtr], qb[:len(qtr)]])
            Xf_val = np.hstack([base[qval], qb[len(qtr):]])
            trl = lab[qtr]
            m = LGBMRegressor(**dz_params)
            m.fit(Xf_tr[trl], ytarget[qtr][trl])
            dz_oof[iso][qval] = m.predict(Xf_val)
        print(f"  deadzone {iso}: OOF trained", flush=True)

    # Save the dead-zone member in the standard ft_oof format
    dz_df = pd.DataFrame({"SMILES": smi})
    for iso in ISO:
        dz_df[iso] = dz_oof[iso]
    dz_path = os.path.join(CACHE, "ft_oof_deadzone.csv")
    dz_df.to_csv(dz_path, index=False)
    print(f"  saved {os.path.basename(dz_path)}", flush=True)

    # --- 4. Evaluate: deadzone standalone + cp17 = cp16 + deadzone ---
    # Add deadzone to the OOF pool
    oofs["ft_deadzone"] = dz_df
    names17 = names + ["ft_deadzone"]

    def nested_pool(pool_names, label):
        nested = {}
        for iso in ISO:
            y = targets[R.DIRECT[iso]].values
            m = ~np.isnan(y)
            zs_all = {}
            for k, df in oofs.items():
                if k not in pool_names:
                    continue
                v = df[iso].values.astype(float)
                z = np.full(len(v), np.nan)
                z[m] = (v[m] - v[m].mean()) / v[m].std()
                zs_all[k] = z
            rs = []
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
                rs.append(float(pearsonr(y[vam], zf[vam]).statistic))
            r_honest = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
            nested[iso] = r_honest
        mac = float(np.mean([nested[i] ** 2 for i in ISO]))
        print(f"  {label}: " + "  ".join(f"{i} {nested[i]:.4f}" for i in ISO) + f"  macro {mac:.4f}", flush=True)
        return nested, mac

    # deadzone standalone
    dz_nested = {}
    for iso in ISO:
        y = targets[R.DIRECT[iso]].values
        m = ~np.isnan(y)
        rs = []
        for f in range(K):
            vam = m & (folds == f)
            dz_nested.setdefault(iso, [])
            r = pearsonr(y[vam], dz_oof[iso][vam]).statistic
            dz_nested[iso].append(float(r))
        dz_nested[iso] = float(np.tanh(np.mean(np.arctanh(np.clip(dz_nested[iso], -0.999, 0.999)))))
    dz_macro = float(np.mean([dz_nested[i] ** 2 for i in ISO]))
    print(f"\n  deadzone standalone: " + "  ".join(f"{i} {dz_nested[i]:.4f}" for i in ISO) + f"  macro {dz_macro:.4f}", flush=True)

    cp17_nested, cp17_macro = nested_pool(names17, "cp17 (cp16+deadzone)")

    # --- Summary ---
    print("\n" + "=" * 66)
    print("R4 DEAD-ZONE ST-RAE REFIT - NESTED-HONEST")
    print("=" * 66)
    print(f"  cp16  macro R2: {cp16_macro:.4f}  (bar 0.4569)")
    print(f"  deadzone standalone macro: {dz_macro:.4f}")
    print(f"  cp17  macro R2: {cp17_macro:.4f}  (delta {cp17_macro - cp16_macro:+.4f})")
    for iso in ISO:
        d = cp17_nested[iso] - cp16_nested[iso]
        print(f"    {iso}: cp16 {cp16_nested[iso]:.4f} -> cp17 {cp17_nested[iso]:.4f}  ({d:+.4f})")

    out = {
        "cp16": {"nested": cp16_nested, "macro": cp16_macro},
        "deadzone_standalone": {"nested": dz_nested, "macro": dz_macro},
        "cp17": {"nested": cp17_nested, "macro": cp17_macro},
        "delta_macro": cp17_macro - cp16_macro,
    }
    with open(os.path.join(CACHE, "r4_deadzone.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved r4_deadzone.json")

    if cp17_macro > cp16_macro:
        print(f"\n  cp17 BEATS cp16 by {cp17_macro - cp16_macro:+.4f} macro R2 (honest nested).")
        print("  Next: family-block paranoid audit of the deadzone member (new family,")
        print("  must not be a cp/UMTM twin) before candidate status.")
    else:
        print(f"\n  cp17 does NOT beat cp16 ({cp17_macro - cp16_macro:+.4f}). Dead-zone refit is neutral.")


if __name__ == "__main__":
    main()
