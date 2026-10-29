"""R3-5c: gate computed docking scores as regression features.

Gate rows = TRAIN molecules with dock scores (1200 sample) - compare
GBM stack vs stack+DOCK4 on IDENTICAL rows (subset gate). Test side is
fully docked (750). Dock features: 4 Vina scores + min/max/range.
Seeds 0-3, macro MAE delta + on-A2 delta. CPU.
"""
import json
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

CACHE = H.CACHE
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def load_dock():
    sc = {}
    for iso in ISO:
        for fn in os.listdir(os.path.join(CACHE, "dock")):
            if fn.startswith(f"dock_{iso}_w") and fn.endswith(".json"):
                sc.update(json.load(open(os.path.join(CACHE, "dock", fn))))
    return sc


def feats_for(smis, sc):
    X = np.full((len(smis), 7), np.nan)
    for i, s in enumerate(smis):
        d = sc.get(s)
        if not d:
            continue
        v = [d.get(k) for k in ISO]
        if any(x is None for x in v):
            continue
        row = v + [min(v), max(v), max(v) - min(v)]
        X[i] = row
    return X


def HistGB(seed):
    return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                         random_state=seed)


def mae_gate(X, y, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        g = HistGB(seed)
        g.fit(X[tr], y[tr])
        pred[te] = g.predict(X[te])
    return float(np.mean(np.abs(pred - y)))


def main():
    sc = load_dock()
    tr_sm = H.Xtr["SMILES"].tolist()
    Dtr = feats_for(tr_sm, sc)
    ok = np.isfinite(Dtr[:, 0])
    print(f"docked train rows: {ok.sum()}", flush=True)
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    A2 = np.nan_to_num(np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy")), nan=0.0)
    Dtr = np.nan_to_num(Dtr, nan=0.0)
    for seed in [0, 1, 2, 3]:
        for iso in ISO:
            y = H.targets[f"{iso}_pIC50_direct_inhibition"].values.astype(np.float32)
            m = ok & np.isfinite(y)
            ym = y[m]
            m0 = mae_gate(Xbase[m], ym, seed)
            md = mae_gate(np.hstack([Xbase[m], Dtr[m]]), ym, seed)
            ma = mae_gate(np.hstack([Xbase[m], A2[m]]), ym, seed)
            mad = mae_gate(np.hstack([Xbase[m], A2[m], Dtr[m]]), ym, seed)
            print(f"seed{seed} {iso}: +DOCK {m0-md:+.4f} | DOCK-on-A2 {ma-mad:+.4f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
