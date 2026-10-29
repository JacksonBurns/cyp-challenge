"""R3-1b: stack mechanism (bioactivation SMARTS) + ADME2 physics features
on top of the v5-ship pooled OOF probabilities for TDI classification.

Base = v6b capped pooled OOF (the v5 ship machinery, per-labeled-row).
Stack = scaffold-fold LGBM on [logit(p_base), MECH26] and
[+ ADME2 block]; evaluate E[MCC] under the board pi-prior framework
(rank-conditional curve, v7 machinery) at tuned fraction.
Gate: E[MCC] gain > 0.02 on >=1 isoform, stack beats base at argmax.
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
CACHE = os.path.join(ROOT, "cache")
DATA = os.path.join(ROOT, "data")

from run_regression import make_folds, scaffold_groups  # noqa: E402
from tdi_fraction_opt import rank_conditional_curve, rescale_to_pi, mcc_counts  # noqa: E402
import r31_tdi_mech as M  # noqa: E402  (MECH builder reuse)

ISO = ["CYP2D6", "CYP3A4"]
PI_PRIOR = {0.08: 0.10, 0.11: 0.20, 0.14: 0.25, 0.17: 0.22, 0.213: 0.15, 0.25: 0.08}
fracs = np.arange(0.05, 0.46, 0.01)
N_TRIALS = 1500
rng = np.random.default_rng(11)


def emcc(probs_by_rank, f):
    n = len(probs_by_rank)
    k = max(1, min(n - 1, int(round(f * n))))
    tot = 0.0
    for _ in range(N_TRIALS):
        y = (rng.random(n) < probs_by_rank).astype(int)
        tp = int(y[:k].sum()); fp = k - tp
        P = int(y.sum()); fn = P - tp; tn = n - k - fn
        tot += mcc_counts(tp, fp, tn, fn)
    return tot / N_TRIALS


def best_emcc(p, yl):
    rates, _ = rank_conditional_curve(p, yl)
    w = np.zeros(len(fracs))
    for pi, pw in PI_PRIOR.items():
        pr = rescale_to_pi(rates, pi)
        for i, f in enumerate(fracs):
            w[i] += pw * emcc(pr, f)
    i = int(np.argmax(w))
    return float(w[i]), float(fracs[i])


def main():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv"))
    lab = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                     emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab.reindex(Xtr["SMILES"])

    MECH = M.bioact_feats(Xtr["SMILES"].tolist())
    A2 = np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy"))
    # align A2 (built on H.Xtr order) to this Xtr order by SMILES
    import hinge_blend as H
    pos = {s: i for i, s in enumerate(H.Xtr["SMILES"])}
    idx = [pos.get(s, -1) for s in Xtr["SMILES"]]
    A2 = A2[np.array(idx)]

    smi = Xtr["SMILES"].tolist()
    fold_assign = make_folds(scaffold_groups(smi), k=5, seed=3)

    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        m = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[m]], dtype=int)
        pb = np.load(os.path.join(CACHE, f"tdi_fb4_v6b_merged_{iso}cap.npy"))
        assert len(pb) == m.sum(), (len(pb), m.sum())
        lb = np.log(np.clip(pb, 1e-6, 1 - 1e-6) / (1 - np.clip(pb, 1e-6, 1 - 1e-6)))
        b0, f0 = best_emcc(pb, yl)
        print(f"{iso} BASE v6bcap E[MCC] {b0:.4f} @ f={f0:.2f} (n={m.sum()}, pi_hat={yl.mean():.3f})", flush=True)

        Xl = Xtr[m].reset_index(drop=True)
        for nm, extra in [("mech", MECH[m]), ("adme2", A2[m]), ("mech+adme2", np.hstack([MECH[m], A2[m]]))]:
            Xs = np.hstack([lb[:, None], extra])
            oof = np.zeros(m.sum())
            for fv in range(5):
                te = np.where(fold_assign == fv)[0]
                tr = np.where(fold_assign != fv)[0]
                g = LGBMClassifier(n_estimators=300, learning_rate=0.06, num_leaves=31,
                                   random_state=0, verbose=-1)
                g.fit(Xs[tr], yl[tr])
                oof[te] = g.predict_proba(Xs[te])[:, 1]
            b1, f1 = best_emcc(oof, yl)
            print(f"{iso} +{nm:11s} E[MCC] {b1:.4f} @ f={f1:.2f}  delta {b1-b0:+.4f}", flush=True)


if __name__ == "__main__":
    main()
