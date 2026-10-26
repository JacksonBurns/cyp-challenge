"""Evaluate pcgrad TDI head under field scoring: E[MCC] at fixed call
fractions across prior grid pi (matches q1_tdi_imbalance emcc_at_fraction).
Compare vs v4_candidate E[MCC] (2D6 0.163 / 3A4 0.409, f=0.33/0.26)."""
import numpy as np
import pandas as pd

CACHE = "/home/jackson/cyp-challenge/cache"
DATA = "/home/jackson/cyp-challenge/data/"
PI = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def mcc(y, yp):
    tp = np.sum((yp == 1) & (y == 1)); tn = np.sum((yp == 0) & (y == 0))
    fp = np.sum((yp == 1) & (y == 0)); fn = np.sum((yp == 0) & (y == 1))
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den


def emcc_at_fraction(y, prob, f, pi):
    # relabel to prior pi via quantile shift of logit(=rank) then cut at f
    rank = np.argsort(np.argsort(prob)) / (len(prob) - 1)
    lab = (rank >= 1.0 - pi).astype(int)  # pseudo-labels at prior pi
    thr = np.quantile(prob[lab == 1], 1.0 - f) if (lab == 1).sum() else 1.0
    yp = (prob >= thr).astype(int)
    return mcc(y, yp)


def main():
    p = pd.read_csv(f"{CACHE}/joint_oof_tdi_pcgrad.csv")
    emx = pd.read_csv(DATA + "cyp-challenge-TRAIN_Emax.csv")
    emx = emx.drop_duplicates("SMILES").set_index("SMILES")
    p = p.drop_duplicates("SMILES").set_index("SMILES")
    for col, head, f in [("CYP2D6_is_TDI", "is_TDI_2D6", 0.33),
                         ("CYP3A4_is_TDI", "is_TDI_3A4", 0.26)]:
        y = emx[col].astype(float)
        pr = p[head].astype(float)
        common = y.index.intersection(pr.index)
        m = y.loc[common].notna()
        yv = y.loc[common][m].values.astype(int)
        pv = pr.loc[common][m].values
        # best-threshold MCC
        order = np.argsort(pv)
        best, bt = -2, 0.5
        for t in np.quantile(pv, np.linspace(0.5, 0.999, 60)):
            mc = mcc(yv, (pv >= t).astype(int))
            if mc > best:
                best, bt = mc, t
        e_f = float(np.mean([emcc_at_fraction(yv, pv, f, pi) for pi in PI]))
        e_best = float(np.mean([emcc_at_fraction(
            yv, pv, np.quantile(pv[yv == 1], q), pi)
            for q in (0.4, 0.5, 0.6, 0.7) for pi in PI]))
        print(f"{head}: n={len(yv)} pos={yv.sum()} "
              f"MCC_best={best:.4f}@thr={bt:.3f} "
              f"E[MCC]@f={f}: {e_f:.4f} | E[MCC]@q-sweep: {e_best:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
