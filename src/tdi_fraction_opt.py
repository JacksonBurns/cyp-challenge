"""Choose TDI operating point (predicted-positive fraction on test) with a
nonparametric rank-conditional model + base-rate hypothesis posterior.

P(pos | rank) shape: observed OOF top-k enrichment curve (raw probas are
over-dispersed; observed labels are more concentrated at top ranks than the
probas imply, so we must NOT simulate from raw probas).

Base rate pi on test: posterior from direct-pIC50 decile conditioning
(labeling rule ties pi to the direct-activity distribution) blended with the
train labeled rate. Simulates MCC(f, pi), integrates over pi, also minimax.
Writes cache/tdi_fraction_optima.json
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
rng = np.random.default_rng(7)

ISO = ["CYP2D6", "CYP3A4"]
N_BINS = 24  # rank bins for the observed conditional curve


def mcc_counts(tp, fp, tn, fn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def rank_conditional_curve(scores, y, n_bins=N_BINS):
    """Observed positive rate per rank bin, sorted by score desc."""
    order = np.argsort(-scores)
    ys = np.asarray(y)[order]
    bins = np.array_split(ys, n_bins)
    rates = np.array([b.mean() for b in bins])
    sizes = np.array([len(b) for b in bins])
    return rates, sizes


def rescale_to_pi(rates, pi, lo=0.002, hi=0.98):
    """Shift the rank-bin conditional curve in logit space so mean == pi."""
    r = np.clip(rates, lo, hi)
    a, b = -30.0, 30.0
    for _ in range(60):
        m = (a + b) / 2
        v = 1 / (1 + np.exp(-(np.log(r / (1 - r)) + m)))
        if v.mean() > pi:
            b = m
        else:
            a = m
    return 1 / (1 + np.exp(-(np.log(r / (1 - r)) + (a + b) / 2)))


def emcc(probs_by_rank, f, n_trials=800):
    n = len(probs_by_rank)
    k = max(1, min(n - 1, int(round(f * n))))
    tot = 0.0
    for _ in range(n_trials):
        y = (rng.random(n) < probs_by_rank).astype(int)
        tp = int(y[:k].sum()); fp = k - tp
        P = int(y.sum()); fn = P - tp; tn = n - k - fn
        tot += mcc_counts(tp, fp, tn, fn)
    return tot / n_trials


def main():
    oof = np.load(os.path.join(CACHE, "tdi_oof.npz"))
    ana = json.load(open(os.path.join(CACHE, "tdi_threshold_analysis.json")))
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates(
        "SMILES").reset_index(drop=True)
    tdi = pd.read_csv(os.path.join(ROOT, "data", "cyp-challenge-TRAIN_TDI.csv"))
    emx = pd.read_csv(os.path.join(ROOT, "data", "cyp-challenge-TRAIN_Emax.csv"))
    lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in ISO]],
                            emx[["SMILES"] + [f"{i}_is_TDI" for i in ISO]]]).drop_duplicates(
        "SMILES").set_index("SMILES")
    Y = lab_by_smi.reindex(Xtr["SMILES"])

    pi_post = {
        # pi informed by direct-pIC50 decile conditioning, with mass kept on
        # the train rate to respect model uncertainty.
        "CYP3A4": {0.25: 0.08, 0.30: 0.15, 0.35: 0.22, 0.40: 0.28, 0.45: 0.17, 0.50: 0.10},
        "CYP2D6": {0.216: 0.15, 0.26: 0.22, 0.30: 0.26, 0.34: 0.22, 0.38: 0.10, 0.42: 0.05},
    }
    fracs = np.arange(0.05, 0.61, 0.01)
    out = {}
    for iso in ISO:
        y = Y[f"{iso}_is_TDI"].values
        lab = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[lab]], dtype=int)
        po = oof[iso][lab]
        rates, _ = rank_conditional_curve(po, yl)
        print(f"{iso} observed rank-conditional (top->bottom, {N_BINS} bins):")
        print("  ", np.round(rates, 3))
        curves = {}
        for pi in pi_post[iso]:
            pr = rescale_to_pi(rates, pi)
            curves[pi] = [round(emcc(pr, f), 4) for f in fracs]
        weighted = np.zeros(len(fracs))
        for pi, c in curves.items():
            weighted += pi_post[iso][pi] * np.array(c)
        worst = np.min(np.array([c for c in curves.values()]), axis=0)
        ib = int(np.argmax(weighted)); im = int(np.argmax(worst))
        # also validate the simulator: MCC at each f using pi=train rate vs OOF observed
        out[iso] = {
            "rank_conditional_rates": [round(float(r), 4) for r in rates],
            "posterior_optimal_frac": float(fracs[ib]),
            "posterior_optimal_mcc": round(float(weighted[ib]), 4),
            "minimax_optimal_frac": float(fracs[im]),
            "weighted_mcc_at_f": {f"{f:.2f}": round(float(weighted[int(round((f - 0.05) / 0.01))]), 4)
                                   for f in [0.08, 0.15, 0.213, 0.25, 0.287, 0.338, 0.389, 0.45]},
            "curves_by_pi": {str(k): v for k, v in curves.items()},
            "fracs": [round(float(f), 2) for f in fracs],
        }
        print(iso, "posterior-opt f:", out[iso]["posterior_optimal_frac"],
              "E[MCC]:", out[iso]["posterior_optimal_mcc"], "| minimax f:", out[iso]["minimax_optimal_frac"])
        for f, v in out[iso]["weighted_mcc_at_f"].items():
            print(f"   f={f}: E[MCC]={v}")
    json.dump(out, open(os.path.join(CACHE, "tdi_fraction_optima.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
