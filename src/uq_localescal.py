"""FINAL battery for the family-disagreement/LLOD line (honest CV, CPU).

Direction analysis (uq_direction): high-UQ rows are LOW-side; most of
their metric error is below-lo. The assay floor is a global constant
(~1.03 pIC50, LLOD of 10 nM) -> the user's "set them to LLOD" proposal
IS testable on the blinded test set (no per-row bounds needed).

Battery (zoo6 ship blend OOF, per-isoform, scaffold 5-fold):
  T1  LLOD floor: if UQ >= q: P := max(P, LLOD)
  T2  shrink-to-median: high-UQ rows: P := w*P + (1-w)*median(P)
  T3  shrink-to-iso-mean: same with isoform train mean
  T4  LLOD floor with lower quantile thresholds (25/50/75%)
All fitted honestly: UQ from train folds only, thresholds on train folds,
scored on held-out fold. Gate: mean gain > 0.005 soft_rae on >=3/5 folds
for >=1 isoform.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
LLOD = 1.03  # global assay floor (pIC50), verified: mode of every iso's lo
RNG = np.random.default_rng(7)


def blend_oof(iso, W):
    """zoo6 hinge-weighted OOF for one isoform, aligned to H.Xtr SMILES."""
    smi = H.Xtr["SMILES"].values
    P = np.zeros(len(smi))
    tot = np.zeros(len(smi))
    for fam, w in W[iso]["w"].items():
        if not isinstance(w, (int, float)) or abs(w) < 1e-9:
            continue
        v = H.raw_oof_fam(fam)[iso].reindex(smi).values.astype(float)
        ok = ~np.isnan(v)
        P[ok] += w * v[ok]
        tot[ok] += w
    P[tot > 0] /= tot[tot > 0]
    return smi, P


def fam_spread(iso, W):
    """Cross-family spread (UQ) from zoo6 member OOFs, nan-safe."""
    smi = H.Xtr["SMILES"].values
    cols = []
    for fam, w in W[iso]["w"].items():
        if not isinstance(w, (int, float)) or abs(w) < 1e-9:
            continue
        v = H.raw_oof_fam(fam)[iso].reindex(smi).values.astype(float)
        cols.append(v)
    M = np.stack(cols, axis=1)
    sd = np.nanstd(M, axis=1)
    return np.nan_to_num(sd, nan=0.0)


def sr(y, p, lo, hi):
    err = np.clip(p - hi, 0, None) + np.clip(lo - p, 0, None)
    num = np.sqrt(np.mean(err ** 2))
    from scipy.stats import spearmanr
    rho = spearmanr(y, p).statistic
    return num * (2.0 - rho)


def main():
    import json
    W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
    print(f"{LLOD=}")
    report = {}
    for iso in H.ISO:
        smi, P = blend_oof(iso, W)
        UQ = fam_spread(iso, W)
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
        lo = H.bounds[f"{iso}_lo"].reindex(smi).values.astype(float)
        hi = H.bounds[f"{iso}_hi"].reindex(smi).values.astype(float)
        m = ~(np.isnan(y) | np.isnan(P) | np.isnan(lo) | np.isnan(hi))
        idx = np.where(m)[0]
        folds = np.array_split(RNG.permutation(idx), 5)
        def floor_q(q):
            return lambda ptr, utr, ute, k: np.where(ute >= np.quantile(utr, q), np.maximum(k, LLOD), k)
        def shrink(q, w, mode):
            return lambda ptr, utr, ute, k: np.where(
                ute >= np.quantile(utr, q),
                w * k + (1 - w) * (np.median(ptr) if mode == "med" else np.mean(ptr)), k)
        tests = {
            "T1_floor_q90": floor_q(0.90), "T2_floor_q75": floor_q(0.75),
            "T3_floor_q50": floor_q(0.50),
            "T4_shrinkmed90": shrink(0.90, 0.5, "med"),
            "T5_shrinkmean90": shrink(0.90, 0.5, "mean"),
        }
        base, gains = [], {t: [] for t in tests}
        for f in folds:
            tr = np.concatenate([g for g in folds if g is not f])
            b0 = sr(y[f], P[f], lo[f], hi[f])
            base.append(b0)
            for tname, fn in tests.items():
                Pk = fn(P[tr], UQ[tr], UQ[f], P[f])
                gains[tname].append(b0 - sr(y[f], Pk, lo[f], hi[f]))
        row = {"base": float(np.mean(base))}
        for tname, g in gains.items():
            row[tname] = (round(float(np.mean(g)), 4), sum(1 for x in g if x > 0))
        report[iso] = row
        print(iso, json.dumps(row))
    json.dump(report, open(os.path.join(CACHE, "uq_localescal.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
