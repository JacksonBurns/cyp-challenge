"""CYP2D6 ceiling diagnostic (CPU, honest nested).

The Oct-18 iteration closed on a no-op (x_d dead, pool saturated) and flagged
2D6 as the binding constraint. This script answers the one question the data
hasn't settled: is the 2D6 nested ~0.49 a HARD representation ceiling, or does
the *combiner* (per-fold greedy, used everywhere in the repo) leave headroom?

Method (exactly the repo honest recipe, blend_nested_all.py / pool_simplify.py):
  per scaffold fold f (seed 0, K=5):
    train mask = labeled & (fold != f);  val mask = labeled & (fold == f)
    z-score each member's full OOF vector with TRAIN-fold stats
    fit the combiner on the train fold, score Pearson on the val fold
  nested = Fisher-z avg of per-fold Pearson;  R2 = nested^2

Three combiners, all honest/nested, on the CYP2D6 column only:
  A. greedy  - the current repo combiner (reference; should reproduce ~0.49)
  B. rankavg - rank-average of members (combiner-free, robust to correlation)
  C. stack   - ridge on members' OOF -> target (fit per train fold)

If B or C beat A by more than fold noise (~0.004), the blend recipe has real
2D6 headroom -> a zero-GPU win. If not, 2D6 is a representation ceiling and
the next jump needs a genuinely new member, not a reshuffle.

Run:  ~/miniforge3/envs/cyp/bin/python src/d2d6_ceiling.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = "CYP2D6"

ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
y = ft[ISO].values.astype(float)
folds = ft["fold"].values
m = ~np.isnan(y)

# canonical families (drop _sN seed suffixes); seed-average within family
files = sorted(glob.glob(os.path.join(CACHE, "ft_oof_*.csv")))
fam = {}
for f in files:
    tag = os.path.basename(f).replace("ft_oof_", "").replace(".csv", "")
    base = tag.split("_s")[0]
    df = pd.read_csv(f)
    if ISO not in df.columns or len(df) != len(y):
        continue
    v = df[ISO].values.astype(float)
    fam.setdefault(base, []).append(v)

# seed-average per family
members = {k: np.mean(vs, axis=0) for k, vs in fam.items()}
# drop degenerate (constant / all-nan) members
members = {k: v for k, v in members.items()
           if np.nanstd(v) > 1e-8 and not np.isnan(v).all()}
names = list(members)
print(f"members ({len(names)}):", ", ".join(names))


def zfull(v, trm):
    vm = v[trm]
    mu, sd = vm.mean(), vm.std()
    return (v - mu) / (sd if sd > 0 else 1)


def greedy(zt, y, trm, vam, rounds=12):
    keys = list(zt)
    w = {k: 0.0 for k in keys}
    for _ in range(rounds):
        bestk, bestr = None, -2
        for k in keys:
            cand = sum((w[j] + (1 if j == k else 0)) * zt[j] for j in keys)
            tot = sum(w.values()) + 1
            r = pearsonr(y[trm], (cand / tot)[trm]).statistic
            if np.isnan(r):
                continue
            if r > bestr:
                bestr, bestk = r, k
        w[bestk] += 1
    zf = sum(w[k] * zt[k] for k in w)
    r = pearsonr(y[vam], zf[vam]).statistic
    return r, w


def rankavg(zt, y, trm, vam):
    # rank-average: per fold, rank each member's TRAIN values, average ranks
    # on the VAL fold (rank is monotone -> combiner-free, no weights to fit)
    n = len(y)
    acc = np.zeros(n)
    for k in zt:
        order = np.argsort(np.argsort(zt[k]))  # global rank 0..n-1
        acc += order
    acc /= len(zt)
    r = pearsonr(y[vam], acc[vam]).statistic
    return r


def stack(zt, y, trm, vam, alpha=10.0):
    # ridge on members' (train-fold-zscored) OOF -> target, fit on train fold
    X = np.column_stack([zt[k][trm] for k in zt])
    yt = y[trm]
    Xv = np.column_stack([zt[k][vam] for k in zt])
    rg = Ridge(alpha=alpha)
    rg.fit(X, yt)
    pv = rg.predict(Xv)
    r = pearsonr(y[vam], pv).statistic
    return r


def nested(score_fn):
    rs = []
    for fd in range(5):
        trm = m & (folds != fd)
        vam = m & (folds == fd)
        zt = {k: zfull(members[k], trm) for k in names}
        out = score_fn(zt, y, trm, vam)
        r = out if isinstance(out, float) else out[0]
        if not np.isnan(r):
            rs.append(r)
    if len(rs) < 5:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))


def nested_greedy():
    rs = []
    for fd in range(5):
        trm = m & (folds != fd)
        vam = m & (folds == fd)
        zt = {k: zfull(members[k], trm) for k in names}
        r, w = greedy(zt, y, trm, vam)
        if not np.isnan(r):
            rs.append(r)
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))


res = {
    "greedy": nested_greedy(),
    "rankavg": nested(rankavg),
    "stack": nested(stack),
}
print()
print("CYP2D6 honest nested (all combiners, full member set):")
for k, v in res.items():
    print(f"  {k:8s}  nested={v:.4f}  R2={v**2:.4f}")

best = max(res, key=res.get)
delta = res[best] - res["greedy"]
print()
print(f"best combiner: {best}  delta_vs_greedy={delta:+.4f}")
print("fold-noise band ~ +/-0.004; a real win needs |delta| > ~0.005")
verdict = ("COMBINER HEADROOM" if delta > 0.005 else
           "2D6 IS A REPRESENTATION CEILING (combiner not the lever)")
print("VERDICT:", verdict)

with open(os.path.join(CACHE, "d2d6_ceiling.json"), "w") as fh:
    json.dump({"nested": res, "best": best, "delta_vs_greedy": delta,
               "n_members": len(names), "members": names, "verdict": verdict},
              fh, indent=2)
print("saved cache/d2d6_ceiling.json")
