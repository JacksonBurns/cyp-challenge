"""Q3: can an ensemble member's POOL ROLE be predicted from OOF alone?

Observed dissociation: dscst best-standalone -> NEGATIVE pool weight;
intarm worst-standalone -> POSITIVE weight. Pre-registered predictor:
  d_m = corr(residual_m, residual_incumbent-minus-m) over labeled rows.
Expect sign(w_m) == -sign(d_m) (residual-correlated members = bias
correctors get w<=0; orthogonal = decorrelators get w>0).
GATE: Spearman(d_m, w_m) <= -0.5 pooled over members x isoforms x pools.
Inputs: cache/hinge_blend_*.json (weights_fitted_top) + ft_oof_*.csv.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CACHE = "/home/jackson/cyp-challenge/cache"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def oof_for(mem):
    stem = mem[3:] if mem.startswith("ft_") else mem
    fn = os.path.join(CACHE, f"ft_oof_{stem}.csv")
    if not os.path.exists(fn):
        return None
    d = pd.read_csv(fn).drop_duplicates("SMILES").set_index("SMILES")
    return d


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    lab = ft.drop_duplicates("SMILES").set_index("SMILES")
    lab = lab[~lab.index.duplicated(keep="first")]
    rows = []
    for f in sorted(glob.glob(os.path.join(CACHE, "hinge_blend_*.json"))):
        pname = os.path.basename(f)[len("hinge_blend_"):-len(".json")]
        P = json.load(open(f))
        for iso in ISO:
            wmap = P.get(iso, {}).get("weights_fitted_top")
            if not wmap:
                continue
            preds = {}
            for mem in wmap:
                d = oof_for(mem)
                if d is not None and iso in d.columns:
                    preds[mem] = d[iso].astype(float)
            if len(preds) < 5:
                continue
            M = pd.DataFrame(preds).dropna()
            y = lab[iso].reindex(M.index).astype(float)
            m = y.notna()
            M, yv = M[m], y[m].values
            wv = {k: wmap[k] for k in M.columns}
            for mem in M.columns:
                rest = [k for k in M.columns if k != mem]
                wr = np.array([wv[k] for k in rest])
                if np.abs(wr).sum() < 1e-9:
                    continue
                wrn = wr / np.abs(wr).sum()
                inc_r = (M[rest].values * wrn[None, :]).sum(axis=1)
                r_m = yv - M[mem].values
                r_inc = yv - inc_r
                if np.std(r_m) < 1e-9:
                    continue
                rows.append({
                    "pool": pname, "iso": iso, "member": mem,
                    "w": float(wv[mem]),
                    "d_resid_corr": float(np.corrcoef(r_m, r_inc)[0, 1]),
                    "standalone": float(np.sqrt((r_m ** 2).mean())),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        print("NO DATA - check JSON schema / OOF names")
        return
    df.to_csv(os.path.join(CACHE, "q3_poolrole.csv"), index=False)
    print(f"{len(df)} member-iso points, {df['pool'].nunique()} pools",
          flush=True)
    ok = df[df["w"].abs() > 1e-3]
    res = spearmanr(ok["d_resid_corr"], ok["w"])
    print(f"POOLED Spearman(d_resid_corr, w) = {res.statistic:.3f} "
          f"p={res.pvalue:.2e} n={len(ok)}", flush=True)
    for iso in ISO:
        s = ok[ok["iso"] == iso]
        if len(s) >= 8:
            r2 = spearmanr(s["d_resid_corr"], s["w"])
            print(f"  {iso}: rho={r2.statistic:.3f} n={len(s)}", flush=True)
    for mem in ("ft_dscst", "ft_intarm", "ft_zoo_hinge_w3", "ft_mv",
                "ft_ext", "ft_zoo_mae"):
        s = ok[ok["member"] == mem]
        if len(s):
            print(f"  {mem}: mean w={s['w'].mean():+.3f} "
                  f"mean d={s['d_resid_corr'].mean():+.3f} n={len(s)} "
                  f"mean standalone={s['standalone'].mean():.3f}", flush=True)
    gate = bool(res.statistic <= -0.5)
    print("GATE Spearman<=-0.5:", gate, flush=True)
    json.dump({"rho": float(res.statistic), "p": float(res.pvalue),
               "n": int(len(ok)), "gate_pass": gate},
              open(os.path.join(CACHE, "q3_poolrole.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
