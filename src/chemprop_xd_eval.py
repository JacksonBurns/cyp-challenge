"""Phase 1 eval (NEXT_ITER_PLAN sec 4) - 4-variant D-MPNN table + GATE 1.

Compares the four chemprop D-MPNN variants on the SAME scaffold folds (seed 7,
ft_data.csv 'fold'), per-isoform OOF Pearson + the repo's honest nested macro R2
(Fisher-z avg of per-fold OOF Pearson, squared, averaged over 4 isoforms - the
exact metric in tabpfn_probe.nested_pool):

  V1  control (challenge-only, no x_d)      cache/ft_oof_xdV1.csv
  V2  aux (challenge + ext aux, no x_d)     cache/ft_oof_dmpnn.csv (CACHED)
  V3  x_d (challenge-only + mordred x_d)    cache/ft_oof_xdV3.csv
  V4  aux + x_d (THE BET)                   cache/ft_oof_xdV4.csv

GATE 1: V4 (or V3) must beat V2 (ft_dmpnn, best existing aux recipe) on nested
macro R2 by >= +0.003 to adopt the x_d channel. Also reports the per-isoform
decomposition of {aux, x_d, both}: V3-V1 = x_d effect (challenge-only), V2-V1 =
aux effect, V4-V2 = x_d effect (on aux), V4-V3 = aux effect (on x_d).

Run (chemeleon env): ~/miniforge3/envs/chemeleon/bin/python src/chemprop_xd_eval.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
K = 5
GATE_DELTA = 0.003

VARIANTS = {
    "V1_control": os.path.join(CACHE, "ft_oof_xdV1.csv"),
    "V2_aux": os.path.join(CACHE, "ft_oof_dmpnn.csv"),
    "V3_xd": os.path.join(CACHE, "ft_oof_xdV3.csv"),
    "V4_aux_xd": os.path.join(CACHE, "ft_oof_xdV4.csv"),
}


def nested_pearson(rs):
    rs = np.clip(rs, -0.999, 0.999)
    return float(np.tanh(np.mean(np.arctanh(rs))))


def nested_for(oof_df, ft, iso):
    y = ft[iso].values
    p = oof_df[iso].values.astype(float)
    folds = ft["fold"].values
    m = ~np.isnan(y) & ~np.isnan(p)
    rs = []
    for f in range(K):
        va = m & (folds == f)
        if va.sum() < 5:
            rs.append(0.0)
            continue
        rs.append(float(pearsonr(y[va], p[va]).statistic))
    return nested_pearson(rs)


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    res = {}
    for name, path in VARIANTS.items():
        if not os.path.exists(path):
            print(f"  MISSING {name}: {path}")
            continue
        df = pd.read_csv(path)
        per_iso = {iso: nested_for(df, ft, iso) for iso in ISO}
        macro = float(np.mean([per_iso[i] ** 2 for i in ISO]))
        res[name] = {"per_iso": per_iso, "macro_R2": macro}
    if len(res) < 4:
        print("Not all 4 variants present; cannot evaluate GATE 1.")
        print(json.dumps({k: {i: round(v['per_iso'][i], 4) for i in ISO} | {"macro": round(v['macro_R2'], 4)} for k, v in res.items()}, indent=2))
        return

    print("\n=== PHASE 1: 4-variant D-MPNN (scaffold folds seed 7) ===", flush=True)
    header = "variant".ljust(14) + "".join(i.ljust(9) for i in ISO) + "MACRO_R2"
    print(header, flush=True)
    for name in ["V1_control", "V2_aux", "V3_xd", "V4_aux_xd"]:
        r = res[name]
        row = name.ljust(14) + "".join(f"{r['per_iso'][i]:.4f}".ljust(9) for i in ISO) + f"{r['macro_R2']:.4f}"
        print(row, flush=True)

    print("\n=== DECOMPOSITION (per isoform, nested Pearson) ===", flush=True)
    for i in ISO:
        v1 = res["V1_control"]["per_iso"][i]
        v2 = res["V2_aux"]["per_iso"][i]
        v3 = res["V3_xd"]["per_iso"][i]
        v4 = res["V4_aux_xd"]["per_iso"][i]
        print(f"  {i}: aux={v2-v1:+.4f}  x_d={v3-v1:+.4f}  x_d|aux={v4-v2:+.4f}  aux|x_d={v4-v3:+.4f}  (V1 {v1:.4f} V2 {v2:.4f} V3 {v3:.4f} V4 {v4:.4f})", flush=True)

    print("\n=== GATE 1 (V4 or V3 must beat V2 by >= +0.003 nested macro R2) ===", flush=True)
    v2m = res["V2_aux"]["macro_R2"]
    v3m = res["V3_xd"]["macro_R2"]
    v4m = res["V4_aux_xd"]["macro_R2"]
    print(f"  V2 (aux, baseline):  {v2m:.4f}", flush=True)
    print(f"  V3 (x_d):            {v3m:.4f}  (vs V2 {v3m-v2m:+.4f})", flush=True)
    print(f"  V4 (aux+x_d):        {v4m:.4f}  (vs V2 {v4m-v2m:+.4f})", flush=True)
    best = max(v3m, v4m)
    best_name = "V4_aux_xd" if v4m >= v3m else "V3_xd"
    gate1 = best - v2m >= GATE_DELTA
    print(f"  best x_d variant: {best_name} {best:.4f}  delta vs V2 {best-v2m:+.4f} (need >= +{GATE_DELTA})", flush=True)
    print(f"  GATE 1: {'PASS' if gate1 else 'NOT MET'}", flush=True)
    if not gate1:
        print("  (x_d channel does NOT clear the gate on scaffold folds; "
              "do not adopt it - log and move to Phase 2 with the existing pool)", flush=True)

    out = {
        "variants": {k: {i: round(v["per_iso"][i], 4) for i in ISO} | {"macro_R2": round(v["macro_R2"], 4)} for k, v in res.items()},
        "gate1": {
            "V2_macro": round(v2m, 4),
            "V3_macro": round(v3m, 4),
            "V4_macro": round(v4m, 4),
            "best_xd_variant": best_name,
            "best_xd_macro": round(best, 4),
            "delta_vs_V2": round(best - v2m, 4),
            "threshold": GATE_DELTA,
            "pass": bool(gate1),
        },
        "folds": "ft_data.csv fold col (scaffold seed 7, K=5)",
        "recipe": "per-isoform nested = Fisher-z avg of per-fold OOF Pearson; macro R2 = mean(per-isoform nested^2)",
    }
    with open(os.path.join(CACHE, "chemprop_xd_eval.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved chemprop_xd_eval.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
