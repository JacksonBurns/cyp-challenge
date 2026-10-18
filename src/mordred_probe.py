"""Phase 0 (NEXT_ITER_PLAN sec 3) - cheap tabular pre-filter for the descriptor
channel, BEFORE any GPU.

Compare three ridge models on the SAME scaffold folds (ft_data.csv fold col,
seed 0) per isoform, reporting per-isoform OOF Pearson and the repo's honest
nested macro R2 (Fisher-z avg of per-fold OOF Pearson, squared, averaged over
the 4 isoforms - the exact metric in tabpfn_probe.nested_pool / monroe_ridge):
  (A) embedding only            cpmed 600-d
  (B) embedding + mordred       cpmed 600-d concat kept mordred descriptors
  (C) mordred only              kept mordred descriptors

GATE 0 (go to Phase 1 x_d work only if):
  (B) beats (A) on nested macro R2 by >= +0.003 AND on >= 2 of 4 isoforms.
If (B) does NOT clear the gate, the tabular (ridge) signal is weak, but a
trained x_d FFN can still capture nonlinear interactions ridge cannot - so do
NOT kill the lever; go to Phase 1 and let the trained x_d chemprop be the real
test. Log the probe numbers either way.

Anti-leak: scaler + ridge fit on the TRAIN fold's labeled rows only (never the
validation fold); mordred matrix already imputed/standardized on TRAIN only.

Run (mordred env): ~/miniforge3/envs/mordred/bin/python src/mordred_probe.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
ALPHAS = [100.0, 1000.0, 10000.0]
K = 5
GATE_MACRO_DELTA = 0.003
GATE_N_ISOFORMS = 2


def nested_pearson(rs):
    rs = np.clip(rs, -0.999, 0.999)
    return float(np.tanh(np.mean(np.arctanh(rs))))


def oof_variant(X, y, folds, m):
    """5-fold OOF ridge with per-fold scaler+ridge fit on train-fold rows only.
    Returns per-fold Pearson list and the OOF array."""
    oof = np.full(len(y), np.nan)
    rs = []
    for f in range(K):
        trn = m & (folds != f)
        va = m & (folds == f)
        if va.sum() < 5 or trn.sum() < 10:
            rs.append(0.0)
            continue
        sc = StandardScaler().fit(X[trn])
        Xs_trn = sc.transform(X[trn])
        Xs_va = sc.transform(X[va])
        r = RidgeCV(alphas=ALPHAS)
        r.fit(Xs_trn, y[trn])
        pred = r.predict(Xs_va)
        oof[va] = pred
        rs.append(float(pearsonr(y[va], pred).statistic))
    return oof, rs


def main():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    smi = ft["SMILES"].values
    folds = ft["fold"].values

    cpmed = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet"))
    if "SMILES" in cpmed.columns:
        cpmed = cpmed.set_index("SMILES")
    E_cp = cpmed.reindex(smi).values.astype(np.float64)

    mord = pd.read_parquet(os.path.join(CACHE, "emb_mordred_all.parquet"))
    if "SMILES" in mord.columns:
        mord = mord.set_index("SMILES")
    elif mord.index.name is None:
        mord = mord.reset_index().set_index("SMILES")
    E_md = mord.reindex(smi).values.astype(np.float64)

    print(f"cpmed {E_cp.shape}  mordred {E_md.shape}  (n={len(smi)})", flush=True)

    XA = E_cp
    XB = np.hstack([E_cp, E_md])
    XC = E_md

    variants = {"A_emb": XA, "B_emb+mordred": XB, "C_mordred": XC}

    per_iso = {v: {} for v in variants}
    macro = {v: {} for v in variants}
    oof_store = {v: {i: None for i in ISO} for v in variants}

    for iso in ISO:
        y = ft[iso].values
        m = ~np.isnan(y)
        line = f"  {iso} (n={m.sum()})"
        for v, X in variants.items():
            _, rs = oof_variant(X, y, folds, m)
            nested = nested_pearson(rs)
            per_iso[v][iso] = nested
            macro[v][iso] = nested ** 2
            oof_store[v][iso] = None
            line += f"  | {v} pearson {nested:.4f}"
        print(line, flush=True)

    # macro R2 = mean of per-isoform nested^2
    print("\n=== NESTED MACRO R2 (mean of per-isoform nested^2) ===", flush=True)
    header = "variant".ljust(18) + "".join(i.ljust(10) for i in ISO) + "MACRO".ljust(8)
    print(header, flush=True)
    res = {}
    for v in variants:
        mac = float(np.mean([macro[v][i] for i in ISO]))
        res[v] = {i: round(per_iso[v][i], 4) for i in ISO}
        res[v]["macro_R2"] = round(mac, 4)
        row = v.ljust(18) + "".join(f"{per_iso[v][i]:.4f}".ljust(10) for i in ISO) + f"{mac:.4f}".ljust(8)
        print(row, flush=True)

    # GATE 0
    delta_macro = res["B_emb+mordred"]["macro_R2"] - res["A_emb"]["macro_R2"]
    n_iso_beat = sum(
        1 for i in ISO if per_iso["B_emb+mordred"][i] > per_iso["A_emb"][i]
    )
    gate0 = (delta_macro >= GATE_MACRO_DELTA) and (n_iso_beat >= GATE_N_ISOFORMS)
    print(f"\n=== GATE 0 ===", flush=True)
    print(f"  (B)-(A) nested macro R2 delta: {delta_macro:+.4f} (need >= +{GATE_MACRO_DELTA})", flush=True)
    print(f"  (B) beats (A) on {n_iso_beat}/4 isoforms (need >= {GATE_N_ISOFORMS})", flush=True)
    for i in ISO:
        d = per_iso["B_emb+mordred"][i] - per_iso["A_emb"][i]
        print(f"    {i}: {per_iso['A_emb'][i]:.4f} -> {per_iso['B_emb+mordred'][i]:.4f}  ({d:+.4f})", flush=True)
    print(f"  GATE 0: {'PASS' if gate0 else 'NOT MET'}", flush=True)
    if not gate0:
        print("  (weak tabular signal - do NOT kill the lever; Phase 1 trained x_d is the real test)", flush=True)

    # save OOF preds for the three variants (for reference / Phase 2 pool use)
    for v, X in variants.items():
        tag = v.replace("+", "_")
        oof = pd.DataFrame(index=ft.index)
        oof.insert(0, "SMILES", smi)
        for iso in ISO:
            y = ft[iso].values
            m = ~np.isnan(y)
            oo, _ = oof_variant(X, y, folds, m)
            oof[iso] = oo
        oof.to_csv(os.path.join(CACHE, f"ft_oof_probe_{tag}.csv"), index=False)

    out = {
        "variants": {v: res[v] for v in variants},
        "gate0": {
            "delta_macro_B_minus_A": round(delta_macro, 4),
            "n_isoforms_B_beats_A": n_iso_beat,
            "macro_delta_threshold": GATE_MACRO_DELTA,
            "n_isoform_threshold": GATE_N_ISOFORMS,
            "pass": bool(gate0),
        },
        "embedding": "cpmed 600-d (emb_cpmed_all.parquet)",
        "mordred": "kept descriptors from emb_mordred_all.parquet",
        "folds": "ft_data.csv fold col (scaffold seed 0, K=5)",
        "recipe": "per-fold StandardScaler+RidgeCV([100,1000,10000]) fit on train-fold labeled rows; nested = Fisher-z avg of per-fold OOF Pearson; macro R2 = mean(per-isoform nested^2)",
    }
    with open(os.path.join(CACHE, "mordred_probe.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved mordred_probe.json + ft_oof_probe_{{A_emb,B_emb_mordred,C_mordred}}.csv", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
