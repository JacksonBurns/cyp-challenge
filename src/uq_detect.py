"""UQ arm (Oct 25 directive): per-molecule uncertainty to DETECT
issue-causing molecules. CPU-only, uses existing OOF artifacts only.

Two uncertainty sources available at zero training cost:
 A) SEED SPREAD: families trained with multiple seeds (d2d6_s8/9/10,
    dscst_s1/2, ext_s1..5) -> per-molecule std of OOF preds across seeds
    = epistemic proxy.
 B) CROSS-FAMILY DISAGREEMENT: std of OOF preds across ALL families per
    molecule = ensemble-disagreement proxy (includes model-form uncertainty).

GATE (pre-registered): per isoform, does spread flag high-residual molecules?
 - Spearman(spread, |residual|) over labeled rows,
 - AUC(spread -> top-quartile residual),
 - decile lift: mean |residual| in top spread decile / overall.
PASS if AUC >= 0.62 on >= 2 isoforms for source B (chance 0.5).

Follow-on (GPU, separate arm): UQ-weighted loss - downweight top-decile
high-spread training rows on the 2D6 arm; gate = CV ST-RAE improvement.
Latent-embedding OOD (MPNN fingerprint kNN distance) is a later GPU step.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

CACHE = "/home/jackson/cyp-challenge/cache"
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
MIN_REP = 3  # seeds needed for seed-spread


def load_fams():
    fams = {}
    for f in sorted(glob.glob(os.path.join(CACHE, "ft_oof_*.csv"))):
        base = os.path.basename(f)[len("ft_oof_"):-len(".csv")]
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "SMILES" not in df.columns:
            continue
        fams[base] = df.set_index("SMILES")
    return fams


def seed_groups(fams):
    """family -> list of OOF frames incl seed variants (base name stem)."""
    groups = {}
    for name, df in fams.items():
        stem = name.split("_s")[0] if "_s" in name else name
        # careful: names like 'hinge2' have no _s; 'ext_s3' -> 'ext'; 'mv32_s1' -> 'mv32'
        groups.setdefault(stem, []).append(df)
    return {k: v for k, v in groups.items() if len(v) >= MIN_REP}


def main():
    fams = load_fams()
    data = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    lab = data.drop_duplicates("SMILES").set_index("SMILES")
    print(f"{len(fams)} family OOF files", flush=True)

    out = {}
    # per-isoform accumulators
    seed_spread = {}   # iso -> DataFrame SMILES x fam (std across seeds)
    cross_spread = {}  # iso -> Series SMILES (std across families)
    cross_mean = {}    # iso -> Series mean OOF across families
    sgroups = seed_groups(fams)
    print("seed-replicate families:", {k: len(v) for k, v in sgroups.items()},
          flush=True)

    for iso in ISO:
        cols = {n: df[iso] for n, df in fams.items() if iso in df.columns}
        M = pd.DataFrame(cols)  # SMILES x family
        labiso = lab[iso].reindex(M.index)
        labiso = labiso.dropna()
        Mo = M.loc[labiso.index]
        # drop families that are mostly-NaN on labeled rows
        good = Mo.columns[Mo.notna().mean() > 0.5]
        Mo = Mo[good]
        # cross-family disagreement (nan-aware)
        cs = Mo.std(axis=1, skipna=True)
        cm = Mo.mean(axis=1, skipna=True)
        cross_spread[iso], cross_mean[iso] = cs, cm
        # seed spread: mean over seed-groups of within-group std
        sstd = {}
        for stem, frames in sgroups.items():
            sub = [f[iso] for f in frames if iso in f.columns]
            if len(sub) < MIN_REP:
                continue
            S = pd.DataFrame(sub).loc[labiso.index]
            sstd[stem] = S.std(axis=1, skipna=True)
        seed_spread[iso] = pd.DataFrame(sstd)

        y = labiso.astype(float)
        resid = (y - cm).abs()
        rec = {"n": int(len(y)), "n_families": int(len(good))}
        for src, uq in [("cross_family", cs),
                        ("seed_spread", seed_spread[iso].mean(axis=1, skipna=True))]:
            m = uq.notna() & resid.notna()
            if m.sum() < 200:
                continue
            u, r = uq[m], resid[m]
            thr = r.quantile(0.75)
            top = (r >= thr).astype(int)
            if top.std() == 0:
                continue
            rec[src] = {
                "n_eval": int(m.sum()),
                "spearman_uq_resid": round(float(spearmanr(u, r).statistic), 4),
                "auc_topquartile": round(float(roc_auc_score(top, u)), 4),
                "lift_top_decile": round(
                    float(r[u >= u.quantile(0.9)].mean() / r.mean()), 3),
            }
        out[iso] = rec
        print(iso, json.dumps(rec), flush=True)

    # persist per-molecule UQ for the 2D6 follow-on (loss-weighting arm)
    keep = pd.DataFrame({
        f"{iso}_uq": cross_spread[iso].rename(None) for iso in ISO
    })
    keep.index.name = "SMILES"
    keep.to_csv(os.path.join(CACHE, "uq_cross_spread.csv"))
    ss = pd.DataFrame({f"{iso}_seedstd": seed_spread[iso].mean(axis=1, skipna=True)
                       for iso in ISO})
    ss.index.name = "SMILES"
    ss.to_csv(os.path.join(CACHE, "uq_seed_spread.csv"))

    npass = 0
    for iso in ISO:
        for src in ("cross_family", "seed_spread"):
            if src in out[iso] and out[iso][src].get("auc_topquartile", 0) >= 0.62:
                npass += 1
    out["GATE"] = {"pass_if_auc_ge_0.62_on_2_isoforms": npass >= 2, "n_pass": npass}
    print("GATE:", json.dumps(out["GATE"]), flush=True)
    with open(os.path.join(CACHE, "uq_detect.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
