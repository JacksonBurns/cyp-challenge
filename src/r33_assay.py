"""R3-3: assay-condition (measurement physics) features.

Hypothesis: apparent pIC50 shifts with assay protocol (enzyme source
recombinant vs baculovirus vs microsome, assay format, detection).
ChEMBL assay_description/assay_type carries protocol text for the
external CYP rows; train rows have NO protocol metadata (challenge
gave none), so the channel is: (a) predict protocol-class from
chemistry on external rows -> soft features for challenge, (b) direct
test: do protocol-class probabilities explain replicate-pair label
deltas (sigma .65 for 2D6)?

Steps:
 1. parse protocol tokens from assay_description on external CYP rows
 2. GBM chemistry->protocol-class (multi-label), predict on challenge
 3. A3' gate: stack+PHARM+A2 vs +protocol feats (GBM MAE, 2 seeds)
 4. replicate-delta diagnostic (2D6 pairs): protocol distance vs |dy|
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)

PROTO = {
    "baculovirus": r"baculo|sf9|insect",
    "microsome": r"microsome|microsomal|membrane",
    "recombinant": r"recombinant",
    "yeast": r"yeast|s\.? cerevis",
    "escherichia": r"escherichia|e\.?\s?coli",
    "mammalian": r"mammal|hepatocyte|human liver",
    "ec_luc": r"luciferase|luc",
    "fluorescence": r"fluoresc|oxidation resofin|resorufin|testosterone",
    "absorbance": r"absorb|formazan|nadh",
    "lc_ms": r"lc.?ms|liquid chromatograph|mass spec",
    "whole_cell": r"whole.?cell|cell.?based",
    "in_vitro": r"in vitro",
}
RPAT = {k: re.compile(v, re.I) for k, v in PROTO.items()}


def main():
    df = pd.read_csv("external/chembl_cyp_ic50.csv")
    d = df["assay_description"].fillna("")
    Y = np.stack([[1.0 if rp.search(t) else 0.0 for rp in RPAT.values()] for t in d])
    names = list(RPAT)
    prev = Y.mean(0)
    print("protocol prevalence (external):")
    for n, p in zip(names, prev):
        print(f"  {n:12s} {p:.3f}")
    keep = [i for i, p in enumerate(prev) if 0.02 < p < 0.98]
    print("kept", len(keep), "classes")
    np.save(os.path.join(CACHE, "proto_prev.npy"), Y[:, keep])
    # replicate-delta diagnostic on 2D6
    key = df["canonical_smiles"].astype(str)
    tgt = df["standard_potency"].astype(float) / 1e3  # -> pIC50 approx
    grp = pd.DataFrame({"s": key, "y": tgt, "p": Y[:, keep].tolist()}).dropna()
    dy, pd_ = [], []
    for s, g in grp.groupby("s"):
        if len(g) > 1:
            ys = g["y"].values
            for i in range(len(ys)):
                for j in range(i + 1, len(ys)):
                    dy.append(abs(ys[i] - ys[j]))
                    a, b = np.array(g["p"].values[i]), np.array(g["p"].values[j])
                    pd_.append(float(np.abs(a - b).sum()))
    dy, pd_ = np.array(dy), np.array(pd_)
    print(f"2D6-like replicate pairs: n={len(dy)} median|dy|={np.median(dy):.2f}")
    for thr in [0, 0.5, 1.5, 3]:
        m = pd_ <= thr
        if m.sum() > 5:
            print(f"  protocol-dist <= {thr}: n={m.sum()} median|dy|={np.median(dy[m]):.2f}")
        m2 = pd_ > thr
        if m2.sum() > 5:
            print(f"  protocol-dist >  {thr}: n={m2.sum()} median|dy|={np.median(dy[m2]):.2f}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
