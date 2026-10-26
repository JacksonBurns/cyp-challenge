"""Hard-molecule chemistry, part 2 (per-isoform contrasts, scaffolds, AD).

1. Per-isoform: HARD vs EASY means on the RDKit profile (within isoform).
2. Murcko scaffold enrichment in HARD (all isoforms pooled) vs background.
3. Applicability domain: Tanimoto (ECFP4) distance to nearest TRAINING
   ACTIVE (y >= median) of the same isoform -- are HARD molecules far
   from known actives (novel space) or near-neighbor ambiguity?
4. 2D6 pharmacophore test: HARD 2D6 rows WITHOUT a basic amine?
   1A2 planarity, 2C9 acidity contrasts.
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402
from rdkit import Chem, RDLogger, DataStructs  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator as rfg  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")
CACHE = H.CACHE
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
GEN = rfg.GetMorganGenerator(radius=2, fpSize=2048)

prof = pd.read_csv(os.path.join(CACHE, "chem_groups.csv"), index_col=0)

# rebuild per-row table quickly (same recipe as chem_profile)
import json
W = json.load(open(os.path.join(CACHE, "hinge_w_zoo6.json")))
smi = H.Xtr["SMILES"].values
rows = []
for iso in ISO:
    P = np.zeros(len(smi))
    tot = np.zeros(len(smi))
    for fam, w in W[iso]["w"].items():
        if not isinstance(w, (int, float)) or abs(w) < 1e-9:
            continue
        fo = H.raw_oof_fam(fam)
        col = iso if iso in fo.columns else f"{iso}_pIC50_direct_inhibition"
        v = fo[col].reindex(smi).values.astype(float)
        ok = ~np.isnan(v)
        P[ok] += w * v[ok]
        tot[ok] += w
    P = P / np.maximum(tot, 1e-9)
    y = H.targets[f"{iso}_pIC50_direct_inhibition"].reindex(smi).values.astype(float)
    flr = np.percentile(y[~np.isnan(y)], 1.5)
    for i, s in enumerate(smi):
        if np.isnan(y[i]):
            continue
        r = abs(y[i] - P[i])
        grp = ("FLOOR" if y[i] <= flr else "HARD" if r >= 1.0
               else "EASY" if r <= 0.25 else "MID")
        rows.append((s, iso, y[i], P[i], r, grp))
t = pd.DataFrame(rows, columns=["SMILES", "iso", "y", "P", "res", "grp"])

num = [c for c in prof.columns if c not in ("SMILES", "molgrp")]
tp = t.set_index("SMILES").join(prof[num], how="left")

print("== per-isoform HARD(+) vs EASY(-) contrast (top discriminating feats) ==")
for iso in ISO:
    sub = tp[tp["iso"] == iso]
    h = sub[sub.grp == "HARD"][num].astype(float).mean()
    e = sub[sub.grp == "EASY"][num].astype(float).mean()
    d = (h - e).abs().sort_values(ascending=False)
    print(f"\n{iso}: n_hard={(sub.grp=='HARD').sum()} n_easy={(sub.grp=='EASY').sum()}")
    for f in d.index[:8]:
        print(f"   {f:18s} hard={h[f]:7.2f} easy={e[f]:7.2f}")

# scaffold enrichment
t["scaf"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False)
             for s in t.SMILES]
hard_s = Counter(t[t.grp == "HARD"].scaf)
all_s = Counter(t.scaf)
print("\n== top HARD scaffolds (count, hard-rate vs base) ==")
base = (t.grp == "HARD").mean()
for s, n in hard_s.most_common(15):
    print(f"  {n:4d}  rate={n/max(all_s[s],1):.2f} (base {base:.2f})  {s[:60]}")

# applicability domain: nearest training ACTIVE (same iso) Tanimoto
mols = {s: Chem.MolFromSmiles(s) for s in t.SMILES.unique()}
fps = {s: GEN.GetFingerprint(mols[s]) for s in mols if mols[s] is not None}
print("\n== AD: min-Tanimoto to same-isoform actives (median split) ==")
rng = np.random.default_rng(0)
for iso in ISO:
    sub = t[t["iso"] == iso]
    act = sub[sub.y >= sub.y.median()].SMILES.unique()
    act = [s for s in act if s in fps]
    afps = [fps[s] for s in act]
    res = {"HARD": [], "EASY": [], "FLOOR": [], "MID": []}
    gmap = dict(zip(sub.SMILES, sub.grp))
    seen = set()
    for s in sub.SMILES:
        if s in seen:
            continue
        seen.add(s)
        g = gmap[s]
        if s not in fps:
            continue
        mt = max(DataStructs.BulkTanimotoSimilarity(fps[s], afps))
        res[g].append(mt)
    for g in ["HARD", "EASY", "FLOOR", "MID"]:
        if res[g]:
            print(f"  {iso} {g:6s} n={len(res[g]):4d} median_maxTanimoto={np.median(res[g]):.3f}")
