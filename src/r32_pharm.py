"""R3-2: pharmacophore GEOMETRY features (2D graph distances).

The descriptor stack has counts of groups but NO distance features.
Canonical CYP pharmacophores are geometric:
 2D6: cationic N <-> HBA (ArO/N) at 5-7 A (Arg132/Glu244); N<->aromatic
 2C9: carboxylate/anionic <-> HBA/aromatic; hydroxamate<->aromatic
 1A2: HBA <-> planar fused aromatic
 3A4: hydrophobe spacing
Heme chelators (pyridyl/imidazole/nitrile/hydroxamate) flagged.

Features: for each center-pair type: min topological distance,
|dist - ideal| band distance, count of pairs inside band, has_pair.
~16 features, deterministic, test-computable (100% coverage).

GATE: GBM pIC50 MAE 4 seeds (stack vs +PHARM vs A2+PHARM).
"""
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)

RDLogger.DisableLog("rdApp.*")

CENTER_SMARTS = {
    # positive ionizable (2D6 cationic N)
    "catN": "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$([N+]);!$(N=a);!$(N#C)]",
    "Nplus": "[N+]",
    # HBA centers (alkoxy O, N-oxide, pyridyl N, sulfoxide S)
    "hba": "[OX2;!$(O=C);!$(Oc1ccccc1)]",
    "pyrN": "n",
    # negative ionizable
    "anion": "[CX3](=[OX1])[OX2H1,OX1-]",
    "sulfonimide": "[#6][SX4](=[OX1])(=[OX1])[NX3H][#6]|[NX3][SX4](=[OX1])(=[OX1])[CX3](=[OX1])",
    "tetrazole": "c1nnn[nH]1",
    "hydroxamate": "[CX3](=[OX1])[OX2H]",
    # aromatic ring centroid proxy: any aromatic carbon
    "arom": "a",
    # halogen (2C9 haloaryl)
    "halo": "[F,Cl,Br,I]",
    # heme-chelating motifs
    "pyridine": "c1ccncc1",
    "imidazole": "c1c[nH]cn1",
    "nitrile": "C#N",
}
PAT = {k: Chem.MolFromSmarts(v) for k, v in CENTER_SMARTS.items()}
PAT = {k: v for k, v in PAT.items() if v is not None}

# pair specs: (a, b, ideal_min_bonds, ideal_max_bonds)
PAIRS = [
    ("catN", "hba", 3, 8),
    ("catN", "pyrN", 3, 8),
    ("catN", "arom", 2, 9),
    ("anion", "arom", 2, 7),
    ("anion", "hba", 3, 8),
    ("hydroxamate", "arom", 2, 8),
    ("pyridine", "arom", 2, 12),
    ("halo", "arom", 0, 4),
]
NF = len(PAIRS) * 4  # min_dist, band_dev, band_count, has_pair
CNAMES = list(PAT)
NC = len(CNAMES)

print("center pats:", len(PAT), "pairs:", len(PAIRS), flush=True)


def pharm_feats(smiles):
    X = np.zeros((len(smiles), NF + NC + 2), np.float32)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        centers = {}
        for k, p in PAT.items():
            hits = set()
            for match in m.GetSubstructMatches(p):
                hits.update(match)
            centers[k] = sorted(hits)
        dm = Chem.GetDistanceMatrix(m)
        d = dm.copy()
        d[~np.isfinite(d)] = 99
        col = 0
        for (a, b, lo, hi) in PAIRS:
            A, B = centers[a], centers[b]
            if a == b:
                sub = dm[np.ix_(A, A)]
                sub = np.where(np.eye(len(A), dtype=bool), 99, sub)
            else:
                sub = dm[np.ix_(A, B)]
            if len(A) and len(B):
                mind = float(np.nanmin(sub))
                inb = float(np.sum((sub >= lo) & (sub <= hi)))
                dev = float(np.mean(np.clip(np.nanmin(np.abs(sub - (lo + hi) / 2.0)), 0, None)))
                X[i, col] = min(mind, 20) / 20.0
                X[i, col + 1] = min(dev, 10) / 10.0
                X[i, col + 2] = min(inb, 10) / 10.0
                X[i, col + 3] = 1.0
            col += 4
        for j, k in enumerate(CNAMES):
            X[i, NF + j] = min(len(centers[k]), 10) / 10.0
        X[i, NF + NC] = float(m.GetNumBonds()) / 50.0
        X[i, NF + NC + 1] = len(Chem.GetMolFrags(m)) / 2.0
    return X


def main():
    import hinge_blend as H
    from sklearn.model_selection import KFold

    Ftr = pharm_feats(H.Xtr["SMILES"].tolist())
    Fte = pharm_feats(H.Xte["SMILES"].tolist())
    np.save(os.path.join(CACHE, "pharm_tr.npy"), Ftr)
    np.save(os.path.join(CACHE, "pharm_te.npy"), Fte)
    print("pharm feats", Ftr.shape, "nonzero frac", (Ftr > 0).mean().round(3), flush=True)

    ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
    Xbase = H.Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    A2 = np.nan_to_num(np.load(os.path.join(CACHE, "adme2", "adme2_feats_tr.npy")), nan=0.0)

    def mae_gate(X, iso, seed):
        y = H.targets[f"{iso}_pIC50_direct_inhibition"].values.astype(np.float32)
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        kf = KFold(5, shuffle=True, random_state=seed)
        pred = np.zeros(len(y))
        for tr, te in kf.split(X):
            g = HistGB(seed)
            g.fit(X[tr], y[tr])
            pred[te] = g.predict(X[te])
        return float(np.mean(np.abs(pred - y)))

    def HistGB(seed):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                             random_state=seed)

    for seed in [0, 1, 2, 3]:
        for iso in ISO:
            m0 = mae_gate(Xbase, iso, seed)
            mp = mae_gate(np.hstack([Xbase, Ftr]), iso, seed)
            map_ = mae_gate(np.hstack([Xbase, A2, Ftr]), iso, seed)
            ma = mae_gate(np.hstack([Xbase, A2]), iso, seed)
            print(f"seed{seed} {iso}: +PHARM {m0-mp:+.4f} | PHARM-on-A2 {ma-map_:+.4f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
