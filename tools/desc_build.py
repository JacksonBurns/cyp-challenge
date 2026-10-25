"""3D/flexibility/planarity descriptor arm: build, correlate, GBM OOF.

Descriptors: RDKit 2D (counts/topological) + 3D shape from ETKDG+MMFF
conformers (Rg, asphericity, PMI ratios, Rg-spread = flexibility) +
invented chemistry descriptors (planar-rigid frac, pyramidal-N,
atropisomer strain, flex-efficiency, sp3-depth, pi-face ratio,
sigma-hole halogens, acidic-scaffold flags).

Outputs: cache/desc3d_{train,test}.csv, cache/zoo_eval_dsc3d.json,
cache/ft_oof_dsc3d.csv, cache/ft_test_preds_dsc3d.csv (gate-ready).
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (Descriptors, rdMolDescriptors, Crippen,
                       rdPartialCharges, rdDistGeom, rdMolTransforms,
                       rdMolAlign)

RDLogger.DisableLog("rdApp.*")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
NCONF = 5

D2 = ["MolWt", "MolLogP", "TPSA", "NumRotatableBonds", "RingCount",
      "NumAromaticRings", "FractionCSP3", "NumHAcceptors", "NumHDonors",
      "NumHeteroatoms", "BertzCT", "NumAmideBonds", "HeavyAtomCount",
      "NHOHCount", "NOCount", "FormalCharge", "MolMR", "LabuteASA",
      "BalabanJ", "MaxPartialCharge", "MinPartialCharge"]


def sp3_carbons(m):
    return sum(1 for a in m.GetAtoms()
               if a.GetSymbol() == "C"
               and str(a.GetHybridization()) == "SP3")


def planar_frac(m):
    n = m.GetNumHeavyAtoms()
    if not n:
        return 0.0
    npl = sum(1 for a in m.GetAtoms()
              if str(a.GetHybridization()) in ("SP2", "SP")
              or a.GetIsAromatic())
    return npl / n


def pyramidal_N(m):
    return sum(1 for a in m.GetAtoms()
               if a.GetSymbol() == "N" and a.GetDegree() == 3
               and str(a.GetHybridization()) == "SP3"
               and a.GetFormalCharge() == 0)


def atrop_bonds(m):
    """biaryl single bonds with >=1 ortho substituent on each ring."""
    n = 0
    ri = m.GetRingInfo()
    rings = ri.AtomRings()

    def ring_of(idx):
        for r in rings:
            if idx in r:
                return set(r)
        return set()
    for b in m.GetBonds():
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        if (b.GetBondType() == Chem.BondType.SINGLE
                and a.GetIsAromatic() and c.GetIsAromatic()
                and not ri.NumBondRings(b.GetIdx())):
            ra, rc = ring_of(a.GetIdx()), ring_of(c.GetIdx())
            oa = sum(1 for nb in a.GetNeighbors()
                     if nb.GetIdx() not in ra and nb.GetIdx() != c.GetIdx())
            oc = sum(1 for nb in c.GetNeighbors()
                     if nb.GetIdx() not in rc and nb.GetIdx() != a.GetIdx())
            if oa and oc:
                n += 1
    return n


def sp3_depth(m):
    """longest acyclic chain of consecutive sp3 atoms."""
    best = 0
    seen = set()
    for a in m.GetAtoms():
        if str(a.GetHybridization()) != "SP3" or a.GetIdx() in seen:
            continue
        stack, comp, size = [a], [], 0
        seen.add(a.GetIdx())
        while stack:
            x = stack.pop()
            comp.append(x)
            for nb in x.GetNeighbors():
                if (str(nb.GetHybridization()) == "SP3"
                        and nb.GetIdx() not in seen
                        and not m.GetBondBetweenAtoms(
                            x.GetIdx(), nb.GetIdx()).IsInRing()):
                    seen.add(nb.GetIdx())
                    stack.append(nb)
        # longest path in comp (tree) via 2x BFS
        def far(s):
            dist = {s: 0}
            q = [s]
            while q:
                u = q.pop()
                for nb in u.GetNeighbors():
                    if nb in comp and nb.GetIdx() not in dist:
                        dist[nb] = dist[u] + 1
                        q.append(nb)
            return max(dist, key=dist.get), max(dist.values())
        u, _ = far(comp[0])
        _, d = far(u)
        best = max(best, d + 1)
    return best


def flex_eff(m):
    rot = Descriptors.NumRotatableBonds(m)
    rings = rdMolDescriptors.CalcNumRings(m)
    return rot / (1.0 + rings)


def sigma_halogens(m):
    return sum(1 for a in m.GetAtoms()
               if a.GetSymbol() in ("Cl", "Br", "I") and a.GetIsAromatic())


def acid_flags(m):
    pats = ["[S](=O)(=O)N", "C(=O)O", "c1nnnn1", "S(=O)C(F)(F)F",
            "c1cc(N=NN=N1)", "P(=O)(O)O", "C(=O)[N-]S(=O)(=O)"]
    return sum(1 for p in pats
               if m.HasSubstructMatch(Chem.MolFromSmarts(p)))


def ring_planarity(m):
    ss = m.GetRingInfo().AtomRings()
    if not ss:
        return np.nan
    best = 0.0
    conf = m.GetConformer()
    for ring in ss:
        if len(ring) < 5:
            continue
        tors = []
        L = len(ring)
        for i in range(L):
            a = [ring[j] for j in
                 range(i % L, (i + 4) % L + (L if (i + 4) % L == i % L else 0))][:4]
            if len(a) < 4:
                continue
            try:
                t = abs(rdMolTransforms.GetDihedralDeg(conf, *a))
                tors.append(min(t, 180 - t))
            except Exception:
                continue
        if tors:
            best = max(best, float(np.mean(tors)))
    return best


def shape_3d(m, nconf=NCONF):
    mh = Chem.AddHs(m)
    cids = rdDistGeom.EmbedMultipleConfs(mh, nconf,
                                         randomSeed=42, numThreads=1)
    if len(cids) == 0:
        return {}
    ff = None
    try:
        rdMolAlign.AlignMolConformers(mh)
    except Exception:
        pass
    rgs, pmi, maxd = [], [], []
    for cid in cids:
        c = mh.GetConformer(cid)
        pts = np.array(c.GetPositions())
        w = mh.GetConformer()
        cg = pts.mean(0)
        d2 = ((pts - cg) ** 2).sum(1)
        rgs.append(float(np.sqrt(d2.mean())))
        cov = (pts - cg).T @ (pts - cg) / max(len(pts), 1)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        s1, s2, s3 = ev
        tot = s1 + s2 + s3
        if tot > 1e-9:
            pmi.append((1.5 * (s2 + s3) / tot,
                        1.5 * (s1 + s3) / tot,
                        1.5 * (s1 + s2) / tot,
                        (s1 - 0.5 * (s2 + s3)) / tot))
        maxd.append(float(np.sqrt(((pts[:, None, :] - pts[None, :, :])
                                   ** 2).sum(-1)).max()))
    pmi = np.array(pmi)
    return {
        "rg": float(np.mean(rgs)),
        "rg_spread": float(np.std(rgs)),          # flexibility proxy
        "rg_spread_n": float(np.std(rgs) / np.mean(rgs)),
        "maxd": float(np.mean(maxd)),
        "pmi_asym": float(np.mean(pmi[:, 3])),    # asphericity
        "pmi_len": float(np.mean(pmi[:, 0])),
        "pmi_wid": float(np.mean(pmi[:, 1])),
        "pmi_thr": float(np.mean(pmi[:, 2])),
        "planar3d": float(np.mean(pmi[:, 3])),    # ~0 => planar disk
    }


def build(smiles_list):
    rows = []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        rec = {"SMILES": s}
        if m is None:
            rows.append(rec)
            continue
        for nm in D2:
            try:
                rec[nm] = getattr(Descriptors, nm)(m)
            except Exception:
                rec[nm] = np.nan
        rdPartialCharges.ComputeGasteigerCharges(m)
        rec["sp3C"] = sp3_carbons(m)
        rec["planar_frac"] = planar_frac(m)
        rec["pyrN"] = pyramidal_N(m)
        rec["atrop"] = atrop_bonds(m)
        rec["sp3_depth"] = sp3_depth(m)
        rec["flex_eff"] = flex_eff(m)
        rec["sigmaHal"] = sigma_halogens(m)
        rec["acid_flags"] = acid_flags(m)
        rec["numFrags"] = len(Chem.GetMolFrags(m))
        rec["aromRingsPerHeavy"] = (rdMolDescriptors.CalcNumAromaticRings(m)
                                    / max(m.GetNumHeavyAtoms(), 1))
        try:
            m2 = Chem.AddHs(m)
            cid = rdDistGeom.EmbedMultipleConfs(m2, 1, randomSeed=7)
            rec["ring_plan"] = (ring_planarity(m2)
                                if len(cid) else np.nan)
        except Exception:
            rec["ring_plan"] = np.nan
        try:
            rec.update(shape_3d(m))
        except Exception:
            pass
        rows.append(rec)
        if (i + 1) % 500 == 0:
            print(f"desc {i+1}/{len(smiles_list)}", flush=True)
    return pd.DataFrame(rows)


def main():
    fd = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(HERE, "..", "data",
                                    "cyp-challenge-TEST-BLINDED.csv"))
    tr = build(fd["SMILES"].tolist())
    te = build(test["SMILES"].drop_duplicates().tolist())
    tr.to_csv(os.path.join(CACHE, "desc3d_train.csv"), index=False)
    te.to_csv(os.path.join(CACHE, "desc3d_test.csv"), index=False)
    print("built", tr.shape, te.shape, flush=True)


if __name__ == "__main__":
    main()
