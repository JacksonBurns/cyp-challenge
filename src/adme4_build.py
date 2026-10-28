"""ADME4: computed 3D/conformational physics for all 6895 challenge molecules.

Per molecule: ETKDGw multi-conformer sampling + MMFF94 optimization.
Features (no labels needed - deterministic physics, computable on test):
  e_low/nheavy        strain proxy of best conformer
  e_spread            top-lowest energy range (conformational accessibility)
  n_low               conformers within 2 kcal/mol (flexibility)
  strain_relax        E_low - E_fully_relaxed (accessibility of global min)
  sasa_tot, sasa_polar_frac  desolvation proxies (polar atoms via Crippen)
  rg, asphericity, radgyr, prmax, depth (3D shape)
  n_conf_sampled
Output: cache/adme4/d3_feats_{tr,te}.npy (row order = H.Xtr, test CSV).
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdFreeSASA

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
RDLogger.DisableLog("rdApp.*")
BASE = "/home/jackson/cyp-challenge"
CACHE = os.path.join(BASE, "cache")
OUTD = os.path.join(CACHE, "adme4")
os.makedirs(OUTD, exist_ok=True)

NCONF = 40


def polar_atom_frac(m):
    """Fraction of heavy atoms that are polar (N/O/S with Hs, acceptors)."""
    n = m.GetNumHeavyAtoms()
    if n == 0:
        return np.nan
    p = 0
    for a in m.GetAtoms():
        if a.GetAtomicNum() in (7, 8, 16):
            p += 1
    return p / n


def compute_one(smi):
    """Returns (smi, feats) for pool imap."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        try:
            m = Chem.MolFromSmiles(smi, sanitize=False)
            Chem.SanitizeMol(m, catchErrors=True)
        except Exception:
            m = None
    if m is None:
        return smi, np.full(12, np.nan)
    try:
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3()
        p.randomSeed = 0xf00d
        p.numThreads = 1
        cids = AllChem.EmbedMultipleConfs(m, NCONF, p)
        if len(cids) == 0:
            p.useRandomCoords = True
            cids = AllChem.EmbedMultipleConfs(m, NCONF, p)
        if len(cids) == 0:
            return smi, np.full(12, np.nan)
        res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=800, numThreads=1)
        ok = [i for i, (conv, e) in zip(cids, res) if conv != -1]
        if not ok:
            return smi, np.full(12, np.nan)
        E = np.array([res[i][1] for i in ok])
        E.sort()
        nh = m.GetNumHeavyAtoms()
        e_low = E[0] / nh
        spread = (E[min(len(E) - 1, 19)] - E[0]) / nh
        n_low = float(np.sum(E - E[0] <= 2.0))
        # global-min relaxation -> strain of accessible minima
        strain_relax = np.nan
        try:
            m3 = Chem.AddHs(Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(m))))
            cid3 = AllChem.EmbedMolecule(m3, p)
            cv = AllChem.MMFFOptimizeMolecule(m3, maxIters=2000)
            if cv != -1:
                props = AllChem.MMFFGetMoleculeProperties(m3)
                ff = AllChem.MMFFGetMoleculeForceField(m3, props)
                strain_relax = (E[0] - ff.CalcEnergy()) / nh
        except Exception:
            pass
        conf = m.GetConformer(list(cids)[0])
        rg = Descriptors3D.RadiusOfGyration(m)
        asp = Descriptors3D.Asphericity(m)
        prmax = Descriptors3D.NPR1(m)
        # SASA
        sasa_tot = np.nan
        sasa_polar = np.nan
        try:
            ER = {6: 1.7, 7: 1.55, 8: 1.52, 16: 1.8, 9: 1.47, 17: 1.75,
                  15: 1.8, 12: 1.73, 35: 1.85, 53: 2.15, 1: 1.1}
            ar = {at.GetIdx(): ER.get(at.GetAtomicNum(), 1.7) for at in m.GetAtoms()}
            sasa_tot = rdFreeSASA.CalcSASA(m, ar, confIdx=list(cids)[0])
        except Exception:
            pass
        frac_polar = polar_atom_frac(m)
        sasa_polar = sasa_tot * frac_polar if np.isfinite(sasa_tot) else np.nan
        depth = rdMolDescriptors.CalcNumRotatableBonds(m) / max(nh, 1)
        return smi, np.array([e_low, spread, n_low, strain_relax, sasa_tot,
                              sasa_polar, frac_polar, rg, asp, prmax, depth, nh])
    except Exception:
        return smi, np.full(12, np.nan)


def main():
    import hinge_blend as H
    smi_tr = H.Xtr["SMILES"].values
    test = pd.read_csv(os.path.join(BASE, "data/cyp-challenge-TEST-BLINDED.csv"))
    smi_te = test["SMILES"].values
    allsmi = list(smi_tr) + list(smi_te)
    cpath = os.path.join(OUTD, "d3_cache.pkl")
    cache = {}
    if os.path.exists(cpath):
        with open(cpath, "rb") as fh:
            cache = pickle.load(fh)
    todo = [s for s in allsmi if s not in cache]
    print("3D: todo", len(todo), flush=True)
    from multiprocessing import Pool
    with Pool(12) as pool:
        for i, (smi, f) in enumerate(pool.imap_unordered(compute_one, todo,
                                                          chunksize=32)):
            cache[smi] = f
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(todo)}", flush=True)
                with open(cpath, "wb") as fh:
                    pickle.dump(cache, fh)
    with open(cpath, "wb") as fh:
        pickle.dump(cache, fh)
    F = np.array([cache[s] for s in allsmi])
    Ftr, Fte = F[: len(smi_tr)], F[len(smi_tr):]
    np.save(os.path.join(OUTD, "d3_feats_tr.npy"), Ftr)
    np.save(os.path.join(OUTD, "d3_feats_te.npy"), Fte)
    med = np.nanmedian(F, axis=0)
    nanfrac = np.nanmean(np.isnan(F), axis=0)
    print("d3 feats", Ftr.shape, Fte.shape, flush=True)
    print("medians:", np.round(med, 3).tolist())
    print("nan frac:", np.round(nanfrac, 3).tolist())


if __name__ == "__main__":
    main()
