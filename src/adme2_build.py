"""ADME2 feature builder: predicted assay-physics endpoints for all 6895
challenge molecules, from TDC + prior external sets (zero-overlap transfer).

Endpoints (20):
  v1 (8): LogD, LogS_exp, KSOL, MLM, HLM, Caco2, MPPB, rotbar (see adme_aux_arm)
  v2 (12): pKa-acid, pKa-base (AstraZeneca), AqSolDB logS, Caco2_Wang,
    HIA, Pgp, BBB, VDss, PPB, half-life, CL-hepatocyte, CL-microsome,
    logD-lipophilicity(AZ), + CYP2C9/2D6/3A4/1A2 Veith substrate P(+)
    (4 direct CYP labels -> model prediction = physics supervision proxy)
GBM (5-fold CV OOF for train rows, full-fit -> test) per endpoint.
Caches: cache/adme2/adme2_feats_{tr,te}.npy + row-order aligned.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
RDLogger.DisableLog("rdApp.*")
BASE = "/home/jackson/cyp-challenge"
CACHE = os.path.join(BASE, "cache")
OUTD = os.path.join(CACHE, "adme2")
TDC = os.path.join(BASE, "external/user_supplied/tdc")
os.makedirs(OUTD, exist_ok=True)

DESC_NAMES = [d[0] for d in Descriptors._descList]
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def featurize(smi_list, chunk=2000):
    """200 descriptors + 1024-bit Morgan counts-summed, per SMILES (NaN on fail)."""
    rows = []
    for smi in smi_list:
        m = Chem.MolFromSmiles(smi) if smi and isinstance(smi, str) else None
        if m is None:
            rows.append(np.full(len(DESC_NAMES) + 1024, np.nan))
            continue
        dv = []
        for name in DESC_NAMES:
            f = getattr(Descriptors, name, None)
            if f is None:
                dv.append(np.nan)
                continue
            try:
                v = f(m)
                dv.append(float(v) if np.isfinite(v) else np.nan)
            except Exception:
                dv.append(np.nan)
        bv = GEN.GetFingerprint(m)
        fp = np.zeros(1024)
        for b in bv.GetOnBits():
            fp[b] = 1.0
        rows.append(np.array(dv + list(fp), dtype=np.float64))
        if len(rows) % 2000 == 0:
            print(f"  featurized {len(rows)}", flush=True)
    return np.array(rows, dtype=np.float32)


def canon(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToSmiles(m) if m else None


TDC_SETS = [
    ("caco2_wang", "caco2_wang.csv"),
    ("hia_hou", "hia_hou.csv"),
    ("pgp", "pgp_broccatelli.csv"),
    ("bbb", "bbb_martins.csv"),
    ("vdss", "vdss_lombardo.csv"),
    ("ppbr", "ppbr_az.csv"),
    ("aqsol", "aqsoldb.csv"),
    ("half_life", "half_life_obach.csv"),
    ("cl_hep", "cl_hepatocyte.csv"),
    ("cl_mic", "cl_microsome.csv"),
    ("lipo_az", "lipophilicity_az.csv"),
    ("cyp2c9_veith", "cyp2c9_veith.csv"),
    ("cyp2d6_veith", "cyp2d6_veith.csv"),
    ("cyp3a4_veith", "cyp3a4_veith.csv"),
    ("cyp1a2_veith", "cyp1a2_veith.csv"),
    ("cyp2c9_sub", "cyp2c9_sub.csv"),
    ("cyp2d6_sub", "cyp2d6_sub.csv"),
    ("cyp3a4_sub", "cyp3a4_sub.csv"),
]


def load_tdc():
    """-> dict prefix -> Series(canonical SMILES -> Y)."""
    frames = {}
    for prefix, fn in TDC_SETS:
        path = os.path.join(TDC, fn)
        if not os.path.exists(path):
            print(f"  missing {fn}")
            continue
        df = pd.read_csv(path, sep="\t")
        sc = "Drug" if "Drug" in df.columns else "X"
        df["smi"] = df[sc].map(canon)
        v = pd.to_numeric(df["Y"], errors="coerce")
        s = pd.DataFrame({"smi": df["smi"], "v": v}).dropna()
        s = s.groupby("smi")["v"].mean()
        frames[prefix] = s
        print(f"  {prefix}: {len(s)} mols")
    return frames


def fit_predict(Xfit, yfit, Xq1, Xq2):
    """5-fold GBM ensemble on external labeled rows -> predict onto queries."""
    from sklearn.model_selection import KFold
    models = []
    kf = KFold(5, shuffle=True, random_state=0)
    for tr, _ in kf.split(Xfit):
        g = LGBMRegressor(n_estimators=400, learning_rate=0.06, num_leaves=63,
                          verbose=-1)
        g.fit(Xfit[tr], yfit[tr])
        models.append(g)
    p1 = np.mean([g.predict(Xq1) for g in models], axis=0)
    p2 = np.mean([g.predict(Xq2) for g in models], axis=0)
    return p1, p2


def main():
    import hinge_blend as H
    smi_tr = H.Xtr["SMILES"].values
    smi_te = pd.read_csv(os.path.join(BASE, "data/cyp-challenge-TEST-BLINDED.csv"))["SMILES"].values
    # external descriptor cache
    dpath = os.path.join(OUTD, "ext_desc2.pkl")
    fpath = os.path.join(OUTD, "ext_feats2.npy")
    frames = load_tdc()
    all_smis = sorted(set().union(*[set(f.index) for f in frames.values()]))
    print("external universe:", len(all_smis))
    feats = np.load(fpath) if os.path.exists(fpath) else featurize(all_smis)
    np.save(fpath, feats)
    Xext = pd.DataFrame(feats, index=all_smis,
                        columns=[f"d{i}" for i in range(feats.shape[1])])
    # train/test descriptors
    tpath = os.path.join(OUTD, "challenge_desc2.pkl")
    if os.path.exists(tpath):
        with open(tpath, "rb") as fh:
            Xtr_d, Xte_d = pickle.load(fh)
    else:
        Xtr_d = pd.DataFrame(featurize(list(smi_tr)), columns=[f"d{i}" for i in range(feats.shape[1])])
        Xte_d = pd.DataFrame(featurize(list(smi_te)), columns=[f"d{i}" for i in range(feats.shape[1])])
        with open(tpath, "wb") as fh:
            pickle.dump((Xtr_d, Xte_d), fh)
    cols = list(Xext.columns)
    Ftr, Fte, names = [], [], []
    for name, s in frames.items():
        idx = s.index.intersection(Xext.index)
        if len(idx) < 100:
            print(f"  skip {name}: {len(idx)} external rows")
            continue
        Xfit = Xext.loc[idx].values.astype(np.float32)
        yfit = s.loc[idx].values.astype(float)
        ok = np.isfinite(Xfit).all(axis=1) & np.isfinite(yfit)
        p_tr, p_te = fit_predict(Xfit[ok], yfit[ok],
                                 Xtr_d.values.astype(np.float32),
                                 Xte_d.values.astype(np.float32))
        Ftr.append(p_tr)
        Fte.append(p_te)
        names.append(name)
        print(f"  {name}: fit on {ok.sum()} external rows")
    Atr = np.column_stack(Ftr)
    Ate = np.column_stack(Fte)
    np.save(os.path.join(OUTD, "adme2_feats_tr.npy"), Atr)
    np.save(os.path.join(OUTD, "adme2_feats_te.npy"), Ate)
    open(os.path.join(OUTD, "adme2_names.txt"), "w").write("\n".join(names))
    print("ADME2 feats:", Atr.shape, Ate.shape)


if __name__ == "__main__":
    main()
