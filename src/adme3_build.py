"""ADME3: experimental pKa (user IUPAC/pKa high-confidence set, 24k rows).

Per canonical molecule: median per pKa type (pKaH1, pKa1, pKa2, pKaH2).
GBM (desc+Morgan feats, 5-fold OOF for train rows, full-fit -> test) per
endpoint -> predicted pKa profile for all 6895 challenge molecules.
Derived physics: fraction protonated/deprotonated at pH 7.4, zwitterion
window (pKaH1 - pKa1). Output: cache/adme3/adme3_feats_{tr,te}.npy + names.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
from adme2_build import featurize, canon, fit_predict  # noqa: E402

BASE = "/home/jackson/cyp-challenge"
CACHE = os.path.join(BASE, "cache")
OUTD = os.path.join(CACHE, "adme3")
os.makedirs(OUTD, exist_ok=True)


def main():
    d = pd.read_csv(os.path.join(BASE, "external/user_supplied/iupac_highconf_v24.csv"))
    d["canon"] = d["SMILES"].map(canon)
    d["pka"] = pd.to_numeric(d["pka_value"], errors="coerce")
    ok = d.dropna(subset=["canon", "pka"]).copy()
    # sanity: physical pKa range
    ok = ok[ok["pka"].between(-5, 18)]
    wide = ok.groupby(["canon", "pka_type"])["pka"].median().unstack()
    types = [t for t in ["pKaH1", "pKa1", "pKa2", "pKaH2"] if t in wide.columns]
    print("external molecules:", len(wide), "types:", types, flush=True)

    # external feats (cached)
    epath = os.path.join(OUTD, "ext_desc3.pkl")
    ext_canon = list(wide.index)
    if os.path.exists(epath):
        with open(epath, "rb") as fh:
            Xext = pickle.load(fh)
    else:
        F = featurize(ext_canon)
        Xext = pd.DataFrame(F, index=ext_canon)
        with open(epath, "wb") as fh:
            pickle.dump(Xext, fh)
    print("ext feats", Xext.shape, flush=True)

    # challenge feats (reuse adme2 cache)
    tpath = os.path.join(CACHE, "adme2", "challenge_desc2.pkl")
    with open(tpath, "rb") as fh:
        Xtr_d, Xte_d = pickle.load(fh)

    import hinge_blend as H  # noqa: E402
    Ftr, Fte, names = [], [], []
    for t in types:
        s = wide[t].dropna()
        idx = s.index.intersection(Xext.index)
        if len(idx) < 200:
            print(f"  skip {t}: {len(idx)}")
            continue
        Xfit = Xext.loc[idx].values.astype(np.float32)
        yfit = s.loc[idx].values.astype(float)
        m = np.isfinite(Xfit).all(axis=1)
        p_tr, p_te = fit_predict(Xfit[m], yfit[m],
                                 Xtr_d.values.astype(np.float32),
                                 Xte_d.values.astype(np.float32))
        Ftr.append(p_tr)
        Fte.append(p_te)
        names.append(t)
        print(f"  {t}: fit on {m.sum()} rows", flush=True)

    Atr = np.column_stack(Ftr)
    Ate = np.column_stack(Fte)
    # derived physics from predicted profile
    col = {n: i for i, n in enumerate(names)}
    for tag, A in [("tr", Atr), ("te", Ate)]:
        fh = np.zeros(len(A))
        fa = np.zeros(len(A))
        win = np.full(len(A), np.nan)
        if "pKaH1" in col:
            fh = 1.0 / (1.0 + 10 ** (7.4 - A[:, col["pKaH1"]]))
        if "pKa1" in col:
            fa = 1.0 / (1.0 + 10 ** (A[:, col["pKa1"]] - 7.4))
        if "pKaH1" in col and "pKa1" in col:
            win = A[:, col["pKaH1"]] - A[:, col["pKa1"]]
        np.save(os.path.join(OUTD, f"derived_{tag}.npy"),
                np.column_stack([fh, fa, win]))
    np.save(os.path.join(OUTD, "adme3_feats_tr.npy"), Atr)
    np.save(os.path.join(OUTD, "adme3_feats_te.npy"), Ate)
    open(os.path.join(OUTD, "adme3_names.txt"), "w").write("\n".join(names + ["frac_cat74", "frac_an74", "zwin"]))
    print("ADME3 feats:", Atr.shape, Ate.shape)


if __name__ == "__main__":
    main()
