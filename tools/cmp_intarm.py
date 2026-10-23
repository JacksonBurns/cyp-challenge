"""Compare seed-averaged OOF column(s) on the hinge (ST-RAE) denominator.

Usage: python tools/cmp_intarm.py  (intarm-avg vs cpmed, fixed paths)
"""
import glob
import numpy as np
import pandas as pd

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def hinge_sums(y, yhat, lo, hi):
    h = np.maximum(lo - yhat, 0) + np.maximum(yhat - hi, 0)
    hm = np.maximum(lo - y.mean(), 0) + np.maximum(y.mean() - hi, 0)
    return h.sum(), hm.sum()


def avg(paths, col):
    dfs = sorted(paths)
    frames = [pd.read_csv(p).sort_values("SMILES").reset_index(drop=True)
              for p in dfs]
    base = frames[0]
    for d in frames[1:]:
        assert (d["SMILES"].values == base["SMILES"].values).all()
    return np.mean([d[col].values for d in frames], axis=0)


def main(paths_a, name_a, paths_b, name_b):
    ft = (pd.read_csv("cache/ft_data.csv")
          .drop_duplicates("SMILES").sort_values("SMILES")
          .reset_index(drop=True))
    inh = (pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv")
           .drop_duplicates("SMILES").set_index("SMILES"))
    lo_all, hi_all = {}, {}
    for iso in ISO:
        lo_all[iso] = inh[f"{iso}_pIC50_direct_inhibition_conf_low"] \
            .reindex(ft["SMILES"].values).values
        hi_all[iso] = inh[f"{iso}_pIC50_direct_inhibition_conf_high"] \
            .reindex(ft["SMILES"].values).values
    tot_a = tot_b = tot_d = 0.0
    for iso in ISO:
        c = iso
        y = ft[c].values
        pa = avg(paths_a, c)
        pb = avg(paths_b, c)
        lo, hi = lo_all[iso], hi_all[iso]
        ok = (~np.isnan(y)) & (~np.isnan(lo)) & (~np.isnan(hi))
        ok &= ~np.isnan(pa) & ~np.isnan(pb)
        na, da = hinge_sums(y[ok], pa[ok], lo[ok], hi[ok])
        nb, _ = hinge_sums(y[ok], pb[ok], lo[ok], hi[ok])
        print(f"{iso}: {name_a}={na/da:.4f}  {name_b}={nb/da:.4f}  "
              f"delta={nb/da-na/da:+.4f}  n={int(ok.sum())}")
        tot_a += na
        tot_b += nb
        tot_d += da
    print(f"macro: {name_a}={tot_a/tot_d:.4f}  {name_b}={tot_b/tot_d:.4f}  "
          f"delta={tot_b/tot_d-tot_a/tot_d:+.4f}")


if __name__ == "__main__":
    A = sorted(glob.glob("cache/ft_oof_intarm*.csv"))
    A = [p for p in A if "smoke" not in p]
    main(A, "intarm-avg", ["cache/ft_oof_cpmed.csv"], "cpmed")
