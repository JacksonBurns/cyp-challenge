"""Compare intarm vs cpmed OOF on the hinge (ST-RAE) over labeled+bounded rows."""
import numpy as np
import pandas as pd

ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


def hinge_sum(y, yhat, lo, hi):
    return np.maximum(lo - yhat, 0).sum() + np.maximum(yhat - hi, 0).sum()


def main():
    ft = pd.read_csv("cache/ft_data.csv")
    cp = pd.read_csv("cache/ft_oof_cpmed.csv")
    ia = pd.read_csv("cache/ft_oof_intarm.csv")
    assert (ft.SMILES.values == ia.SMILES.values).all()
    assert (ft.SMILES.values == cp.SMILES.values).all()
    inh = pd.read_csv("data/cyp-challenge-TRAIN_inhibition.csv").drop_duplicates("SMILES")
    den_all, num_cp_all, num_ia_all = 0.0, 0.0, 0.0
    for iso in ISO:
        y = ft[iso].values
        p_cp = cp[iso].values
        p_ia = ia[iso].values
        lo = inh.set_index("SMILES").reindex(ft.SMILES.values)[
            f"{iso}_pIC50_direct_inhibition_conf_low"].values
        hi = inh.set_index("SMILES").reindex(ft.SMILES.values)[
            f"{iso}_pIC50_direct_inhibition_conf_high"].values
        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(p_cp) & np.isfinite(p_ia)
        yy, loo, hii = y[ok], lo[ok], hi[ok]
        den = np.abs(yy - yy.mean()).sum()
        ncp = hinge_sum(yy, p_cp[ok], loo, hii)
        nia = hinge_sum(yy, p_ia[ok], loo, hii)
        print(f"{iso}: n={ok.sum()}  st_rae cpmed={ncp/den:.4f}  intarm={nia/den:.4f}  "
              f"delta={ncp/den - nia/den:+.4f}  "
              f"viol_cp={int((loo > p_cp[ok]).sum() + (p_cp[ok] > hii).sum())}  "
              f"viol_ia={int((loo > p_ia[ok]).sum() + (p_ia[ok] > hii).sum())}")
        den_all += den; num_cp_all += ncp; num_ia_all += nia
    print(f"POOLED: cpmed={num_cp_all/den_all:.4f}  intarm={num_ia_all/den_all:.4f}  "
          f"delta={num_cp_all/den_all - num_ia_all/den_all:+.4f}")


if __name__ == "__main__":
    main()
