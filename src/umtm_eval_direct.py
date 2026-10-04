"""Gate eval: compare a UMTM OOF (4 direct heads) against legacy OOF files.

Prints per-isoform OOF Pearson + RMSE + macro Pearson for any cache/*_oof file
whose columns include the 4 isoforms (umtm_oof_*.csv maps DIRECT heads; legacy
ft_oof_*.csv uses iso-named columns). Usage:
  python src/umtm_eval_direct.py cache/umtm_oof_L2_ext.csv cache/ft_oof_ext.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

from umtm_data import DIRECT  # noqa: E402


def load_pair(path):
    df = pd.read_csv(path)
    ycols = {}
    for iso in ISO:
        ycols[iso] = f"{iso}_direct" if f"{iso}_direct" in df.columns else iso
    return df, ycols


def main(paths):
    # truth = ft_data (X_train order)
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    per_iso = {}
    for p in paths:
        df, ycols = load_pair(p)
        assert (df["SMILES"].values == ft["SMILES"].values).all(), f"{p}: row order != ft_data"
        name = os.path.basename(p).replace(".csv", "")
        rs, rms = [], []
        for i, iso in enumerate(ISO):
            y, pr = ft[iso].values, df[ycols[iso]].values
            m = ~np.isnan(y) & ~np.isnan(pr)
            r = float(np.corrcoef(y[m], pr[m])[0, 1])
            rmse = float(np.sqrt(((y[m] - pr[m]) ** 2).mean()))
            rs.append(r)
            rms.append(rmse)
            per_iso.setdefault(iso, []).append((name, r, rmse))
        print(f"{name:28s} macro pearson {np.mean(rs):.4f} | per-iso " +
              " ".join(f"{i[3:]}:{r:.3f}/{e:.3f}" for i, r, e in zip(ISO, rs, rms)), flush=True)
    print("\nper-iso detail:")
    for iso in ISO:
        for name, r, rmse in sorted(per_iso[iso], key=lambda x: -x[1]):
            print(f"  {iso:7s} {name:28s} r={r:.4f} rmse={rmse:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
