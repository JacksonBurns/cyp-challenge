"""Convert a UMTM OOF file to the legacy ft_oof_* schema (SMILES + 4 iso cols),
so blendN_all / blend_nested_all / regression_family_block / regression_final
consume UMTM members with zero changes to the legacy machinery.

  python tools/umtm_to_ftoof.py cache/umtm_oof_L2_s0.csv   -> cache/ft_oof_umtmL2.csv
  python tools/umtm_to_ftoof.py cache/umtm_oof_L2_s1.csv   -> cache/ft_oof_umtmL2_s1.csv
Row order must already equal ft_data.csv (asserted).
"""
import os
import re
import sys

import pandas as pd

ROOT = os.path.expanduser("~/cyp-challenge")
sys.path.insert(0, os.path.join(ROOT, "src"))
from umtm_data import DIRECT  # noqa: E402

ISO = [c.replace("_direct", "") for c in DIRECT]

for path in sys.argv[1:]:
    df = pd.read_csv(path)
    m = re.search(r"umtm_oof_(L\d)(?:_s(\d+))?\.csv", os.path.basename(path))
    tag = f"umtm{m.group(1)}" + (f"_s{m.group(2)}" if m.group(2) else "")
    out = pd.DataFrame({"SMILES": df["SMILES"].values})
    for iso in ISO:
        out[iso] = df[f"{iso}_direct"].values
    ft = pd.read_csv(os.path.join(ROOT, "cache", "ft_data.csv"), usecols=["SMILES"])
    assert (out["SMILES"].tolist() == ft["SMILES"].tolist()), f"{path}: row order != ft_data"
    dst = os.path.join(ROOT, "cache", f"ft_oof_{tag}.csv")
    out.to_csv(dst, index=False)
    print("wrote", dst, out.shape)
