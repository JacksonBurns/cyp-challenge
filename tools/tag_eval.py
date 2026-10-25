"""Pooled ST-RAE eval for any cache/ft_oof_<tag>.csv (mirrors loss_zoo)."""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from interval_arm import soft_rae, ISO  # noqa: E402

TRIPWIRE = 0.05
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")


def main(tag):
    oof = (pd.read_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"))
           .drop_duplicates("SMILES").set_index("SMILES"))
    inh = (pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))
           .drop_duplicates("SMILES").set_index("SMILES"))
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    yfull, lo_full, hi_full = {}, {}, {}
    for iso in ISO:
        yfull[iso] = ft[iso].values
        lo_full[iso] = inh[f"{iso}_pIC50_direct_inhibition_conf_low"] \
            .reindex(ft["SMILES"]).values
        hi_full[iso] = inh[f"{iso}_pIC50_direct_inhibition_conf_high"] \
            .reindex(ft["SMILES"]).values
    folds = ft["fold"].values
    oofv = oof[ISO].reindex(ft["SMILES"]).values
    ev = {}
    for j, iso in enumerate(ISO):
        m = ~np.isnan(yfull[iso]) & ~np.isnan(oofv[:, j])
        pooled = soft_rae(yfull[iso][m], oofv[m, j],
                          lo_full[iso][m], hi_full[iso][m])
        per_fold = []
        for f in range(5):
            mf = m & (folds == f)
            per_fold.append(soft_rae(yfull[iso][mf], oofv[mf, j],
                                     lo_full[iso][mf], hi_full[iso][mf]))
        off = max(abs(x - pooled) for x in per_fold)
        ev[iso] = {"per_fold_st_rae": [round(float(x), 4) for x in per_fold],
                   "pooled_st_rae": round(float(pooled), 4),
                   "max_fold_off_pooled": round(float(off), 4),
                   "quarantine": bool(off > TRIPWIRE)}
        print(f"[{tag}]", iso, json.dumps(ev[iso]), flush=True)
    json.dump(ev, open(os.path.join(CACHE, f"zoo_eval_{tag}.json"), "w"),
              indent=2)
    print("macro", round(sum(v["pooled_st_rae"] for v in ev.values()) / 4, 4))


if __name__ == "__main__":
    main(sys.argv[1])
