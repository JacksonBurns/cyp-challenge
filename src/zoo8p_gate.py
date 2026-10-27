"""Permutation control: zoo6 + SHUFFLED gbm_adme. If this passes too,
the zoo8 gain is pool inflation, not ADME signal."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
H.FAMS["gbm_adme"] = [os.path.join(CACHE, "ft_oof_gbm_adme_shuf.csv")]
H.POOLS["zoo8p"] = H.POOLS["zoo6"] + ["gbm_adme"]

if __name__ == "__main__":
    H.main(emit=False, do_l4=True, pool="zoo8p",
           wout="hinge_w_zoo8p.json", report_name="hinge_blend_zoo8p.json",
           out_csv="regression_zoo8p_GATEONLY.csv")
