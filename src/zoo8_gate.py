"""Pool gate: zoo6 vs zoo6+gbm_adme (zoo8). CPU. Gate: rotation-CV mean
gain > 0 per isoform, L4 bootstrap CI excludes 0."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
H.FAMS["gbm_adme"] = [os.path.join(CACHE, "ft_oof_gbm_adme.csv")]
H.POOLS["zoo8"] = H.POOLS["zoo6"] + ["gbm_adme"]

if __name__ == "__main__":
    H.main(emit=False, do_l4=True, pool="zoo8",
           wout="hinge_w_zoo8.json", report_name="hinge_blend_zoo8.json",
           out_csv="regression_zoo8_GATEONLY.csv")
