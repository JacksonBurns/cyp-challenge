"""Pool gate for the PCGrad family: zoo6 vs zoo6+ft_pcgrad (L1 hinge-fit
+ L4 bootstrap). Gate per isoform: rotation-CV mean gain > 0 and L4 CI
excludes 0. CPU only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
H.FAMS["ft_pcgrad"] = [os.path.join(CACHE, "ft_oof_pcgrad.csv")]
H.POOLS["zoo7"] = H.POOLS["zoo6"] + ["ft_pcgrad"]

if __name__ == "__main__":
    H.main(emit=False, do_l4=True, pool="zoo7",
           wout="hinge_w_zoo7.json", report_name="hinge_blend_zoo7.json",
           out_csv="regression_zoo7_GATEONLY.csv")
