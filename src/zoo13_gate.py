"""zoo13 gates: zoo6 pool + seed-averaged ft_ext (quantitative pIC50 aux,
w0.05) family, with shuffled-label control. Evidence = real-minus-shuffled.
Usage: zoo13_extw005.py [real|shuf]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
mode = sys.argv[1] if len(sys.argv) > 1 else "real"
SRC = {"real": "ft_oof_extw005_avg.csv", "shuf": "ft_oof_extw005_avg_shuf.csv"}
H.FAMS["ft_extfam"] = [os.path.join(CACHE, SRC[mode])]
H.POOLS["zoo13"] = H.POOLS["zoo6"] + ["ft_extfam"]

if __name__ == "__main__":
    tag = "zoo13_" + mode
    H.main(emit=False, do_l4=True, pool="zoo13",
           wout=f"hinge_w_{tag}.json", report_name=f"hinge_blend_{tag}.json",
           out_csv=f"regression_{tag}_GATEONLY.csv")
