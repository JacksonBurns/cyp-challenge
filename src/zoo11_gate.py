"""Pool gate for ft_adme2 (31-head ADME2 aux-target MPNN): zoo11 = zoo6 +
ft_adme2, with shuffled-member control zoo11p. Evidence = delta real-minus-shuf."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
mode = sys.argv[1] if len(sys.argv) > 1 else "real"
src = os.path.join(CACHE, "ft_oof_adme2.csv" if mode == "real"
                   else "ft_oof_adme2_shuf.csv")
H.FAMS["ft_adme2"] = [src]
H.POOLS["zoo11"] = H.POOLS["zoo6"] + ["ft_adme2"]

if __name__ == "__main__":
    tag = "zoo11" if mode == "real" else "zoo11p"
    H.main(emit=False, do_l4=True, pool="zoo11",
           wout=f"hinge_w_{tag}.json", report_name=f"hinge_blend_{tag}.json",
           out_csv=f"regression_{tag}_GATEONLY.csv")
