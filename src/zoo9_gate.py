"""Pool gate for ft_adme (ADME-aux-target MPNN arm): zoo6+ft_adme (zoo9),
plus shuffled-member control zoo9p. ADME-specific evidence =
zoo9 gain MINUS zoo9p gain (real-minus-shuffled), per isoform."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
mode = sys.argv[1] if len(sys.argv) > 1 else "real"
src = os.path.join(CACHE, "ft_oof_adme.csv" if mode == "real"
                   else "ft_oof_adme_shuf.csv")
H.FAMS["ft_adme"] = [src]
H.POOLS["zoo9"] = H.POOLS["zoo6"] + ["ft_adme"]

if __name__ == "__main__":
    tag = "zoo9" if mode == "real" else "zoo9p"
    H.main(emit=False, do_l4=True, pool="zoo9",
           wout=f"hinge_w_{tag}.json", report_name=f"hinge_blend_{tag}.json",
           out_csv=f"regression_{tag}_GATEONLY.csv")
