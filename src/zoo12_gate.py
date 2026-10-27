"""zoo12 family gates: zoo6 + seed-averaged ft_adme2 families.
modes: w002 | w002shuf | w01 | w01shuf | band | bandshuf.
Evidence = real-minus-shuffled per isoform."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
mode = sys.argv[1] if len(sys.argv) > 1 else "band"
SRC = {
    "w002": "ft_oof_adme2w002_avg.csv", "w002shuf": "ft_oof_adme2w002_avg_shuf.csv",
    "w01": "ft_oof_adme2w01_avg.csv", "w01shuf": "ft_oof_adme2w01_avg_shuf.csv",
    "band": "ft_oof_adme2band_avg.csv", "bandshuf": "ft_oof_adme2band_avg_shuf.csv",
}
H.FAMS["ft_adme2fam"] = [os.path.join(CACHE, SRC[mode])]
H.POOLS["zoo12"] = H.POOLS["zoo6"] + ["ft_adme2fam"]

if __name__ == "__main__":
    tag = "zoo12_" + mode
    H.main(emit=False, do_l4=True, pool="zoo12",
           wout=f"hinge_w_{tag}.json", report_name=f"hinge_blend_{tag}.json",
           out_csv=f"regression_{tag}_GATEONLY.csv")
