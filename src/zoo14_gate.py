"""zoo14: zoo6 + ft_extfam (quantitative w0.05) + ft_adme2fam (physics band).
Modes: ext | extshuf | adme | admeshuf | both | bothshuf.
Evidence: real-minus-shuffled per isoform.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hinge_blend as H  # noqa: E402

CACHE = H.CACHE
mode = sys.argv[1] if len(sys.argv) > 1 else "both"
EXT = {"real": "ft_oof_extw005_avg.csv", "shuf": "ft_oof_extw005_avg_shuf.csv"}
ADM = {"real": "ft_oof_adme2band_avg.csv", "shuf": "ft_oof_adme2band_avg_shuf.csv"}
shuf = mode.endswith("shuf")
kind = "shuf" if shuf else "real"
BASE = {"ext": "ext", "extshuf": "ext", "adme": "adme", "admeshuf": "adme",
        "both": "both", "bothshuf": "both"}
fam_ext = EXT[kind]
fam_adm = ADM[kind]
H.FAMS["ft_extfam"] = [os.path.join(CACHE, fam_ext)]
H.FAMS["ft_admefam"] = [os.path.join(CACHE, fam_adm)]
base = BASE[mode]
pool = H.POOLS["zoo6"][:]
if base in ("ext", "both"):
    pool += ["ft_extfam"]
if base in ("adme", "both"):
    pool += ["ft_admefam"]
H.POOLS["zoo14"] = pool

if __name__ == "__main__":
    tag = "zoo14_" + mode
    H.main(emit=False, do_l4=True, pool="zoo14",
           wout=f"hinge_w_{tag}.json", report_name=f"hinge_blend_{tag}.json",
           out_csv=f"regression_{tag}_GATEONLY.csv")
