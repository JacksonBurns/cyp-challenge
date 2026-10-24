import json, sys
d = json.load(open(sys.argv[1]))
for iso, v in d.items():
    top = sorted(v["weights_fitted_top"].items(),
                 key=lambda kv: -abs(kv[1]))[:8]
    pt = {k: round(w, 3) for k, w in v["weights_fitted_top"].items()
          if k.startswith("ft_pt")}
    print(iso, "cv", round(v["cv_mean_gain"], 4),
          f"{v['cv_pos_folds']}/5", "PASS" if v["gate_pass_L1"] else "FAIL")
    print("    top:", {k: round(w, 3) for k, w in top})
    print("    pt :", pt)
