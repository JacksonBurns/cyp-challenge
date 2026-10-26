import sys

import numpy as np

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa

for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
    for kind in ["direct_inhibition", "TDI_condition"]:
        col = f"{iso}_pIC50_{kind}"
        if col not in H.targets.columns:
            continue
        y = H.targets[col].values.astype(float)
        lab = y[~np.isnan(y)]
        if not len(lab):
            continue
        print(f"{col}: n={len(lab)} min={lab.min():.2f} "
              f"frac<=1.1={(lab<=1.1).mean():.3f} "
              f"frac<=2={(lab<=2).mean():.3f}")
