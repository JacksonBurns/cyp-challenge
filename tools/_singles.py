import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, "src")
import run_regression as R

Xtr, Xte, targets, bounds, S, Ste, test = R.build_all()
files = sys.argv[1:]
for f in files:
    d = pd.read_csv(f)
    print("==", f)
    for iso in R.ISOFORMS:
        y = targets[R.DIRECT[iso]].values
        m = ~np.isnan(y)
        print(" ", iso, round(float(pearsonr(y[m], d[iso].values[m]).statistic), 3))
