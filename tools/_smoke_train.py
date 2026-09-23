"""Smoke test ft_ext train path: tiny subset, 2 epochs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import ft_ext

ft_ext.MAX_EPOCHS = 2
ft_ext.PATIENCE = 2
ft_ext.EXT_CAP = 300
ft, test, allrows = ft_ext.load_tables()
ch_mask = (allrows["_src"] == "challenge").values
lab = allrows[ft_ext.COLS].notna().any(axis=1).values
tr_pool = allrows[ch_mask].iloc[:400]
extra = allrows[~ch_mask & lab].iloc[:300]
ev = allrows[ch_mask].iloc[400:450]
fit = pd.concat([tr_pool, extra], ignore_index=True)
model = ft_ext.train_model(fit, ev, True, "smoke", seed=0)
P = ft_ext.predict(model, ev)
print("preds shape", P.shape, "finite cols", np.isfinite(P).mean(0))
