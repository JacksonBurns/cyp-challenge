"""Verify no inf/nan-label corruption remains in ft_ext tables."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import ft_ext

ft, test, allrows = ft_ext.load_tables()
V = allrows[ft_ext.COLS].values.astype(float)
print("inf anywhere:", int(np.isinf(V).sum()))
print("finite range:", np.nanmin(V[np.isfinite(V)]), np.nanmax(V[np.isfinite(V)]))
print("chembl rows", int((allrows['_src']=='chembl').sum()))
