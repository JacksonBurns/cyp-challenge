"""Smoke test for ft_ext.load_tables without touching GPU."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import ft_ext

ft, test, allrows = ft_ext.load_tables()
ch = (allrows["_src"] == "challenge").values
lab = allrows[ft_ext.COLS].notna().any(axis=1).values
print("rows", len(allrows), "challenge", ch.sum(), "labeled", lab.sum())
print("chembl rows", (allrows["_src"] == "chembl").sum(),
      "pubchem rows", (allrows["_src"] == "pubchem").sum())
print("counts per col:")
print(allrows[ft_ext.COLS].notna().sum())
# challenge rows must keep primary labels only
prim = allrows[[f"{i}" for i in ft_ext.PRIMARY]].notna().any(axis=1).values
auxl = allrows[[a for a in ft_ext.AUX]].notna().any(axis=1).values
print("challenge rows w/ aux labels (should be 0):", (ch & auxl).sum())
print("external rows w/ primary labels (should be 0):", ((~ch) & prim).sum())
