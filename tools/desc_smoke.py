import sys
sys.path.insert(0, "tools")
import desc_build as D
import pandas as pd

fd = pd.read_csv("cache/ft_data.csv")
sm = fd["SMILES"].head(30).tolist()
sm += ["c1ccc(-c2ccccc2)c1", "CN(C)CCc1c[nH]c2ccccc12",
       "C1=CC=C(C=C1)C2=CC=CC=C2", "O=S(=O)(N)c1ccc2ccccc2c1"]
tr = D.build(sm)
pd.set_option("display.max_columns", 60, "display.width", 250)
print(tr.tail(4).round(3).to_string())
print("cols:", len(tr.columns))
