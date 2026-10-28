import sys
sys.path.insert(0, "src")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import r32_pharm as R
bad = [k for k, v in R.PAT.items() if v is None]
print("failed patterns:", bad)
print("n patterns:", len(R.PAT))
# probe alternatives for the complex ones
alts = {
    "catN": ["[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$([N+]);!$(N=a);!$(N#C)]",
             "[NX3;!$(N[a,O]);!$(N[#6]=[O,N,S])]"],
    "anion": ["[CX3](=[OX1])[OX2H1]", "[OX2H1][CX3](=[OX1])",
              "[CX3](=[OX1])[OX1-]", "c1nnn[nH]1"],
}
for k, vs in alts.items():
    for v in vs:
        p = Chem.MolFromSmarts(v)
        print(k, repr(v), "OK" if p is not None else "FAIL")
