import csv
import collections

iso = collections.Counter()
smis = set()
with open("external/chembl_cyp_ic50.csv") as f:
    for row in csv.DictReader(f):
        iso[row["isoform"]] += 1
        if row["canonical_smiles"]:
            smis.add(row["canonical_smiles"])
print(dict(iso), "unique smiles:", len(smis))
