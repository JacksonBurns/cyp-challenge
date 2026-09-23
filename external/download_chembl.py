"""Download ChEMBL CYP IC50 activities (4 scored isoforms) via paginated JSON API.

Kept OUT of the model path: external data may only inform RANKING (pretraining
features or multitask auxiliary targets), never the DRC-calibrated placement.

Output: external/chembl_cyp_ic50.csv  (one row per activity)
Columns: isoform, canonical_smiles, molecule_chembl_id, standard_relation,
standard_value, pchembl_value, assay_chembl_id, assay_description, assay_type,
document_chembl_id, target_pref_name, activity_comment, data_validity_comment,
potential_duplicate, src_id, year
Run: python3 external/download_chembl.py   (stdlib only)
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chembl_cyp_ic50.csv")

TARGETS = {
    "CYP1A2": "CHEMBL3356",
    "CYP2C9": "CHEMBL3397",
    "CYP2D6": "CHEMBL289",
    "CYP3A4": "CHEMBL340",
}
BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
PAGE = 1000
KEEP = ["canonical_smiles", "molecule_chembl_id", "standard_relation", "standard_value",
        "pchembl_value", "assay_chembl_id", "assay_description", "assay_type",
        "document_chembl_id", "target_pref_name", "activity_comment",
        "data_validity_comment", "potential_duplicate", "src_id", "standard_units",
        "target_organism", "bao_format"]


def fetch(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {i+1}: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def main():
    fields = ["isoform"] + KEEP
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for iso, tid in TARGETS.items():
            offset, total, n = 0, None, 0
            while total is None or offset < total:
                q = urllib.parse.urlencode({
                    "target_chembl_id": tid, "standard_type": "IC50",
                    "limit": PAGE, "offset": offset})
                d = fetch(f"{BASE}?{q}")
                total = d["page_meta"]["total_count"]
                for a in d["activities"]:
                    row = {"isoform": iso}
                    for k in KEEP:
                        row[k] = a.get(k)
                    w.writerow(row)
                n += len(d["activities"])
                offset += PAGE
                print(f"{iso}: {n}/{total}", flush=True)
    print("DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
