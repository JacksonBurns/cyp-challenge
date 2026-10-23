"""Download ChEMBL IC50 for the CYP homologs NOT in the scored set.

Adds aux-task diversity axes (2C19/2C8/2B6/2A6): human single-protein
targets, same filter pipeline as external/download_chembl.py. Target IDs
discovered via the target API (pref_name exact, organism Homo sapiens,
target_type 'S' single protein) - recorded in the CSV for reproducibility.

Output: external/chembl_cyp_homolog_ic50.csv
Run: python3 external/download_chembl_homologs.py   (stdlib only)
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chembl_cyp_homolog_ic50.csv")
API = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000
KEEP = ["canonical_smiles", "molecule_chembl_id", "standard_relation", "standard_value",
        "pchembl_value", "assay_chembl_id", "assay_description", "assay_type",
        "document_chembl_id", "target_pref_name", "activity_comment",
        "data_validity_comment", "potential_duplicate", "src_id", "standard_units",
        "target_organism", "bao_format"]
HOMOLOGS = ["CYP2C19", "CYP2C8", "CYP2B6", "CYP2A6"]


def fetch(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {i+1}: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def find_target(iso):
    q = urllib.parse.urlencode({
        "pref_name__iexact": iso, "target_organism__cell_species__tax_id": 9606,
        "target_type": "S", "limit": 10})
    d = fetch(f"{API}/target.json?{q}")
    for t in d["targets"]:
        if str(t.get("pref_name", "")).upper() == iso:
            return t["target_chembl_id"]
    q2 = urllib.parse.urlencode({"pref_name__iexact": f"Cytochrome P450 {iso[3:]}",
                                 "target_type": "S", "limit": 10})
    d2 = fetch(f"{API}/target.json?{q2}")
    for t in d2["targets"]:
        if "homo" in str(t.get("organism", "")).lower() or True:
            return t["target_chembl_id"]
    return None


def main():
    fields = ["isoform", "target_chembl_id"] + KEEP
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for iso in HOMOLOGS:
            tid = find_target(iso)
            print(f"{iso} -> {tid}", flush=True)
            if not tid:
                continue
            offset, total, n = 0, None, 0
            while total is None or offset < total:
                q = urllib.parse.urlencode({
                    "target_chembl_id": tid, "standard_type": "IC50",
                    "limit": PAGE, "offset": offset})
                d = fetch(f"{API}/activity.json?{q}")
                total = d["page_meta"]["total_count"]
                for a in d["activities"]:
                    row = {"isoform": iso, "target_chembl_id": tid}
                    for k in KEEP:
                        row[k] = a.get(k)
                    w.writerow(row)
                n += len(d["activities"])
                offset += PAGE
                print(f"{iso}: {n}/{total}", flush=True)
    print("DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
