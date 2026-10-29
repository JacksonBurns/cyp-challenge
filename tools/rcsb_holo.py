"""Pick holo CYP structures: UniProt PDB list (X-ray <=2.8A), download,
keep entries with HEM + >=1 non-solvent/non-clip ligand. Print top picks."""
import json
import os
import urllib.request

DOCKD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "dock")
os.makedirs(DOCKD, exist_ok=True)
UP = {"CYP1A2": "P05177", "CYP2C9": "P11712", "CYP2D6": "P10635", "CYP3A4": "P08684"}
SKIP = {"HEM", "HOH", "SO4", "PO4", "GOL", "EDO", "PEG", "CL", "NA", "MG", "ACT",
        "MES", "EPE", "TRS", "CSX", "MSE", "DMS", "MPD", "IOD", "NO3", "CA", "ZN",
        "ANM", "BME", "CAD", "CIT", "FMT", "POC", "13P", "P6G", "SR", "K", "TL",
        "PRO", "TLA", "SCN", "BTB", "CPT", "SPM", "SIN", "DOX", "UNL"}


def pdb_ids_for(up):
    url = f"https://rest.uniprot.org/uniprotkb/{up}.json?fields=xref_pdb"
    d = json.loads(urllib.request.urlopen(url, timeout=30).read())
    ids = []
    for x in d.get("uniProtKBCrossReferences", []):
        if x["database"] == "PDB":
            pid = x["id"].lower()
            props = {p["key"]: p["value"] for p in x.get("properties", [])}
            raw = str(props.get("Resolution", "")).strip().rstrip("A ").strip()
            try:
                res = float(raw)
            except ValueError:
                res = 99
            if "X-ray" in props.get("Method", "") and res <= 2.8:
                ids.append((pid, res))
    return sorted(ids, key=lambda t: t[1])


def het_set(f):
    hets = set()
    for ln in open(f):
        if ln.startswith("HETATM"):
            hets.add(ln[17:20].strip())
    return hets


def fe_center(f):
    import numpy as np
    pts = []
    for ln in open(f):
        if ln.startswith("HETATM") and ln[17:20].strip() == "HEM" and ln[12:16].strip() == "FE":
            pts.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return np.mean(pts, axis=0).tolist() if pts else None


for iso, up in UP.items():
    picks = []
    for pid, res in pdb_ids_for(up)[:25]:
        f = os.path.join(DOCKD, f"{pid}.pdb")
        if not os.path.exists(f):
            try:
                urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid.upper()}.pdb", f)
            except Exception:
                continue
        hets = het_set(f)
        lig = hets - SKIP
        if "HEM" in hets and lig:
            c = fe_center(f)
            picks.append({"pid": pid, "res": res, "lig": sorted(lig)[:4], "fe": c})
        if len(picks) >= 2:
            break
    print(iso, json.dumps(picks), flush=True)
