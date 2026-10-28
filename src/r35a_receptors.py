"""R3-5a: download CYP co-crystal structures, extract heme-Fe box centers.

1Z19 (1A2), 1EU9 (2C9), 4UOU (2D6), 3MO1 (3A4). Center = mean over heme
(HEM) Fe atom; box 24 A. Writes dock_centers.json.
"""
import json
import os
import urllib.request

DOCKD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "dock")
os.makedirs(DOCKD, exist_ok=True)
PDBS = {"CYP1A2": "1z19", "CYP2C9": "1nw8", "CYP2D6": "4uou", "CYP3A4": "3mo1"}


def fetch(pdb):
    f = os.path.join(DOCKD, f"{pdb}.pdb")
    if not os.path.exists(f):
        url = f"https://files.rcsb.org/download/{pdb.upper()}.pdb"
        urllib.request.urlretrieve(url, f)
    return f


def fe_center(f):
    pts = []
    for ln in open(f):
        if ln.startswith("HETATM") and ln[17:20].strip() == "FE" and "HEM" in ln:
            pts.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    if not pts:
        return None
    import numpy as np
    return np.mean(pts, axis=0).tolist()


def main():
    out = {}
    for iso, pdb in PDBS.items():
        f = fetch(pdb)
        c = fe_center(f)
        print(iso, pdb, "heme-Fe center", c, flush=True)
        out[iso] = {"pdb": pdb, "center": c, "box": 24.0}
    json.dump(out, open(os.path.join(DOCKD, "dock_centers.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
