"""ADME2 acquisition: TDC datasets via dataverse datafile IDs (no pytdc pkg).

Downloads CSVs to external/user_supplied/tdc/. IDs from tdc/metadata.py.
"""
import os

import requests

OUT = "/home/jackson/cyp-challenge/external/user_supplied/tdc"
os.makedirs(OUT, exist_ok=True)
BASE = "https://dataverse.harvard.edu/api/access/datafile/"

IDS = {
    "cyp2c9_veith.csv": 4259577,
    "cyp2d6_veith.csv": 4259580,
    "cyp3a4_veith.csv": 4259582,
    "cyp1a2_veith.csv": 4259573,
    "caco2_wang.csv": 4259569,
    "ppbr_az.csv": 6413140,
    "aqsoldb.csv": 4259610,
    "half_life_obach.csv": 4266799,
    "cl_hepatocyte.csv": 4266187,
    "cl_microsome.csv": 4266186,
    "lipophilicity_az.csv": 4259595,
    "hia_hou.csv": 4259591,
    "pgp_broccatelli.csv": 4259597,
    "bbb_martins.csv": 4259566,
    "vdss_lombardo.csv": 4267387,
    "pka_astrazeneca.csv": None,  # looked up below via metadata list
}


def fetch(fn, fid):
    path = os.path.join(OUT, fn)
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        print(f"  have {fn} ({os.path.getsize(path)} bytes)")
        return
    url = BASE + str(fid)
    try:
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and len(r.content) > 5000:
            with open(path, "wb") as fh:
                fh.write(r.content)
            print(f"  ok {fn} {len(r.content)} bytes")
        else:
            print(f"  FAIL {fn}: {r.status_code} {len(r.content)}b head={r.content[:60]!r}")
    except Exception as e:
        print(f"  FAIL {fn}: {e}")


def main():
    import re
    src = open("/tmp/ptdcsdist/pytdc-1.1.15/tdc/metadata.py").read()
    seg = src[src.find("name2id = {"):src.find("name2idlist")]
    for k, v in re.findall(r'"([a-z0-9_]+)":\s*([0-9]+)', seg):
        for key in ["pka", "ppdb", "amds", "cyp2c19", "cyp2c9_substr",
                    "cyp2d6_substr", "cyp3a4_substr"]:
            if k.startswith(key):
                print("  found:", k, v)
    for fn, fid in IDS.items():
        if fid:
            fetch(fn, fid)


if __name__ == "__main__":
    main()
