"""Download PubChem AID 1851 (NCATS qHTS cytochrome panel) - ranking-only data.

Adapted for our repo from jeremycheminf/openadmet_scripts (public). Uses urllib
(no requests dep needed). The full per-SID CSV carries continuous Fit_LogAC50;
the concise export only has binary outcomes. Pivots to per-CID pIC50-like +
binary per isoform, attaches CanonicalSMILES, drops exact SMILES overlap with
challenge train/test (kept anyway - external data informs only RANKING features,
never the DRC-calibrated placement moments).

Output: external/pubchem_cyp_qhts_aid1851.csv
"""
import io
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "pubchem_cyp_qhts_aid1851.csv")

AID = 1851
ACCESSION_TO_ISO = {
    "NP_000752": "CYP1A2", "NP_000760": "CYP2C19", "NP_000762": "CYP2C9",
    "NP_001020332": "CYP2D6", "NP_059488": "CYP3A4",
}
CONCISE_URL = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/concise/CSV"
FULL_URL = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/CSV"
PROPERTY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/CanonicalSMILES/CSV"
SID_BATCH, CID_BATCH = 3000, 200


def get(url, data=None, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=(urllib.parse.urlencode(data).encode() if data else None))
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode()
        except Exception as e:
            print(f"  retry {i+1}: {e}", flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


def main():
    full_cache = os.path.join(HERE, "pubchem_aid1851_raw.csv")
    if os.path.exists(full_cache):
        full = pd.read_csv(full_cache)
        print(f"loaded raw from cache: {len(full)} rows", flush=True)
    else:
        concise = pd.read_csv(io.StringIO(get(CONCISE_URL)))
        sids = concise["SID"].unique().tolist()
        print(f"AID {AID}: {len(sids)} SIDs, {concise['CID'].nunique()} CIDs", flush=True)

        frames = []
        for i in range(0, len(sids), SID_BATCH):
            txt = get(FULL_URL, data={"sid": ",".join(map(str, sids[i:i + SID_BATCH]))})
            frames.append(pd.read_csv(io.StringIO(txt), skiprows=[1, 2]))
            print(f"  dose-response: {min(i + SID_BATCH, len(sids))}/{len(sids)}", flush=True)
            time.sleep(0.3)
        full = pd.concat(frames, ignore_index=True)
        full.to_csv(full_cache, index=False)

    full = full.rename(columns={"PUBCHEM_CID": "CID", "Panel Target": "target_accession",
                                "PUBCHEM_ACTIVITY_OUTCOME": "outcome"})
    full["iso"] = full["target_accession"].str.split(".").str[0].map(ACCESSION_TO_ISO)
    full = full.dropna(subset=["iso", "CID"])
    full["CID"] = full["CID"].astype(int)
    full["pIC50_like"] = -pd.to_numeric(full["Fit_LogAC50"], errors="coerce")

    potency = full.pivot_table(index="CID", columns="iso", values="pIC50_like", aggfunc="mean")
    binary = full.assign(active=full["outcome"].map({"Active": 1.0, "Inactive": 0.0})).pivot_table(
        index="CID", columns="iso", values="active", aggfunc="first")

    cids = potency.index.tolist()
    print(f"fetching SMILES for {len(cids)} CIDs...", flush=True)

    def fetch_smi(batch, depth=0):
        try:
            txt = get(PROPERTY_URL, data={"cid": ",".join(map(str, batch))}, tries=4)
            return pd.read_csv(io.StringIO(txt))
        except Exception:
            if len(batch) <= 10 or depth > 3:
                return None
            mid = len(batch) // 2
            a, b = fetch_smi(batch[:mid], depth + 1), fetch_smi(batch[mid:], depth + 1)
            return pd.concat([x for x in (a, b) if x is not None], ignore_index=True)

    smi_frames = []
    for i in range(0, len(cids), CID_BATCH):
        f = fetch_smi(cids[i:i + CID_BATCH])
        if f is not None:
            smi_frames.append(f)
        if i % 2000 == 0:
            print(f"  smiles: {i}/{len(cids)}", flush=True)
        time.sleep(0.4)
    smiles_df = pd.concat(smi_frames, ignore_index=True).drop_duplicates(subset="CID").set_index("CID")

    df = potency.add_suffix("_pIC50").join(binary.add_suffix("_active"), how="outer").join(
        smiles_df, how="inner")
    col = [c for c in df.columns if "SMILES" in c.upper()][0]
    df = df.dropna(subset=[col]).reset_index().rename(columns={col: "smiles"})

    ch = set(pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))["SMILES"])
    ch |= set(pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))["SMILES"])
    ch |= set(pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))["SMILES"])
    n0 = len(df)
    df = df[~df["smiles"].isin(ch)].reset_index(drop=True)
    print(f"dropped {n0 - len(df)} exact overlap with challenge", flush=True)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, len(df), "rows", flush=True)


if __name__ == "__main__":
    main()
