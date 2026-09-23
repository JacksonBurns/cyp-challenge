"""Extract frozen CheMeLeon MP embeddings for all train + test compounds.

Runs in the `chemeleon` conda env (torch + chemprop 2.x). Loads the pretrained
BondMessagePassing from chemeleon_mp.pt, freezes it, mean-aggregates atom states,
and saves a SMILES-keyed parquet for the cyp-env LightGBM pipeline.

Usage:
  ~/miniforge3/envs/chemeleon/bin/python src/extract_chemeleon.py [out_name.parquet]

Output: cache/chemeleon_emb.parquet  (SMILES, emb_0..emb_{d-1})
Rows = unique parseable SMILES across X_train.parquet + X_test.parquet
(unparseable SMILES are skipped; downstream joins fill zeros).
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, nn
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
WEIGHTS = os.path.expanduser("~/chemeleon_nazarov/chemeleon_mp.pt")
BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_frozen_mp():
    state = torch.load(WEIGHTS, map_location=DEVICE, weights_only=False)
    mp = nn.BondMessagePassing(**state["hyper_parameters"])
    mp.load_state_dict(state["state_dict"])
    mp.to(DEVICE)
    mp.eval()
    for p in mp.parameters():
        p.requires_grad = False
    return mp, mp.output_dim


def main(out_path):
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"), columns=["SMILES"])
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"), columns=["SMILES"])
    smiles = pd.concat([Xtr["SMILES"], Xte["SMILES"]]).drop_duplicates().tolist()
    print(f"unique SMILES: {len(smiles)}", flush=True)

    dps, kept = [], []
    n_fail = 0
    for smi in smiles:
        try:
            dps.append(data.MoleculeDatapoint.from_smi(smi, [0.0]))
            kept.append(smi)
        except Exception:
            n_fail += 1
    print(f"datapoints: {len(dps)} (skipped unparseable: {n_fail})", flush=True)

    mp, d_out = load_frozen_mp()
    print(f"MP loaded output_dim={d_out} device={DEVICE}", flush=True)
    agg = nn.MeanAggregation()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(dps, featurizer)
    loader = DataLoader(dset, batch_size=BATCH, shuffle=False,
                        collate_fn=data.collate_batch, num_workers=0)

    all_embs = []
    t0 = time.time()
    with torch.no_grad():
        for it, batch in enumerate(loader):
            bmg = batch[0]
            bmg.to(DEVICE)  # in-place; returns None
            H_v = mp(bmg)  # chemprop 2.3.1 returns the node-state tensor directly
            emb = agg(H_v, bmg.batch)
            all_embs.append(emb.detach().float().cpu().numpy())
            if it % 20 == 0:
                print(f"  batch {it}/{len(loader)} ({time.time()-t0:.0f}s)", flush=True)
    E = np.vstack(all_embs)
    assert E.shape[0] == len(kept), (E.shape, len(kept))
    df = pd.DataFrame(E, columns=[f"emb_{j}" for j in range(E.shape[1])])
    df.insert(0, "SMILES", kept)
    df.to_parquet(out_path)
    print(f"DONE {df.shape} -> {out_path} ({time.time()-t0:.0f}s)", flush=True)
    print("nan rows:", int(np.isnan(E).any(1).sum()), flush=True)


if __name__ == "__main__":
    out = os.path.join(CACHE, sys.argv[1] if len(sys.argv) > 1 else "chemeleon_emb.parquet")
    main(out)
