"""Monroe L5 (FINAL_PUSH_PLAN 1.3 stretch lineage): frozen-encoder embeddings.

Embeds ft_data.csv + ft_test_smiles.csv SMILES with the public Monroe GRIT
checkpoint (blazejba/Monroe, arXiv:2608.18982) exactly as its run_inference
example does (InChI -> build_single_graph ETKDG+MMFF94 -> Data -> encoder
forward = 720-d graph embedding). Writes cache/monroe_emb.parquet.
Featurizer is ~40 ms/mol; ~6.9k molecules. Failures are skipped + counted
(the ridge probe reindexes by SMILES so partial coverage degrades gracefully).

Run (monroe env): ~/miniforge3/envs/monroe/bin/python src/monroe_embed.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE = os.path.join(ROOT, "cache")
CKPT = "/tmp/monroe_repo/checkpoint"


def main():
    from monroe.model.ckpt import load_ckpt
    from monroe.model.featurizer import build_single_graph
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"), usecols=["SMILES"])
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"), usecols=["SMILES"])
    smis = sorted(set(ft["SMILES"]) | set(test["SMILES"]))
    print(f"{len(smis)} unique SMILES to embed", flush=True)

    encoder = load_ckpt(CKPT)
    encoder.eval()
    device = next(encoder.parameters()).device

    out, failed = {}, 0
    t0 = time.time()
    with torch.no_grad():
        for i, smi in enumerate(smis):
            try:
                inchi = Chem.MolToInchi(Chem.MolFromSmiles(smi))
                if inchi is None:
                    raise ValueError("no inchi")
                g = build_single_graph(inchi=inchi, symmetrize=True)
                pos = np.asarray(g["pos_rdkit"], dtype=np.float32)
                data = Data(
                    x=torch.tensor(g["node_float"], dtype=torch.float32),
                    node_codes=torch.tensor(g["node_codes"], dtype=torch.long),
                    edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
                    edge_codes=torch.tensor(g["edge_codes"], dtype=torch.long),
                    pos_in=torch.tensor(pos, dtype=torch.float32),
                )
                data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
                data = data.to(device)
                emb, _ = encoder(data)
                out[smi] = emb[0].float().cpu().numpy()
            except Exception as e:
                failed += 1
                if failed <= 10:
                    print(f"  fail {smi}: {e}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(smis)} {time.time()-t0:.0f}s failed={failed}", flush=True)
    dim = len(next(iter(out.values())))
    df = pd.DataFrame({smi: v for smi, v in out.items()}).T
    df.index.name = "SMILES"
    df.columns = [f"m{i}" for i in range(dim)]
    df.to_parquet(os.path.join(CACHE, "monroe_emb.parquet"))
    print(f"DONE {df.shape} failed={failed} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
