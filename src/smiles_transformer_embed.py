"""R3/T3 shared: SMILES-sequence transformer embeddings (ChemBERTa).

Embeds all 6895 train+test SMILES with a FROZEN pretrained SMILES-sequence
transformer (ChemBERTa-zinc-base-v1, 768-d, mean-pooled) -> cache/emb_smiles_all.parquet.
This is the one representation class we lack: every pool member is a 2D graph
MPNN; a SMILES text transformer is genuinely decorrelated (stir_bar + jeremy both
use it). Shared by R3 (regression ridge/TabPFN) and T3 (TDI TabICL).

Env: smtransformer (python 3.11, transformers, torch cu130). GPU.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
MODEL = "seyonec/ChemBERTa-zinc-base-v1"
BATCH = 64


def main():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES")
    test = pd.read_csv(os.path.join(HERE, "..", "data", "cyp-challenge-TEST-BLINDED.csv"))
    smi_all = list(pd.unique(pd.concat([Xtr["SMILES"], test["SMILES"]])))
    print(f"embedding {len(smi_all)} SMILES with {MODEL}...", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    print(f"device: {dev}", flush=True)

    outs = []
    t0 = time.time()
    for i in range(0, len(smi_all), BATCH):
        batch = smi_all[i:i + BATCH]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=128).to(dev)
        with torch.no_grad():
            out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        outs.append(emb.cpu().numpy())
        if i % (BATCH * 20) == 0:
            print(f"  {i}/{len(smi_all)} {time.time()-t0:.0f}s", flush=True)
    E = np.vstack(outs)
    dall = pd.DataFrame(E, index=pd.Index(smi_all, name="SMILES")).reset_index()
    fp = os.path.join(CACHE, "emb_smiles_all.parquet")
    dall.to_parquet(fp)
    print(f"saved {os.path.basename(fp)} {dall.shape}", flush=True)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
