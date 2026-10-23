"""Transductive masked-atom pretraining of the chemprop bond-MPNN encoder.

Maeno-style: mask the atomic-number one-hot of 15% of atoms (zero it),
predict the 12-way atomic-number class from the MPNN atom embedding.
Corpus = 750 TEST SMILES (transductive!) + unlabeled external rows +
ChEMBL CYP/homolog actives. Saves cache/pt_encoder.pt for encoder init.

Run: ~/miniforge3/envs/chemeleon/bin/python src/pt_encoder.py [--epochs 10]
"""
import os
import sys

import pandas as pd
import torch
import lightning as pl
from lightning import seed_everything

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import ft_ext  # noqa: E402
from chemprop import data, nn  # noqa: E402


MASK_FRAC = 0.15
ONEHOT_SL = slice(1, 12)   # atomic-number one-hot inside the 72-wide atom feat
NCLS = 11                   # classes 0..10 = atomic number cls (argmax + 1)
BATCH, LR, WD = 64, 1e-3, 1e-5


def corpus():
    ft, test, allrows = ft_ext.load_tables()
    smis = set(test["SMILES"])
    smis |= set(allrows.loc[allrows["_src"] != "challenge", "SMILES"])
    for f in ["chembl_cyp_ic50.csv", "chembl_cyp_homolog_ic50.csv"]:
        p = os.path.join(HERE, "..", "external", f)
        if os.path.exists(p):
            smis |= set(pd.read_csv(p)["canonical_smiles"].dropna())
    return sorted(s for s in smis if s)


class PTModel(pl.LightningModule):
    def __init__(self, mp, head, **kw):
        super().__init__(**kw)
        self.mp, self.head = mp, head

    def step(self, batch):
        bmg = batch.bmg
        bmg.to(self.device)
        V = bmg.V
        tgt = V[:, ONEHOT_SL].argmax(dim=1)
        m = torch.rand(V.shape[0], device=V.device) < MASK_FRAC
        Vm = V.clone()
        Vm[m, ONEHOT_SL] = 0.0
        bmg.V = Vm
        emb = self.mp(bmg)          # atom-level embeddings, (natoms, dim)
        pred = self.head(emb[m])
        return pred, tgt[m]

    def training_step(self, batch, _):
        pred, tgt = self.step(batch)
        loss = torch.nn.functional.cross_entropy(pred, tgt)
        self.log("train_loss", loss, batch_size=pred.shape[0])
        return loss

    def validation_step(self, batch, _):
        pred, tgt = self.step(batch)
        acc = (pred.argmax(1) == tgt).float().mean()
        self.log("val_acc", acc, batch_size=pred.shape[0])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=WD)


def main(epochs):
    smis = corpus()
    print(f"PT corpus: {len(smis)} SMILES", flush=True)
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(
        [data.MoleculeDatapoint.from_smi(s) for s in smis], featurizer)
    loader = data.build_dataloader(dset, batch_size=BATCH, num_workers=0,
                                   )
    st = torch.load(ft_ext.WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    head = torch.nn.Linear(mp.output_dim, NCLS)
    model = PTModel(mp, head)
    seed_everything(0)
    trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, val_dataloaders=loader,
                         check_val_every_n_epoch=1)
    trainer.fit(model, train_dataloaders=loader)
    torch.save(mp.state_dict(), os.path.join(CACHE, "pt_encoder.pt"))
    print("WROTE cache/pt_encoder.pt", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    main(ap.parse_args().epochs)
