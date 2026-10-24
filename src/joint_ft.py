"""Joint classification+regression model (cross-track multitask).

One shared bond-MPNN encoder with:
  - regression heads: 4 prim pIC50 + 8 aux + 8 bounds, interval hinge loss
    (the zoo hinge_w3 recipe: hinge primary, aux_w=3)
  - classification heads: 2 binary is_TDI (2D6, 3A4), masked BCE
Losses sum with CLASS_W weighting. The two tasks are mechanistically linked
(high-potency occupancy drives TDI positivity); joint training lets the
regression gradients shape features that classify TDI and vice versa.

Outputs:
  cache/joint_oof.csv          - OOF regression preds, inhibition-train rows
  cache/joint_oof_tdi.csv      - OOF is_TDI probs (2D6/3A4), TDI-labeled rows
  cache/joint_test_tdi_probs.csv - full-model test-set is_TDI probs (750)
Family tag 'joint' for the regression pool; probs feed the TDI blend.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning import seed_everything
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
sys.path.insert(0, HERE)
import ft_ext  # noqa: E402
from loss_zoo import set_cols, build_points, BOUNDC, ISO as _ISO  # noqa: E402

TDI_LABELS = ["CYP2D6_is_TDI", "CYP3A4_is_TDI"]
NCLS_T = len(TDI_LABELS)
MAX_EPOCHS, PATIENCE = 60, 10
AUX_W = 3.0
CLASS_W = 1.0
NFOLD = 5
SEED = 0


class JointLoss(torch.nn.Module):
    def __init__(self, n_reg, mu, sd, mu_a, sd_a, w0=0.0, lam=0.3):
        super().__init__()
        self.n_reg = n_reg
        self.mu, self.sd = mu, sd
        self.mu_a, self.sd_a = mu_a, sd_a
        self.w0, self.lam = w0, lam

    def forward(self, preds, targets, mask):
        prim_z = (preds[:, :4] - self.mu) / self.sd
        prim_t = (targets[:, :4] - self.mu) / self.sd
        m4 = mask[:, :4]
        lo_sl = slice(self.n_reg, self.n_reg + 4)
        hi_sl = slice(self.n_reg + 4, self.n_reg + 8)
        lo_z = (preds[:, lo_sl] - self.mu) / self.sd
        hi_z = (preds[:, hi_sl] - self.mu) / self.sd
        bm = mask[:, lo_sl] & mask[:, hi_sl]
        hinge = (torch.relu(prim_t - lo_z) + torch.relu(hi_z - prim_t))
        reg = (hinge * m4).sum() / m4.sum()
        aux = (F.mse_loss(preds[:, 4:self.n_reg], targets[:, 4:self.n_reg],
                          reduction="none") * mask[:, 4:self.n_reg])
        aux = aux.sum() / mask[:, 4:self.n_reg].sum()
        bnd = (F.mse_loss(preds[:, self.n_reg:], targets[:, self.n_reg:],
                          reduction="none") * bm)
        bnd = bnd.sum() / bm.sum()
        return reg + AUX_W * aux + 0.3 * bnd


def loss_joint(preds, targets, mask, reg_loss, tdi_mask, cls_w):
    reg = reg_loss(preds[:, :mask.shape[1]], targets, mask)
    tdi_mask = tdi_mask.bool()
    p_tdi = preds[:, mask.shape[1]:]
    lab = targets[:, mask.shape[1]:]
    ce = F.binary_cross_entropy_with_logits(p_tdi[tdi_mask],
                                            lab[tdi_mask])
    return reg + cls_w * ce, reg.item(), ce.item()


def tdi_aug(fit_all):
    """Merge is_TDI labels into the union-table frame by canonical SMILES."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    def ck(s):
        try:
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m) if m else None
        except Exception:
            return None
    t = pd.read_csv(os.path.join(HERE, "..", "data",
                                 "cyp-challenge-TRAIN_TDI.csv"))
    lab = t[["SMILES"] + TDI_LABELS].drop_duplicates("SMILES")
    lab["ck"] = lab["SMILES"].map(ck)
    lab = lab.dropna(subset=["ck"]).set_index("ck")[TDI_LABELS]
    fit_all = fit_all.copy()
    fit_all["ck"] = fit_all["SMILES"].map(ck)
    for c in TDI_LABELS:
        fit_all[c] = fit_all["ck"].map(lab[c])
    return fit_all


def main(seed=0, smoke=False, auxw=None):
    global AUX_W, MAX_EPOCHS, PATIENCE
    from chemprop import data, nn
    from chemprop.models import MPNN
    from chemprop.nn import losses
    import ft_ext
    AUX_W = 3.0 if auxw is None else auxw
    if auxw is not None:
        AUX_W = auxw
    if smoke:
        MAX_EPOCHS, PATIENCE = 2, 2
        ft_ext.EXT_CAP = 1200
        seed_everything(seed)
    set_cols(hom=False)
    from loss_zoo import COLS as ZCOLS
    ALLC = ZCOLS + TDI_LABELS
    n_reg = len(ZCOLS)
    n_all = n_reg + NCLS_T

    fit_all = tdi_aug(ft_ext.union_table())
    Xs = ft_ext.featurize(fit_all["SMILES"])
    Ys = fit_all[ALLC].values.astype(np.float32)
    lab_mask = (~pd.isna(Ys[:, :n_all])).astype(np.float32)
    ymu = np.nanmean(Ys[:, :n_reg], axis=0)
    ysd = np.nanstd(Ys[:, :n_reg], axis=0)
    Ys = np.where(pd.isna(Ys), 0.0, Ys)
    Ys[:, :n_reg] = (Ys[:, :n_reg] - ymu) / ysd
    a_mu = np.nanmean(fit_all[ft_ext.AUX].values, axis=0)
    a_sd = np.nanstd(fit_all[ft_ext.AUX].values, axis=0)

    kf = KFold(n_splits=1 if smoke else NFOLD, shuffle=True,
               random_state=1234)
    oof_reg = np.full((len(fit_all), 4), np.nan, dtype=np.float32)
    oof_tdi = np.full((len(fit_all), NCLS_T), np.nan, dtype=np.float32)
    folds = []
    for vi, (ti, va) in enumerate(kf.split(np.arange(len(fit_all)))):
        if smoke:
            ti = ti[:2000]
            va = va[:500]
        seed_everything(seed + vi)
        st = torch.load(ft_ext.WEIGHTS, map_location="cpu",
                        weights_only=False)
        mp = nn.BondMessagePassing(**st["hyper_parameters"])
        mp.load_state_dict(st["state_dict"])
        model = MPNN(mp, FFN=ft_ext.make_ffn(mp.output_dim, n_all),
                     loss=losses.MSELoss())
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        reg_loss = JointLoss(n_reg, ymu[:4], ysd[:4], a_mu, a_sd)
        tr = torch.tensor(Xs[ti]), torch.tensor(Ys[ti]), \
            torch.tensor(lab_mask[ti])
        loader = DataLoader(torch.utils.data.TensorDataset(*tr),
                            batch_size=128, shuffle=True, num_workers=0)
        best, best_ep, bad = None, 0, 0
        va_pred = None
        for ep in range(MAX_EPOCHS):
            model.train()
            for xb, yb, mb in loader:
                xb = ft_ext.move_batch(model, xb)
                preds = model(xb)
                loss, _, _ = loss_joint(preds, yb, mb, reg_loss,
                                        mb[:, n_reg:], CLASS_W)
                opt.zero_grad()
                loss.backward()
                opt.step()
            # val: mean CE over TDI rows + hinge over reg rows
            model.eval()
            with torch.no_grad():
                vp = []
                for i in range(0, len(va), 256):
                    idx = va[i:i + 256]
                    xb = ft_ext.move_batch(model, Xs[idx])
                    vp.append(model(xb).cpu().numpy())
                vp = np.concatenate(vp)
            vlab = lab_mask[va]
            tdi_m = vlab[:, n_reg:].sum(axis=1) > 0
            ce = F.binary_cross_entropy_with_logits(
                torch.tensor(vp[:, n_reg:][tdi_m]),
                torch.tensor(Ys[va][:, n_reg:][tdi_m])).item()
            hinge = loss_joint(torch.tensor(vp), torch.tensor(Ys[va]),
                               torch.tensor(vlab), reg_loss,
                               vlab[:, n_reg:], CLASS_W)[2]
            if ce > best:
                best, best_ep = ce, ep
                bad += 1
                if bad >= PATIENCE:
                    break
            else:
                bad = 0
                best = ce
                best_ep = ep
                va_pred = vp
        print(f"[fold {vi}] ce_best={best:.4f} ep={best_ep} "
              f"hinge={hinge:.4f}", flush=True)
        # map z preds back: reg 4 prim
        pred4 = va_pred[:, :4] * ysd[:4] + ymu[:4]
        oof_reg[va] = pred4
        oof_tdi[va] = vp[:, n_reg:]
        folds.append((ti, va))
    # full model on everything (for test TDI probs)
    seed_everything(seed)
    st = torch.load(ft_ext.WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    model = MPNN(mp, FFN=ft_ext.make_ffn(mp.output_dim, n_all),
                 loss=losses.MSELoss())
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    reg_loss = JointLoss(n_reg, ymu[:4], ysd[:4], a_mu, a_sd)
    loader = DataLoader(torch.utils.data.TensorDataset(
        torch.tensor(Xs), torch.tensor(Ys), torch.tensor(lab_mask)),
        batch_size=128, shuffle=True, num_workers=0)
    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb, mb in loader:
            xb = ft_ext.move_batch(model, xb)
            loss, _, _ = loss_joint(model(xb), yb, mb, reg_loss,
                                    mb[:, n_reg:], CLASS_W)
            opt.zero_grad()
            loss.backward()
            opt.step()
    test = pd.read_csv(os.path.join(HERE, "..", "data",
                                    "cyp-challenge-TEST-BLINDED.csv"))
    Xte = ft_ext.featurize(test["SMILES"])
    with torch.no_grad():
        tp = []
        for i in range(0, len(Xte), 256):
            xb = ft_ext.move_batch(model, Xte[i:i + 256])
            tp.append(model(xb).cpu().numpy())
        tp = np.concatenate(tp)
    probs = torch.sigmoid(torch.tensor(tp[:, n_reg:])).numpy()
    out = test[["Molecule_Name"]].copy()
    out["CYP2D6_joint_tdi"] = probs[:, 0]
    out["CYP3A4_joint_tdi"] = probs[:, 1]
    sfx = "" if seed == 0 else f"_s{seed}"
    sfx += "_smoke" if smoke else ""
    out.to_csv(os.path.join(CACHE, f"joint_test_tdi_probs{sfx}.csv"),
               index=False)
    reg_out = fit_all[["SMILES"] + ft_ext.COLS].copy()
    reg_out[ft_ext.COLS] = oof_reg
    reg_out.to_csv(os.path.join(CACHE, f"joint_oof{sfx}.csv"), index=False)
    tdi_out = fit_all[["SMILES"]].copy()
    tdi_out["CYP2D6_is_TDI_oof"] = oof_tdi[:, 0]
    tdi_out["CYP3A4_is_TDI_oof"] = oof_tdi[:, 1]
    tdi_out.to_csv(os.path.join(CACHE, f"joint_oof_tdi{sfx}.csv"),
                   index=False)
    tp_out = fit_all[["SMILES"]].copy()
    tp_out[ft_ext.COLS] = oof_reg
    tp_out.to_csv(os.path.join(CACHE, f"ft_oof_joint{sfx}.csv"), index=False)
    # test preds for the pool (emit)
    tpred = tp[:, :4] * ysd[:4] + ymu[:4]
    tpo = fit_all[["SMILES"]].copy()
    tpo[ft_ext.COLS] = np.nan
    tpo.loc[~tpo["SMILES"].isin(test["SMILES"]), ft_ext.COLS] = tpred
    tpo.to_csv(os.path.join(CACHE, f"ft_test_preds_joint{sfx}.csv"),
               index=False)
    print("JOINT DONE", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--auxw", type=float, default=None)
    a = ap.parse_args()
    main(a.seed, a.smoke, a.auxw)
