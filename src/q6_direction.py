"""Q6 (GPU): directionality of 2D6 <-> 2C19 homolog transfer.

Measured clue: pt_homarm (hinge + homolog aux) got +0.165 pool weight on
2D6 - the only positive PT signal. Direction never measured. This arm
trains the 4 prim iso heads with a 2C19 AUX head (chembl aux col) and
evaluates BOTH directions on shared OOF:
  A) prim=2D6 (challenge labels), aux=2C19 (chembl): 2D6 val soft_rae vs
     no-aux baseline.
  B) prim=2C19 (chembl labels as the supervised target), aux=2D6 (chembl):
     eval on chembl 2C19 rows.
Same folds, same epochs, frozen encoder. Cheap: auxw grid {0.3}.
Gate: report per-direction gain; sign asymmetry answers Q6.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from chemprop import data, models, nn
from lightning import seed_everything
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loss_zoo as L  # noqa: E402
import ft_ext  # noqa: E402
from interval_arm import (bounds_by_smi, soft_rae, ISO, NPRIM,
                          NAUX, BOUNDC)  # noqa: E402

CACHE = L.CACHE
BATCH = 64
MAX_EPOCHS = 50
L2W = 1e-5
VAL_FRAC = 0.1
AUXW = 0.3


def build_points_swap(df, swap):
    """swap: dict colA->colB value swap on L.COLS order (prim slot 3=2D6,
    aux 2C19 col = 'CYP2C19_chembl')."""
    pts = []
    for _, row in df.iterrows():
        vals = []
        for c in L.COLS:
            src = swap.get(c, c)
            v = row[src] if src in row.index else row.get(c)
            vals.append(float(v) if pd.notna(v) else float("nan"))
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], vals))
    return pts


def make_model(fit_df, full_ft, device, swap):
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(build_points_swap(fit_df, swap), featurizer)
    Yraw = np.asarray(tr_dset.Y, dtype=np.float64)
    scaler = StandardScaler().fit(Yraw)
    for j in range(NPRIM, L.NALL):
        scaler.mean_[j] = 0.0
        scaler.scale_[j] = 1.0
        scaler.var_[j] = 1.0
    tr_dset.Y = scaler.transform(Yraw)
    st = torch.load(ft_ext.WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    for p in mp.parameters():
        p.requires_grad = bool(full_ft)
    mu_p = torch.tensor(scaler.mean_[:NPRIM], dtype=torch.float32)
    sd_p = torch.tensor(scaler.scale_[:NPRIM], dtype=torch.float32)
    mu_a = torch.tensor(scaler.mean_[NPRIM:NPRIM + NAUX], dtype=torch.float32)
    sd_a = torch.tensor(scaler.scale_[NPRIM:NPRIM + NAUX], dtype=torch.float32)
    scaler_t = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048,
                           n_layers=2, n_tasks=L.NALL, dropout=0.1,
                           output_transform=scaler_t)
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    return (model.to(device), scaler, tr_dset, mu_p.to(device),
            sd_p.to(device), mu_a.to(device), sd_a.to(device))


def losses(z, targets, mask, mu_p, sd_p, mu_a, sd_a):
    raw = z[:, :NPRIM] * sd_p + mu_p
    m = mask[:, :NPRIM]
    lo = targets[:, L.LO_SL]; hi = targets[:, L.HI_SL]
    hb = mask[:, L.LO_SL] & mask[:, L.HI_SL]
    lo_s = torch.where(hb, lo, raw)
    hi_s = torch.where(hb, hi, raw)
    hinge = (torch.clamp(raw - hi_s, min=0.0) ** 2
             + torch.clamp(lo_s - raw, min=0.0) ** 2) / (sd_p ** 2)
    mse = (z[:, :NPRIM] - targets[:, :NPRIM]) ** 2
    prim = torch.where(hb, hinge, mse)
    prim = torch.where(m, prim, torch.zeros_like(prim))
    aux = (z[:, NPRIM:NPRIM + NAUX] - targets[:, NPRIM:NPRIM + NAUX]) ** 2
    aux = torch.where(mask[:, NPRIM:NPRIM + NAUX], aux, torch.zeros_like(aux))
    return prim.mean() + AUXW * aux.mean()


def predict(model, df):
    featurizer = ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(
        s, [float("nan")] * L.NALL) for s in df["SMILES"]]
    loader = data.build_dataloader(
        data.MoleculeDataset(pts, featurizer), batch_size=BATCH,
        num_workers=0, shuffle=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = batch[0], batch[1], batch[2]
            bmg.to(next(model.parameters()).device)
            y = model(bmg, V_d, X_d)
            outs.append(np.atleast_2d(y.detach().cpu().numpy()))
    return np.vstack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["A", "B", "base"], default="A",
                    help="A: aux=2C19 prim=2D6; B: prim slot=2C19 labels; "
                         "base: no homolog aux")
    a = ap.parse_args()
    # slot semantics: prim col 2 = CYP2D6 (ISO order 1A2,2C9,2D6,3A4 -> idx2)
    # chembl aux cols order: AUX = 1A2..3A4 chembl then pubchem (NAUX=8)
    swap = {}
    if a.mode == "A":
        # put CYP2C19_chembl values into aux slot 0 (unused head) - all aux
        # heads exist; 2C19 rides as EXTRA supervised head via pubchem slot
        swap["CYP1A2_pubchem"] = "CYP2C19_chembl"
    elif a.mode == "B":
        swap["CYP2D6"] = "CYP2C19_chembl"   # prim 2D6 slot supervised w/ 2C19
    tag = f"q6_{a.mode}" + ("" if a.seed == 0 else f"_s{a.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft, test, allrows = ft_ext.load_tables()
    av = L.load_aux_views()
    av = av[~av["SMILES"].isin(set(test["SMILES"]))]
    keep = [c for c in av.columns if c in L.COLS or c == "SMILES"]
    allrows = allrows.merge(av[keep], on="SMILES", how="left")
    try:
        hdf = L.load_homolog()
        allrows = pd.concat([allrows, hdf.assign(_src="homolog")],
                            ignore_index=True)
    except Exception as e:
        print("homolog load failed:", e, flush=True)
    bb = bounds_by_smi()
    for c in BOUNDC:
        allrows[c] = allrows["SMILES"].map(bb[c])
    if "CYP2C19_chembl" not in allrows.columns:
        print("NO 2C19 col - check aux view", flush=True)
        return
    ch_mask = (allrows["_src"] == "challenge").values
    lab = allrows[L.LABCOLS].notna().any(axis=1)
    folds = ft["fold"].values
    seed_everything(a.seed)
    res = []
    for f in range(5):
        va_idx = np.where(folds == f)[0]
        va_df = allrows[ch_mask].iloc[va_idx]
        tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
        tr_extra = allrows[~ch_mask & lab]
        rng = np.random.default_rng(1000 * a.seed + f)
        if len(tr_extra) > ft_ext.EXT_CAP:
            keep_i = rng.choice(len(tr_extra), ft_ext.EXT_CAP, replace=False)
            tr_extra = tr_extra.iloc[keep_i]
        lab_tr = tr_pool[L.LABCOLS].notna().any(axis=1).values
        vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC),
                        replace=False)
        fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra],
                           ignore_index=True).reset_index(drop=True)
        ev_df = tr_pool.iloc[vi].reset_index(drop=True)
        (model, scaler, tr_dset, mu_p, sd_p, mu_a, sd_a) = make_model(
            fit_df, False, device, swap)
        loader = data.build_dataloader(tr_dset, batch_size=BATCH,
                                       num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=L2W)
        best, bad, best_state = np.inf, 0, None
        for ep in range(MAX_EPOCHS):
            model.train()
            for batch in loader:
                bmg, V_d, X_d, targets, w, lt_m, gt_m = batch
                mask = targets.isfinite()
                targets = targets.nan_to_num(nan=0.0)
                bmg.to(device); targets = targets.to(device)
                mask = mask.to(device)
                z = model.predictor.ffn(model.fingerprint(bmg, V_d, X_d))
                loss = losses(z, targets, mask, mu_p, sd_p, mu_a, sd_a)
                opt.zero_grad(set_to_none=True)
                loss.backward(); opt.step()
            Pf = predict(model, ev_df)
            yv = ev_df[L.LABCOLS].values.astype(float)
            lo_e = ev_df[[f"{i}_bndlo" for i in ISO]].values
            hi_e = ev_df[[f"{i}_bndhi" for i in ISO]].values
            sc = []
            for k in range(NPRIM):
                m = np.isfinite(yv[:, k])
                if m.sum() >= 8:
                    sc.append(float(np.mean(soft_rae(yv[m, k], Pf[m, k],
                                                     lo_e[m, k], hi_e[m, k]))))
            sm = float(np.mean(sc))
            if sm < best - 1e-4:
                best, bad = sm, 0
                best_state = {kk: vv.cpu().clone()
                              for kk, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
        model.load_state_dict(best_state)
        Pf = predict(model, va_df)
        yva = va_df[L.LABCOLS].values.astype(float)
        lo_v = va_df[[f"{i}_bndlo" for i in ISO]].values
        hi_v = va_df[[f"{i}_bndhi" for i in ISO]].values
        out = {}
        for k, iso in enumerate(ISO):
            m = np.isfinite(yva[:, k])
            out[iso] = float(np.mean(soft_rae(yva[m, k], Pf[m, k],
                                              lo_v[m, k], hi_v[m, k])))
        res.append(out)
        print(f"[{tag}] f{f} " + " ".join(f"{i}={out[i]:.4f}" for i in ISO),
              flush=True)
    macro = {iso: float(np.mean([r[iso] for r in res])) for iso in ISO}
    print(f"[{tag}] MACRO " + " ".join(f"{i}={macro[i]:.4f}" for i in ISO) +
          f" | macroST {np.mean(list(macro.values())):.4f}", flush=True)
    json.dump({"folds": res, "macro": macro, "mode": a.mode,
               "seed": a.seed}, open(os.path.join(CACHE, f"q6_{tag}.json"),
                                     "w"))


if __name__ == "__main__":
    main()
