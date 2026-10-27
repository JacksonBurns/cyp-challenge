"""CheMeLeon multitask FT: CYP primary + CYP external aux + ADME aux heads.

ADME2 aux-target arm (Oct 27): v1 13 heads + 18 TDC endpoints (4 Veith
CYP binary x4 iso + 3 substrate + 11 ADME) on ~35k external molecules.
Gate: zoo11 pool w/ shuffled control. Skeleton = src/ft_ext.py (proven).
Legacy v1 note: assay-physics aux-task arm (user idea, Oct 26). ADME heads:
  expansion (user, 5326 rows, ZERO overlap w/ challenge): LogD, KSOL,
    HLM_CLint, MLM_CLint, Caco-2 Papp, efflux, MPPB, MBBP, MGMB, Aq_Sol
  ESOL logS (1128), amide rotbar dG (885), Sangster logP (25.7k sheets).
Gate: zoo6 pool-weight + CV ST-RAE gain from adding this arm.
Skeleton = src/ft_ext.py (proven).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from chemprop import data, featurizers, models, nn
from chemprop.nn.metrics import MSE
from lightning import pytorch as pl, seed_everything
from lightning.pytorch.callbacks import EarlyStopping

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
EXT = os.path.join(HERE, "..", "external")
USER = os.path.join(EXT, "user_supplied")
WEIGHTS = os.path.expanduser("~/chemeleon_nazarov/chemeleon_mp.pt")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
ADME = ["LogD", "KSOL", "HLM_CLint", "MLM_CLint", "Papp", "Efflux",
        "MPPB", "MBBP", "MGMB", "Aq_Sol", "logS_esol", "rotbar_dG",
        "logP_sang",
        "caco2_wang", "hia_hou", "pgp", "bbb", "vdss", "ppbr", "aqsol",
        "half_life", "cl_hep", "cl_mic", "lipo_az",
        "cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith", "cyp1a2_veith",
        "cyp2c9_sub", "cyp2d6_sub", "cyp3a4_sub"]
TDC_SETS = ["caco2_wang", "hia_hou", "pgp", "bbb", "vdss", "ppbr",
            "aqsol", "half_life", "cl_hep", "cl_mic", "lipo_az",
            "cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith",
            "cyp1a2_veith", "cyp2c9_sub", "cyp2d6_sub", "cyp3a4_sub"]
TDC_FILES = dict(zip(TDC_SETS, [
    "caco2_wang.csv", "hia_hou.csv", "pgp_broccatelli.csv",
    "bbb_martins.csv", "vdss_lombardo.csv", "ppbr_az.csv", "aqsoldb.csv",
    "half_life_obach.csv", "cl_hepatocyte.csv", "cl_microsome.csv",
    "lipophilicity_az.csv", "cyp2c9_veith.csv", "cyp2d6_veith.csv",
    "cyp3a4_veith.csv", "cyp1a2_veith.csv", "cyp2c9_sub.csv",
    "cyp2d6_sub.csv", "cyp3a4_sub.csv"]))
CYP_AUX = [f"{i}_chembl" for i in ISO] + [f"{i}_pubchem" for i in ISO]
import os as _os
if _os.environ.get("FT_ABLATE") == "no_cyp_tdc":
    # ablation: drop the 4 Veith + 3 substrate TDC heads (pure ADME-physics aux)
    ADME = [a for a in ADME if not (a.endswith("_veith") or a.endswith("_sub"))]
    TDC_SETS = [t for t in TDC_SETS if not (t.endswith("_veith") or t.endswith("_sub"))]
PRIMARY = ISO
AUX = CYP_AUX + ADME
COLS = PRIMARY + AUX
MAX_EPOCHS = 40
PATIENCE = 8
BATCH = 128
VAL_FRAC = 0.1
AUX_W = 0.3
if _os.environ.get("FT_AUXW"):
    AUX_W = float(_os.environ["FT_AUXW"])
EXT_CAP = 14000


def numify(v):
    s = str(v).strip()
    num = ""
    for ch in s:
        if ch in "+-eE.0123456789":
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return np.nan


def canon(s):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def load_adme_rows():
    rows = []
    ex = pd.read_csv(os.path.join(USER, "expansion_data_train.csv"))
    sm = ex["SMILES"].map(canon)
    df = pd.DataFrame({"SMILES": sm})
    KEY = {"LogD": "LogD", "KSOL": "KSOL", "HLM CLint": "HLM_CLint",
           "MLM CLint": "MLM_CLint", "Caco-2 Permeability Papp A>B": "Papp",
           "Caco-2 Permeability Efflux": "Efflux", "MPPB": "MPPB",
           "MBPB": "MBBP", "MGMB": "MGMB"}
    for c, key in KEY.items():
        if c in ex.columns:
            df[key] = ex[c].map(numify)
    if "Aq_Sol" not in df:
        df["Aq_Sol"] = np.nan
    rows.append(df.assign(_src="adme_exp"))
    es = pd.read_csv(os.path.join(USER, "delaney-processed.csv"))
    ec = [c for c in es.columns if "SMILES" in c.upper()][0]
    lc = [c for c in es.columns if "log" in c.lower() and "sol" in c.lower()][0]
    sdf = pd.DataFrame({"SMILES": es[ec].map(canon),
                        "logS_esol": pd.to_numeric(es[lc], errors="coerce")})
    rows.append(sdf.assign(_src="adme_esol"))
    rot = pd.read_csv(os.path.join(USER, "rotational_barrier_xinli.csv"),
                      encoding="utf-8-sig")
    rows.append(pd.DataFrame({"SMILES": rot["SMILES"].map(canon),
                              "rotbar_dG": pd.to_numeric(rot["ΔG(kcal/mol)"],
                                                         errors="coerce")}
                             ).assign(_src="adme_rot"))
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(USER, "sangster_datasets.xlsx"),
                                read_only=True)
    for sn in wb.sheetnames:
        ws = wb[sn]
        it = ws.iter_rows(values_only=True)
        hdr = next(it, None)
        if not hdr:
            continue
        jj = [j for j, h in enumerate(hdr) if h and str(h).strip().upper() == "SMILES"]
        kk = [j for j, h in enumerate(hdr) if h and str(h).strip().lower() in ("logp", "lipophilicity")]
        if not jj or not kk:
            continue
        si, li = jj[0], kk[0]
        dat = [(r[si], r[li]) for r in it if r[si]]
        sdf2 = pd.DataFrame(dat, columns=["s", "p"])
        sdf2["SMILES"] = sdf2["s"].map(canon)
        sdf2["logP_sang"] = pd.to_numeric(sdf2["p"], errors="coerce")
        rows.append(sdf2[["SMILES", "logP_sang"]].assign(_src="adme_sang"))
    TDIR = os.path.join(USER, "tdc")
    for prefix, fn in TDC_FILES.items():
        path = os.path.join(TDIR, fn)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t")
        sc = "Drug" if "Drug" in df.columns else "X"
        t = pd.DataFrame({"SMILES": df[sc].map(canon),
                          prefix: pd.to_numeric(df["Y"], errors="coerce")})
        rows.append(t.dropna().assign(_src="adme_tdc"))
    allad = pd.concat(rows, ignore_index=True)
    for c in ADME:
        if c not in allad:
            allad[c] = np.nan
    allad = allad.dropna(subset=["SMILES"])
    allad = allad.groupby("SMILES").mean(numeric_only=True).reset_index()
    allad["_src"] = "adme"
    return allad


def load_tables():
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    test = pd.read_csv(os.path.join(CACHE, "ft_test_smiles.csv"))
    rows = [ft.assign(_src="challenge")]
    ch = pd.read_csv(os.path.join(EXT, "chembl_cyp_ic50.csv"))
    ch = ch[ch["standard_relation"] == "="]
    ch = ch[ch["standard_units"].isin(["nM", "uM", "µM", "mM", "μM"])]
    ch = ch[ch["standard_value"].between(0.01, 1e5)]
    ch["pic50"] = 9.0 - np.log10(ch["standard_value"])
    g = ch.groupby(["isoform", "canonical_smiles"])["pic50"].median().reset_index()
    wide = g.pivot(index="canonical_smiles", columns="isoform", values="pic50")
    cdf = pd.DataFrame({f"{i}_chembl": wide[i] if i in wide else np.nan for i in ISO})
    cdf["SMILES"] = cdf.index
    rows.append(cdf.assign(_src="chembl"))
    pc = pd.read_csv(os.path.join(EXT, "pubchem_cyp_qhts_aid1851.csv"))
    pdf = pd.DataFrame({f"{i}_pubchem": pd.to_numeric(pc.get(f"{i}_pIC50"), errors="coerce")
                        for i in ISO})
    pdf["SMILES"] = pc["smiles"]
    pdf = pdf.dropna(subset=[c for c in pdf.columns if c != "SMILES"], how="all")
    pdf = pdf.groupby("SMILES").first().reset_index()
    rows.append(pdf.assign(_src="pubchem"))
    rows.append(load_adme_rows())
    allrows = pd.concat(rows, ignore_index=True)
    for c in COLS:
        if c not in allrows:
            allrows[c] = np.nan
    ch_smiles = set(ft["SMILES"]) | set(test["SMILES"])
    adme = allrows[allrows["_src"] == "adme"].drop_duplicates("SMILES")
    core = allrows[(~allrows["_src"].eq("adme"))
                   & ~allrows["SMILES"].isin(ch_smiles)].drop_duplicates("SMILES")
    core = core.assign(_src="cyp_ext")
    allrows = pd.concat([allrows[allrows["_src"] == "challenge"], core, adme],
                        ignore_index=True)
    return ft, test, allrows


def build_points(df):
    pts = []
    for _, row in df.iterrows():
        t = [float(row[c]) if pd.notna(row[c]) else float("nan") for c in COLS]
        pts.append(data.MoleculeDatapoint.from_smi(row["SMILES"], t))
    return pts


def train_model(tr_df, va_df, full_ft, tag, seed=0, pretrained=None):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr_dset = data.MoleculeDataset(build_points(tr_df), featurizer)
    scaler = tr_dset.normalize_targets()
    va_dset = data.MoleculeDataset(build_points(va_df), featurizer)
    va_dset.normalize_targets(scaler)
    tr_loader = data.build_dataloader(tr_dset, batch_size=BATCH, num_workers=0)
    va_loader = data.build_dataloader(va_dset, batch_size=BATCH, num_workers=0, shuffle=False)
    st = torch.load(pretrained or WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    if not full_ft:
        for p in mp.parameters():
            p.requires_grad = False
    counts = tr_df[COLS].notna().sum().values.astype(float)
    prim_scale = (1.0 / np.maximum(counts[:4], 1))
    prim_scale = prim_scale / prim_scale.mean()
    wts = np.concatenate([prim_scale, np.full(len(AUX), AUX_W)])
    ffn = nn.RegressionFFN(input_dim=mp.output_dim, hidden_dim=2048, n_layers=2,
                           n_tasks=len(COLS), dropout=0.1,
                           output_transform=nn.UnscaleTransform.from_standard_scaler(scaler),
                           criterion=MSE(task_weights=wts))
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=False)
    print(f"[{tag}] trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M", flush=True)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[es])
    seed_everything(seed)
    trainer.fit(model, train_dataloaders=tr_loader, val_dataloaders=va_loader)
    return model


def predict(model, smiles_df):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pts = [data.MoleculeDatapoint.from_smi(s, [float("nan")] * len(COLS)) for s in smiles_df["SMILES"]]
    dset = data.MoleculeDataset(pts, featurizer)
    loader = data.build_dataloader(dset, batch_size=BATCH, num_workers=0, shuffle=False)
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d = batch[0], batch[1], batch[2]
            bmg.to(dev)
            y = model(bmg, V_d, X_d)
            outs.append(np.atleast_2d(y.detach().cpu().numpy()))
    return np.vstack(outs)[:, : len(COLS)]


def main(full_ft, seed, skip_folds=False, pretrained=None, prefix="adme",
         smoke=False, epochs=None, patience=None):
    global MAX_EPOCHS, PATIENCE
    if smoke:
        MAX_EPOCHS, PATIENCE = 2, 2
    if epochs:
        MAX_EPOCHS = epochs
    if patience:
        PATIENCE = patience
    sfx = "" if seed == 0 else f"_s{seed}"
    tag = f"{prefix}{sfx}"
    ft, test, allrows = load_tables()
    lab = allrows[COLS].notna().any(axis=1)
    ch_mask = (allrows["_src"] == "challenge").values
    print("rows:", len(allrows), "challenge:", ch_mask.sum(),
          "ext:", (~ch_mask & lab.values).sum(), flush=True)
    oof = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    folds = ft["fold"].values
    t0 = time.time()
    if not skip_folds:
        for f in ([0] if smoke else range(5)):
            va_idx = np.where(folds == f)[0]
            va_df = allrows[ch_mask].iloc[va_idx]
            tr_pool = allrows[ch_mask].drop(index=va_idx).reset_index(drop=True)
            tr_extra = allrows[~ch_mask & lab]
            rng = np.random.default_rng(1000 * seed + f)
            if len(tr_extra) > EXT_CAP:
                keep = rng.choice(len(tr_extra), EXT_CAP, replace=False)
                tr_extra = tr_extra.iloc[keep]
            lab_tr = tr_pool[COLS].notna().any(axis=1).values
            vi = rng.choice(np.where(lab_tr)[0], size=int(lab_tr.sum() * VAL_FRAC), replace=False)
            fit_df = pd.concat([tr_pool.drop(index=vi), tr_extra], ignore_index=True)
            ev_df = tr_pool.iloc[vi]
            model = train_model(fit_df, ev_df, full_ft, f"fold{f}", seed=seed, pretrained=pretrained)
            P = predict(model, va_df)
            for j, iso in enumerate(ISO):
                oof.loc[va_idx, iso] = P[:, j]
            del model
            torch.cuda.empty_cache()
            print(f"fold {f} done {time.time()-t0:.0f}s", flush=True)
    else:
        src = os.path.join(CACHE, f"ft_oof_{tag}.csv")
        prev = pd.read_csv(src)
        for iso in ISO:
            oof[iso] = prev[iso].values
        print("skipped folds; reused", src, flush=True)
    oof.insert(0, "SMILES", ft["SMILES"])
    oof.to_csv(os.path.join(CACHE, f"ft_oof_{tag}.csv"), index=False)

    if smoke:
        print("SMOKE OK: data + fold0 trained", flush=True)
        return
    lab_all = allrows[COLS].notna().any(axis=1)
    tr_pool = allrows[ch_mask].reset_index(drop=True)
    rng = np.random.default_rng(99 + seed)
    l = tr_pool[COLS].notna().any(axis=1).values
    vi = rng.choice(np.where(l)[0], size=int(l.sum() * VAL_FRAC), replace=False)
    ext = allrows[(~ch_mask) & lab_all.values]
    if len(ext) > EXT_CAP:
        ext = ext.iloc[rng.choice(len(ext), EXT_CAP, replace=False)]
    fit_df = pd.concat([tr_pool.drop(index=vi), ext], ignore_index=True)
    model = train_model(fit_df, tr_pool.iloc[vi], full_ft, "all", seed=seed, pretrained=pretrained)
    P = predict(model, test)
    tdf = pd.DataFrame(P[:, :4], columns=ISO)
    tdf.insert(0, "SMILES", test["SMILES"])
    tdf.to_csv(os.path.join(CACHE, f"ft_test_preds_{tag}.csv"), index=False)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ft", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--prefix", default="adme")
    ap.add_argument("--skip-folds", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.full_ft, a.seed, skip_folds=a.skip_folds, pretrained=a.pretrained, prefix=a.prefix, smoke=a.smoke)
