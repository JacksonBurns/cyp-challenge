"""Train per-isoform + stacked multi-task LightGBM regressors; grouped-CV OOF eval.

Outputs:
  cache/oof_direct.csv   OOF predictions per isoform (rows = labeled compounds)
  cache/raw_test_preds.csv  raw (uncalibrated) test predictions, 750 rows
  cache/cv_results.json   metrics for every config tried
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, RDLogger
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import mol_of  # noqa: E402

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "cache")

ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
DIRECT = {i: f"{i}_pIC50_direct_inhibition" for i in ISOFORMS}
TDIARM = {i: f"{i}_pIC50_TDI_condition" for i in ISOFORMS}
FP_COLS = [f"m2_{i}" for i in range(2048)]
K = 5


def murcko_key(m):
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def scaffold_groups(smiles_list):
    keys = []
    for smi in smiles_list:
        m = mol_of(smi)
        keys.append(murcko_key(m) if m is not None else "")
    # empty-scaffold compounds: split into pseudo-groups so they don't form one giant group
    fixed, bucket = [], 0
    for k in keys:
        if k == "":
            fixed.append(f"__NONE__{bucket // 40}")
            bucket += 1
        else:
            fixed.append(k)
    return fixed


def make_folds(group_keys, k=K, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(group_keys, return_inverse=True)
    sizes = np.bincount(inv)
    order = rng.permutation(len(uniq))
    # shuffle group order, sort descending by size for greedy balancing
    order = order[np.argsort(-sizes[order])]
    fold = np.empty(len(group_keys), dtype=int)
    load = np.zeros(k)
    for g in order:
        f = int(np.argmin(load))
        fold[inv == g] = f
        load[f] += sizes[g]
    return fold


def soft_rae(y, yhat, lo, hi):
    err = np.clip(yhat - hi, 0, None) + np.clip(lo - yhat, 0, None)
    base = np.clip(np.mean(y) - hi, 0, None) + np.clip(lo - np.mean(y), 0, None)
    return err.sum() / base.sum()


def build_all():
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet"))
    Xte = pd.read_parquet(os.path.join(CACHE, "X_test.parquet"))
    # featurize concatenated inhibition+TDI SMILES then dropped dup on Molecule_Name only;
    # SMILES can repeat -> dedupe feature rows by SMILES before anything is indexed by them
    Xtr = Xtr.drop_duplicates("SMILES").reset_index(drop=True)
    inh = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_inhibition.csv"))
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv"))
    test = pd.read_csv(os.path.join(DATA, "cyp-challenge-TEST-BLINDED.csv"))
    inh = inh.drop_duplicates("SMILES")
    tdi = tdi.drop_duplicates("SMILES")

    inh_i = inh.set_index("SMILES")
    tdi_i = tdi.set_index("SMILES")

    targets = pd.DataFrame(index=Xtr["SMILES"].values)
    bounds = pd.DataFrame(index=Xtr["SMILES"].values)
    for iso in ISOFORMS:
        d = inh_i[DIRECT[iso]].reindex(Xtr["SMILES"])
        d2 = tdi_i[DIRECT[iso]].reindex(Xtr["SMILES"])
        targets[DIRECT[iso]] = d.combine_first(d2)
        bounds[f"{iso}_lo"] = inh_i[f"{iso}_pIC50_direct_inhibition_conf_low"].reindex(Xtr["SMILES"])
        bounds[f"{iso}_hi"] = inh_i[f"{iso}_pIC50_direct_inhibition_conf_high"].reindex(Xtr["SMILES"])
        targets[TDIARM[iso]] = tdi_i[TDIARM[iso]].reindex(Xtr["SMILES"])

    # Tanimoto binary morgan sim matrices (train x train, test x train)
    Btr = (Xtr[FP_COLS].values > 0).astype(np.float32)
    Bte = (Xte[FP_COLS].values > 0).astype(np.float32)

    def sim(A, Bb):
        d = A @ Bb.T
        ca = (A > 0).sum(1)[:, None]
        cb = (Bb > 0).sum(1)[None, :]
        return d / np.maximum(ca + cb - d, 1e-9)

    t0 = time.time()
    S = sim(Btr, Btr)
    Ste = sim(Bte, Btr)
    print(f"sim matrices in {time.time()-t0:.0f}s", flush=True)
    return Xtr, Xte, targets, bounds, S, Ste, test


def nn_block(qS, cand_idx, yvals, q_idx=None):
    """Per query row: max sim to labeled candidates, Tanimoto-weighted top-5 label mean,
    top-10 mean. qS: (nq, ntrain) sims. yvals: labels (NaN unlabeled). q_idx: global row
    ids of queries (to blank out self-matches where query is also a candidate)."""
    S2 = qS[:, cand_idx].copy()
    if q_idx is not None:
        pos_of = {c: j for j, c in enumerate(cand_idx)}
        for i, q in enumerate(q_idx):
            j = pos_of.get(q)
            if j is not None:
                S2[i, j] = -1.0
    lab = ~np.isnan(yvals[cand_idx])
    S2 = S2[:, lab]
    yl = yvals[cand_idx][lab]
    n = S2.shape[1]
    k10 = min(10, n)
    part = np.argpartition(-S2, k10 - 1, axis=1)[:, :k10]
    rows = np.arange(S2.shape[0])[:, None]
    topS = S2[rows, part]  # (nq, k10) unsorted
    maxsim = topS[:, :5].max(1)
    w5 = np.clip(topS[:, :5], 0, None) ** 2
    top5y = yl[part[:, :5]]
    wavg = np.nansum(np.where(np.isnan(top5y), 0, w5) * np.nan_to_num(top5y), 1) / np.maximum(
        np.nansum(np.where(np.isnan(top5y), 0, w5), 1), 1e-9
    )
    w10 = np.clip(topS, 0, None) ** 2
    top10y = yl[part]
    avg10 = np.nanmean(top10y, axis=1)
    return np.column_stack([maxsim, wavg, avg10])


def lgbm(**over):
    p = dict(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    p.update(over)
    return LGBMRegressor(**p)


def cv_config(name, Xall_base, nn_iso, targets, bounds, folds, iso_targets, aux, params, seeds=1):
    """iso_targets: list of direct isoforms to evaluate. aux: list of tdiarm isoforms to add as stacked tasks (or None)."""
    oof = {}
    percol = {}
    if aux is None:  # per-isoform models
        eval_isos = iso_targets
        tasks = [(iso, DIRECT[iso], None) for iso in eval_isos]
    else:  # stacked
        eval_isos = iso_targets
        tasks = [("STACK", None, None)]
    for iso in eval_isos:
        y = targets[DIRECT[iso]].values
        lab = ~np.isnan(y)
        oof[iso] = np.full(len(y), np.nan)
    for f in range(K):
        tr_mask = folds != f
        if aux is None:
            for iso in eval_isos:
                y = targets[DIRECT[iso]].values
                cand = np.where((~np.isnan(y)) & (folds != f))[0]
                # queries: all train rows except val-fold candidates leakage? NN features must
                # avoid using val-fold labels; candidates already restricted. Train queries ok.
                qtr = np.where(folds != f)[0]
                qval = np.where(folds == f)[0]
                qb = np.vstack([nn_block(S_ALL[qtr], cand, y, q_idx=qtr),
                                nn_block(S_ALL[qval], cand, y, q_idx=qval)])
                qb = qb.astype(np.float32)
                Xf_tr = np.hstack([Xall_base[qtr], qb[: len(qtr)]])
                Xf_val = np.hstack([Xall_base[qval], qb[len(qtr):]])
                trl = ~np.isnan(y[qtr])
                preds = np.zeros(len(qval))
                for s in range(seeds):
                    m = lgbm(random_state=42 + s, **params)
                    m.fit(Xf_tr[trl], y[qtr][trl])
                    preds += m.predict(Xf_val)
                preds /= seeds
                oof[iso][qval] = preds
        else:
            # stacked: rows = (compound, task) pairs
            tcols = [DIRECT[i] for i in ISOFORMS] + [TDIARM[i] for i in aux]
            rows_tr, rows_val, ys_tr, ys_val = [], [], [], []
            qval_by_ti = {}
            for ti, tcol in enumerate(tcols):
                y = targets[tcol].values
                cand = np.where((~np.isnan(y)) & (folds != f))[0]
                qtr = np.where((~np.isnan(y)) & (folds != f))[0]
                qval = np.where((~np.isnan(y)) & (folds == f))[0]
                qval_by_ti[ti] = qval
                for qset, bucket in ((qtr, rows_tr), (qval, rows_val)):
                    nb = nn_block(S_ALL[qset], cand, y, q_idx=qset)
                    nb = np.hstack([nb, np.full((len(qset), 1), ti, dtype=np.float32)])
                    bucket.append(np.hstack([Xall_base[qset], nb.astype(np.float32)]))
                ys_tr.append(y[qtr])
                ys_val.append(y[qval])
            Xf_tr = np.vstack(rows_tr)
            Xf_val = np.vstack(rows_val)
            y_tr = np.concatenate(ys_tr)
            y_val = np.concatenate(ys_val)
            ti_val = np.concatenate([np.full(len(a), ti) for ti, a in enumerate(ys_val)])
            val_pred = np.zeros(len(y_val))
            for s in range(seeds):
                m = lgbm(random_state=42 + s, **params)
                m.fit(Xf_tr, y_tr)
                val_pred += m.predict(Xf_val)
            val_pred /= seeds
            for ti, tcol in enumerate(tcols):
                iso = tcol.split("_pIC50")[0]
                if iso in eval_isos and tcol == DIRECT[iso]:
                    sel = ti_val == ti
                    oof[iso][qval_by_ti[ti]] = val_pred[sel]
    # metrics
    summary = {}
    for iso in eval_isos:
        y = targets[DIRECT[iso]].values
        lab = ~np.isnan(y)
        lo, hi = bounds[f"{iso}_lo"].values, bounds[f"{iso}_hi"].values
        r = soft_rae(y[lab], oof[iso][lab], lo[lab], hi[lab])
        summary[iso] = {
            "st_rae": float(r),
            "mae": float(np.abs(y[lab] - oof[iso][lab]).mean()),
            "pearson": float(pearsonr(y[lab], oof[iso][lab]).statistic),
            "n": int(lab.sum()),
        }
    summary["MA_st_rae"] = float(np.mean([summary[i]["st_rae"] for i in eval_isos]))
    print(json.dumps({name: summary}, indent=1), flush=True)
    return name, summary, oof


if __name__ == "__main__":
    Xtr, Xte, targets, bounds, S, Ste, test = build_all()
    S_ALL, STE_ALL = S, Ste
    Xbase = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    groups = scaffold_groups(Xtr["SMILES"].tolist())
    folds = make_folds(groups)
    print("fold sizes per scaffold-group ok; largest fold frac:", np.bincount(folds).max() / len(folds))

    results = {}
    store = {}
    configs = [
        ("periso", None),
        ("stacked3a4aux", ["CYP3A4"]),
    ]
    for name, aux in configs:
        n, s, oof = cv_config(name, Xbase, None, targets, bounds, folds, ISOFORMS, aux, {})
        results[n] = s
        store[n] = oof
    with open(os.path.join(CACHE, "cv_results_stage1.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    pd.DataFrame(store["periso"]).to_csv(os.path.join(CACHE, "oof_stage1.csv"))
    np.save(os.path.join(CACHE, "folds.npy"), folds)
