"""Phase 1 T1 v3: conjunction as a NEW POOL MEMBER (the correct test).

The conjunction (gate x shift) does not need to beat the v6b bar STANDALONE -
it needs to add an HONEST new member to the v6b pool, exactly how the UMTM
is_TDI heads added +0.024 on 2D6 (sec 22). This script:

1. Trains gate + shift classifiers (OOF, scaffold seed 7) per isoform,
   producing 6145-length probability arrays.
2. Forms the conjunction product (p_gate * p_shift) and min (soft-AND) as new
   members in the standard {iso: (6145,)} npz format.
3. Runs the v6b + conjunction family-block audit (merged + paranoid maps),
   reusing tdi_blend_family_block4 machinery.
4. Reports the capped nested macro vs the 0.3448 bar + per-isoform
   non-regression + paranoid-merge survival.

CPU-only (LightGBM), no GPU.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CACHE = os.path.join(HERE, "..", "cache")
DATA = os.path.join(HERE, "..", "data")
from run_regression import make_folds, scaffold_groups  # noqa: E402

warnings.filterwarnings("ignore")

TDI_ISO = ["CYP2D6", "CYP3A4"]
GATE_THR = 4.301
SHIFT_THR = 0.301
K = 5

clf_params = dict(
    n_estimators=800, learning_rate=0.02, num_leaves=31,
    min_child_samples=20, subsample=0.7, subsample_freq=1,
    colsample_bytree=0.15, reg_lambda=10.0, reg_alpha=5.0,
    n_jobs=-1, random_state=42, verbosity=-1
)


def oof_classifier(X, y, folds, cand_idx, model_fn):
    n = len(X)
    oof = np.full(n, np.nan)
    for f in range(K):
        tr = cand_idx[folds[cand_idx] != f]
        va = cand_idx[folds[cand_idx] == f]
        if len(tr) == 0 or len(va) == 0:
            continue
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return oof


def build_conj_members(feat_name, X):
    """Train gate+shift OOF per isoform, return member dict {iso: prod_arr}."""
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES").set_index("SMILES")
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    smi = Xtr["SMILES"].values
    groups = scaffold_groups(smi.tolist())
    folds = make_folds(groups, seed=7)

    members = {f"conj_{feat_name}": {}, f"conjmin_{feat_name}": {}}
    for iso in TDI_ISO:
        short = iso[3:]
        tdic_col = f"CYP{short}_pIC50_TDI_condition"
        dir_col = f"CYP{short}_pIC50_direct_inhibition"

        tdic = pd.to_numeric(tdi[tdic_col].reindex(smi), errors="coerce").values
        dirv = pd.to_numeric(tdi[dir_col].reindex(smi), errors="coerce").values

        gate_mask = ~np.isnan(tdic)
        gate_y = (tdic > GATE_THR).astype(float)
        gate_idx = np.where(gate_mask)[0]

        both_mask = ~np.isnan(tdic) & ~np.isnan(dirv)
        delta = tdic - dirv
        shift_y = (delta > SHIFT_THR).astype(float)
        shift_idx = np.where(both_mask)[0]

        gate_oof = oof_classifier(X, gate_y, folds, gate_idx, lambda: LGBMClassifier(**clf_params))
        shift_oof = oof_classifier(X, shift_y, folds, shift_idx, lambda: LGBMClassifier(**clf_params))

        # 6145-length arrays: OOF where the arm exists, neutral 0.5 elsewhere
        g = np.full(len(smi), 0.5); g[gate_mask] = gate_oof[gate_mask]
        s = np.full(len(smi), 0.5); s[both_mask] = shift_oof[both_mask]

        prod = g * s
        mn = np.minimum(g, s)
        members[f"conj_{feat_name}"][iso] = prod.astype(np.float64)
        members[f"conjmin_{feat_name}"][iso] = mn.astype(np.float64)
        print(f"  [{feat_name}] {iso}: gate OOF on {gate_mask.sum()} rows, "
              f"shift OOF on {both_mask.sum()} rows", flush=True)
    return members


def main():
    print("Loading features...", flush=True)
    Xtr = pd.read_parquet(os.path.join(CACHE, "X_train.parquet")).drop_duplicates("SMILES").reset_index(drop=True)
    smi = Xtr["SMILES"].values
    base = Xtr.drop(columns=["SMILES"]).values.astype(np.float32)
    cpmed = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet")).set_index("SMILES")
    cpmed_mat = cpmed.loc[smi].drop(columns=["SMILES"], errors="ignore").values.astype(np.float32)
    if cpmed_mat.ndim == 1:
        cpmed_mat = cpmed_mat.reshape(1, -1)
    umtm_files = sorted(
        [os.path.join(CACHE, f) for f in os.listdir(CACHE)
         if f.startswith("umtm_oof_") and f.endswith(".csv")
         and "_ext" not in f and "_primary" not in f]
    )
    umtm = pd.concat([pd.read_csv(f) for f in umtm_files], ignore_index=True)
    umtm_cols = [c for c in umtm.columns if c not in ("SMILES", "fold", "fold_tdi")]
    umtm_avg = umtm.groupby("SMILES")[umtm_cols].mean().reindex(smi).values.astype(np.float32)
    umtm_avg = np.nan_to_num(umtm_avg)

    feat_sets = {
        "bu": np.hstack([base, umtm_avg]),
        "cu": np.hstack([cpmed_mat, umtm_avg]),
    }

    # Build conjunction members for each feature set
    all_members = {}
    for feat_name, X in feat_sets.items():
        print(f"\nBuilding conjunction members (feat={feat_name})...", flush=True)
        all_members.update(build_conj_members(feat_name, X))

    # Save as npz (standard member format)
    for nm, mem in all_members.items():
        path = os.path.join(CACHE, f"tdi_conj_{nm}.npz")
        np.savez(path, **{iso: mem[iso] for iso in TDI_ISO})
        print(f"  saved {os.path.basename(path)}", flush=True)

    # --- Run the family-block audit: v6b + conjunction members ---
    print("\n" + "=" * 70)
    print("FAMILY-BLOCK AUDIT: v6b + conjunction members")
    print("=" * 70)

    from tdi_blend_family_block2 import best_by_frac, nested_blend, fam_share  # noqa: E402

    SRC = {"base": "tdi_oof.npz", "emb": "tdi_emb_oof.npz",
           "tab": "tdi_tabicl_oof.npz", "tabcp": "tdi_tabicl_cp_oof.npz",
           "tabcpext": "tdi_tabicl_cp_ext_oof.npz",
           "umtm_L1": "umtm_L1_tdi.npz", "umtm_L2": "umtm_L2_tdi.npz",
           "umtm_L3": "umtm_L3_tdi.npz", "umtm_L4": "umtm_L4_tdi.npz"}
    for nm in all_members:
        SRC[nm] = f"tdi_conj_{nm}.npz"

    V3 = ["base", "emb", "tab", "tabcp", "tabcpext"]
    V6B = V3 + ["umtm_L1", "umtm_L2", "umtm_L3", "umtm_L4"]
    FAM_CAP = 0.5

    # Family maps: conjunction is a NEW family in merged; in paranoid, merge it
    # into the UMTM/cp lineage (it uses UMTM head features).
    def make_maps(extra_conj):
        merged = {"base": "base", "emb": "emb", "tab": "emb",
                  "tabcp": "cp", "tabcpext": "cp",
                  "umtm_L1": "umtmL1", "umtm_L2": "umtmL2",
                  "umtm_L3": "umtmL3", "umtm_L4": "umtmL4"}
        paranoid = {"base": "base", "emb": "emb", "tab": "emb",
                    "tabcp": "cp", "tabcpext": "cp",
                    "umtm_L1": "emb", "umtm_L2": "cp",
                    "umtm_L3": "cp", "umtm_L4": "dmpnn"}
        for c in extra_conj:
            merged[c] = "conj"          # its own family
            paranoid[c] = "cp"          # paranoid: merge into chemprop/UMTM lineage
        return merged, paranoid

    Xtr_df = Xtr
    tdi = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_TDI.csv")).drop_duplicates("SMILES")
    emx = pd.read_csv(os.path.join(DATA, "cyp-challenge-TRAIN_Emax.csv")).drop_duplicates("SMILES")
    lab_by_smi = pd.concat([tdi[["SMILES"] + [f"{i}_is_TDI" for i in TDI_ISO]],
                            emx[["SMILES"] + [f"{i}_is_TDI" for i in TDI_ISO]]]).drop_duplicates("SMILES").set_index("SMILES")
    Y = lab_by_smi.reindex(Xtr_df["SMILES"])
    folds = make_folds(scaffold_groups(Xtr_df["SMILES"].tolist()), seed=7)

    members = {nm: np.load(os.path.join(CACHE, fn)) for nm, fn in SRC.items()}

    conj_names = [nm for nm in all_members]
    merged_map, paranoid_map = make_maps(conj_names)

    def audit(pool_names, fam, iso, yl, fi, zs):
        pooled, wl = nested_blend(zs, yl, fi, pool_names, fam, cap=None)
        mcc_n, frac_n = best_by_frac(pooled, yl)
        pooled_c, wl_c = nested_blend(zs, yl, fi, pool_names, fam, cap=FAM_CAP)
        mcc_c, frac_c = best_by_frac(pooled_c, yl)
        # LOFO for the conjunction family
        lofo_conj = None
        rest = [k for k in pool_names if fam[k] != "conj"]
        if len(rest) == len(pool_names):
            lofo_conj = None
        else:
            pooled_f, _ = nested_blend({k: zs[k] for k in rest}, yl, fi, rest, fam, cap=None)
            mcc_f, _ = best_by_frac(pooled_f, yl)
            lofo_conj = mcc_f
        return {"nested": round(mcc_n, 4), "capped": round(mcc_c, 4),
                "frac_capped": round(frac_c, 2), "lofo_conj": None if lofo_conj is None else round(lofo_conj, 4)}

    out = {}
    for iso in TDI_ISO:
        y = Y[f"{iso}_is_TDI"].values
        m = ~pd.isna(y)
        yl = np.array([1 if bool(v) else 0 for v in y[m]])
        fi = folds[m]
        zs = {}
        for nm, src in members.items():
            p = src[iso][m].astype(float)
            if p.std() < 1e-9:
                zs[nm] = np.zeros_like(p)
            else:
                zs[nm] = (p - p.mean()) / p.std()
        out[iso] = {}
        # baseline v6b
        out[iso]["v6b"] = {mn: audit(V6B, m_, iso, yl, fi, zs) for mn, m_ in [("merged", merged_map), ("paranoid", paranoid_map)]}
        # v6b + all conjunction members
        pool_all = V6B + conj_names
        out[iso]["v6b+conj"] = {mn: audit(pool_all, m_, iso, yl, fi, zs) for mn, m_ in [("merged", merged_map), ("paranoid", paranoid_map)]}
        print(f"\n  {iso}:")
        print(f"    v6b:        merged {out[iso]['v6b']['merged']}")
        print(f"    v6b:        paranoid {out[iso]['v6b']['paranoid']}")
        print(f"    v6b+conj:   merged {out[iso]['v6b+conj']['merged']}")
        print(f"    v6b+conj:   paranoid {out[iso]['v6b+conj']['paranoid']}")

    # Macro summary
    print("\n" + "=" * 70)
    print("MACRO (capped nested) - v6b vs v6b+conjunction")
    print("=" * 70)
    for pool in ["v6b", "v6b+conj"]:
        for mn in ["merged", "paranoid"]:
            macro = float(np.mean([out[i][pool][mn]["capped"] for i in TDI_ISO]))
            v2d6 = out["CYP2D6"][pool][mn]["capped"]
            v3a4 = out["CYP3A4"][pool][mn]["capped"]
            print(f"  {pool:<12} {mn:<9} macro {macro:.4f}  (2D6 {v2d6:.4f} / 3A4 {v3a4:.4f})")

    # Gate decision
    best = out["CYP2D6"]["v6b+conj"]["paranoid"]["capped"]
    best3a4 = out["CYP3A4"]["v6b+conj"]["paranoid"]["capped"]
    best_macro = (best + best3a4) / 2
    base_macro = (out["CYP2D6"]["v6b"]["paranoid"]["capped"] + out["CYP3A4"]["v6b"]["paranoid"]["capped"]) / 2
    print(f"\n  v6b paranoid baseline macro: {base_macro:.4f}")
    print(f"  v6b+conj paranoid macro:     {best_macro:.4f}  (delta {best_macro-base_macro:+.4f})")
    print(f"  bar: macro 0.3448, 2D6 0.219, 3A4 0.470")
    print(f"  2D6 {best:.4f} vs 0.219 -> {'PASS' if best >= 0.219 else 'FAIL'}")
    print(f"  3A4 {best3a4:.4f} vs 0.470 -> {'PASS' if best3a4 >= 0.470 else 'FAIL'}")
    if best_macro > 0.3448 and best >= 0.219 and best3a4 >= 0.470:
        print(f"\n  *** CONJUNCTION MEMBER CLEARS THE GATE (paranoid) ***")
    else:
        print(f"\n  *** BELOW gate ***")

    with open(os.path.join(CACHE, "phase1_t1_v3_conj_pool_audit.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved phase1_t1_v3_conj_pool_audit.json")


if __name__ == "__main__":
    main()
