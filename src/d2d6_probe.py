"""2D6 representation-ceiling probe (CPU, honest nested).

Tests whether pKa/protonation + stereochemistry features - chemical
information that 2D graph embeddings capture poorly - can break the
CYP2D6 ceiling that all 5 data-driven lineages saturate at ~0.42-0.46.

Features (all computed from SMILES via RDKit, NO labels used):
  pKa block:
    - n_basic_N: count of basic (non-amide, non-aromatic) nitrogens
    - n_acidic_N: count of acidic nitrogens
    - pKa_max: max estimated pKa among basic Ns
    - pKa_mean: mean estimated pKa among basic Ns
    - frac_protonated_74: fraction of basic Ns protonated at pH 7.4
    - net_charge_74: estimated net charge at pH 7.4
  stereo block:
    - n_stereocenters: potential stereocenters
    - n_stereo_defined: defined stereocenters
    - n_stereo_undef: undefined stereocenters
    - n_EZ_bonds: E/Z double bonds
    - has_stereo: binary
  chemical block (2D6-selective motifs):
    - basic_amine_aromatic: 1 if has basic N AND >=1 aromatic ring
    - n_aromatic_rings
    - logP
    - MW
    - TPSA
    - rotatable_bonds
    - sp3_fraction
    - n_tertiary_amines
    - n_secondary_amines
    - n_primary_amines

Probe (exact repo recipe, mordred_probe.py):
  (A) cpmed 600-d embedding only
  (B) cpmed 600-d + new features (concat)
  (C) new features only
  Per-isoform 5-fold scaffold OOF ridge, nested = Fisher-z avg of Pearson,
  macro R2 = mean(per-isoform nested^2).

GATE: (B) beats (A) on CYP2D6 nested Pearson by >= +0.003.

Run: ~/miniforge3/envs/mordred/bin/python src/d2d6_probe.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Descriptors import MolLogP, TPSA, MolWt, NumRotatableBonds
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
ALPHAS = [100.0, 1000.0, 10000.0]
K = 5
PHYS_PH = 7.4

# --- pKa estimation via SMARTS fragments ---
# Basic nitrogens: not in amides, not in aromatic rings (those are weak bases)
BASIC_N_SMARTS = [
    # Tertiary aliphatic amine
    ("tertiary_amine", "[NX3;H0;+0;!$(*=O)]", 10.5),
    # Secondary aliphatic amine
    ("secondary_amine", "[NX3;H1;+0;!$(*=O)]", 10.5),
    # Primary aliphatic amine
    ("primary_amine", "[NX3;H2;+0;!$(*=O)]", 10.5),
    # Quaternary ammonium (always charged)
    ("quaternary_N", "[NX4;+1]", 12.0),
]
# Aromatic/weakly-basic nitrogens
AROMATIC_N_SMARTS = [
    ("pyridine_N", "[nX3;+0;!H0]", 5.2),
    ("aniline_N", "[NX3;H1;+0;$(*~c)]", 4.6),
]
# Acidic nitrogens
ACIDIC_N_SMARTS = [
    ("carboxylic_acid_N", "[NX3;H1;$(*=O)]", 2.0),  # amide N (very weak acid)
]

# Pre-compile
BASIC_PATTERNS = [(name, Chem.MolFromSmarts(smi), pka) for name, smi, pka in BASIC_N_SMARTS]
AROMATIC_PATTERNS = [(name, Chem.MolFromSmarts(smi), pka) for name, smi, pka in AROMATIC_N_SMARTS]
ACIDIC_PATTERNS = [(name, Chem.MolFromSmarts(smi), pka) for name, smi, pka in ACIDIC_N_SMARTS]


def compute_pka_features(mol):
    """Compute pKa/protonation features for a single molecule."""
    if mol is None:
        return {k: np.nan for k in [
            "n_basic_N", "n_acidic_N", "pKa_max", "pKa_mean",
            "frac_protonated_74", "net_charge_74",
            "n_tertiary_amine", "n_secondary_amine", "n_primary_amine",
        ]}

    basic_pkas = []
    n_tert = n_sec = n_prim = 0

    for name, pat, pka in BASIC_PATTERNS:
        matches = mol.GetSubstructMatches(pat)
        if "tertiary" in name:
            n_tert = len(matches)
        elif "secondary" in name:
            n_sec = len(matches)
        elif "primary" in name:
            n_prim = len(matches)
        for _ in matches:
            basic_pkas.append(pka)

    # Aromatic Ns (weak bases)
    for name, pat, pka in AROMATIC_PATTERNS:
        matches = mol.GetSubstructMatches(pat)
        for _ in matches:
            basic_pkas.append(pka)

    # Acidic Ns
    n_acidic = 0
    for name, pat, pka in ACIDIC_PATTERNS:
        matches = mol.GetSubstructMatches(pat)
        n_acidic += len(matches)

    n_basic = len(basic_pkas)
    pka_max = max(basic_pkas) if basic_pkas else np.nan
    pka_mean = np.mean(basic_pkas) if basic_pkas else np.nan

    # Fraction protonated at pH 7.4 (Henderson-Hasselbalch)
    if basic_pkas:
        frac_prot = np.mean([1.0 / (1.0 + 10 ** (PHYS_PH - p)) for p in basic_pkas])
    else:
        frac_prot = 0.0

    # Net charge at pH 7.4: basic Ns protonated (+1 each) - acidic groups (-1 each)
    # Approximate: basic Ns contribute +frac_prot each, carboxylic acids -0.9
    n_carbox = len(mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[O-]"))) + \
               len(mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)O")))
    net_charge = n_basic * frac_prot - n_carbox * 0.9

    return {
        "n_basic_N": n_basic,
        "n_acidic_N": n_acidic,
        "pKa_max": pka_max,
        "pKa_mean": pka_mean,
        "frac_protonated_74": frac_prot,
        "net_charge_74": net_charge,
        "n_tertiary_amine": n_tert,
        "n_secondary_amine": n_sec,
        "n_primary_amine": n_prim,
    }


def compute_stereo_features(mol):
    """Compute stereochemistry features."""
    if mol is None:
        return {k: np.nan for k in [
            "n_stereocenters", "n_stereo_defined", "n_stereo_undef",
            "n_EZ_bonds", "has_stereo",
        ]}

    # Potential stereocenters
    potential = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    n_potential = len(potential)
    # Defined stereocenters
    defined = Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
    n_defined = len(defined)
    n_undef = n_potential - n_defined

    # E/Z double bonds
    n_ez = 0
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOE or bond.GetStereo() == Chem.BondStereo.STEREOZ:
            n_ez += 1

    return {
        "n_stereocenters": n_potential,
        "n_stereo_defined": n_defined,
        "n_stereo_undef": n_undef,
        "n_EZ_bonds": n_ez,
        "has_stereo": 1.0 if (n_potential > 0 or n_ez > 0) else 0.0,
    }


def compute_chem_features(mol):
    """Compute 2D6-selective chemical features."""
    if mol is None:
        return {k: np.nan for k in [
            "basic_amine_aromatic", "n_aromatic_rings", "logP", "MW",
            "TPSA", "rotatable_bonds", "sp3_fraction",
        ]}

    n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    logp = MolLogP(mol)
    mw = MolWt(mol)
    tpsa = TPSA(mol)
    rot = NumRotatableBonds(mol)
    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    # Basic amine + aromatic: the 2D6 "dextromethorphan-like" motif
    has_basic_n = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3;H0,H1,H2;+0;!$(*=O)]"))) > 0
    basic_amine_aromatic = 1.0 if (has_basic_n and n_aromatic > 0) else 0.0

    return {
        "basic_amine_aromatic": basic_amine_aromatic,
        "n_aromatic_rings": n_aromatic,
        "logP": logp,
        "MW": mw,
        "TPSA": tpsa,
        "rotatable_bonds": rot,
        "sp3_fraction": sp3,
    }


def compute_all_features(smiles_list):
    """Compute all new features for a list of SMILES. Returns DataFrame."""
    t0 = time.time()
    rows = []
    n_fail = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_fail += 1
            row = {k: np.nan for k in [
                "n_basic_N", "n_acidic_N", "pKa_max", "pKa_mean",
                "frac_protonated_74", "net_charge_74",
                "n_tertiary_amine", "n_secondary_amine", "n_primary_amine",
                "n_stereocenters", "n_stereo_defined", "n_stereo_undef",
                "n_EZ_bonds", "has_stereo",
                "basic_amine_aromatic", "n_aromatic_rings", "logP", "MW",
                "TPSA", "rotatable_bonds", "sp3_fraction",
            ]}
        else:
            row = {**compute_pka_features(mol),
                   **compute_stereo_features(mol),
                   **compute_chem_features(mol)}
            Chem.SanitizeMol(mol)
            mol = None
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"  features: {i+1}/{len(smiles_list)} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows, index=smiles_list)
    print(f"  features done: {len(df)} rows, {n_fail} parse failures, {time.time()-t0:.1f}s", flush=True)
    return df


def nested_pearson(rs):
    rs = np.clip(rs, -0.999, 0.999)
    return float(np.tanh(np.mean(np.arctanh(rs))))


def oof_variant(X, y, folds, m):
    """5-fold OOF ridge with per-fold imputer+scaler+ridge fit on train-fold rows only."""
    oof = np.full(len(y), np.nan)
    rs = []
    for f in range(K):
        trn = m & (folds != f)
        va = m & (folds == f)
        if va.sum() < 5 or trn.sum() < 10:
            rs.append(0.0)
            continue
        imp = SimpleImputer(strategy="median").fit(X[trn])
        sc = StandardScaler().fit(imp.transform(X[trn]))
        Xs_trn = sc.transform(imp.transform(X[trn]))
        Xs_va = sc.transform(imp.transform(X[va]))
        r = RidgeCV(alphas=ALPHAS)
        r.fit(Xs_trn, y[trn])
        pred = r.predict(Xs_va)
        oof[va] = pred
        rs.append(float(pearsonr(y[va], pred).statistic))
    return oof, rs


def main():
    # Load master table
    ft = pd.read_csv(os.path.join(CACHE, "ft_data.csv"))
    smi = ft["SMILES"].values
    folds = ft["fold"].values
    print(f"Loaded ft_data.csv: {len(smi)} molecules", flush=True)

    # Compute new features
    print("\n=== Computing pKa/stereo/chemical features ===", flush=True)
    feat_df = compute_all_features(list(smi))
    E_new = feat_df.values.astype(np.float64)
    feat_names = list(feat_df.columns)
    print(f"  {len(feat_names)} features: {feat_names}", flush=True)

    # Save features for later use
    feat_out = pd.concat([ft[["SMILES"]], feat_df], axis=1)
    feat_out.to_csv(os.path.join(CACHE, "d2d6_probe_features.csv"), index=False)
    print(f"  saved cache/d2d6_probe_features.csv", flush=True)

    # Load cpmed embedding
    cpmed = pd.read_parquet(os.path.join(CACHE, "emb_cpmed_all.parquet"))
    if "SMILES" in cpmed.columns:
        cpmed = cpmed.set_index("SMILES")
    E_cp = cpmed.reindex(smi).values.astype(np.float64)
    # Replace any NaN rows (shouldn't happen) with 0
    E_cp = np.nan_to_num(E_cp)
    print(f"\ncpmed {E_cp.shape}  new_features {E_new.shape}", flush=True)

    # Build variants
    XA = E_cp
    XB = np.hstack([E_cp, E_new])
    XC = E_new

    variants = {"A_cpmed": XA, "B_cpmed+new": XB, "C_new": XC}

    per_iso = {v: {} for v in variants}
    macro = {v: {} for v in variants}

    print("\n=== HONEST NESTED PROBE (5-fold scaffold OOF ridge) ===", flush=True)
    oof_c = pd.DataFrame(np.nan, index=ft.index, columns=ISO)
    for iso in ISO:
        y = ft[iso].values
        m = ~np.isnan(y)
        line = f"  {iso} (n={m.sum()})"
        for v, X in variants.items():
            oof, rs = oof_variant(X, y, folds, m)
            if v == "C_new":
                oof_c[iso] = oof
            nested = nested_pearson(rs)
            per_iso[v][iso] = nested
            macro[v][iso] = nested ** 2
            line += f"  | {v} {nested:.4f}"
        print(line, flush=True)

    # Blend-ready OOF for variant C (pool-test format, unlabeled rows -> 0)
    oof_c = oof_c.fillna(0)
    oof_out = oof_c.copy()
    oof_out.insert(0, "SMILES", ft["SMILES"])
    oof_out["fold"] = folds
    oof_out.to_csv(os.path.join(CACHE, "ft_oof_pka2d6.csv"), index=False)
    print("\nsaved cache/ft_oof_pka2d6.csv (variant C, blend-ready)", flush=True)

    # Macro R2
    print("\n=== NESTED MACRO R2 (mean of per-isoform nested^2) ===", flush=True)
    header = "variant".ljust(16) + "".join(i.ljust(10) for i in ISO) + "MACRO".ljust(8)
    print(header, flush=True)
    res = {}
    for v in variants:
        mac = float(np.mean([macro[v][i] for i in ISO]))
        res[v] = {i: round(per_iso[v][i], 4) for i in ISO}
        res[v]["macro_R2"] = round(mac, 4)
        row = v.ljust(16) + "".join(f"{per_iso[v][i]:.4f}".ljust(10) for i in ISO) + f"{mac:.4f}".ljust(8)
        print(row, flush=True)

    # GATE: focus on CYP2D6
    d2d6_A = per_iso["A_cpmed"]["CYP2D6"]
    d2d6_B = per_iso["B_cpmed+new"]["CYP2D6"]
    d2d6_C = per_iso["C_new"]["CYP2D6"]
    delta_2d6 = d2d6_B - d2d6_A
    print(f"\n=== GATE (CYP2D6 focus) ===", flush=True)
    print(f"  (A) cpmed only:        {d2d6_A:.4f}", flush=True)
    print(f"  (B) cpmed + new:       {d2d6_B:.4f}  (delta {delta_2d6:+.4f})", flush=True)
    print(f"  (C) new features only: {d2d6_C:.4f}", flush=True)
    print(f"  GATE: (B)-(A) >= +0.003 on CYP2D6: {'PASS' if delta_2d6 >= 0.003 else 'NOT MET'}", flush=True)

    # Per-feature importance (from variant C ridge coefficients)
    print(f"\n=== PER-FEATURE SIGNAL (variant C, CYP2D6) ===", flush=True)
    y_2d6 = ft["CYP2D6"].values
    m_2d6 = ~np.isnan(y_2d6)
    imp = SimpleImputer(strategy="median").fit(E_new[m_2d6])
    sc = StandardScaler().fit(imp.transform(E_new[m_2d6]))
    Xs = sc.transform(imp.transform(E_new))
    r = RidgeCV(alphas=[10, 100, 1000, 10000])
    r.fit(Xs[m_2d6], y_2d6[m_2d6])
    coefs = pd.Series(r.coef_, index=feat_names).sort_values(key=abs, ascending=False)
    for name, c in coefs.items():
        print(f"  {name:25s}  coef={c:+.4f}", flush=True)

    # Save results
    out = {
        "variants": {v: res[v] for v in variants},
        "gate_2d6": {
            "A_cpmed": round(d2d6_A, 4),
            "B_cpmed_new": round(d2d6_B, 4),
            "C_new": round(d2d6_C, 4),
            "delta_B_minus_A": round(delta_2d6, 4),
            "pass": bool(delta_2d6 >= 0.003),
        },
        "n_features": len(feat_names),
        "features": feat_names,
        "recipe": "per-fold median-imputer+StandardScaler+RidgeCV([100,1000,10000]) fit on train-fold labeled rows; nested = Fisher-z avg of per-fold OOF Pearson; macro R2 = mean(per-isoform nested^2)",
    }
    with open(os.path.join(CACHE, "d2d6_probe.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved cache/d2d6_probe.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
