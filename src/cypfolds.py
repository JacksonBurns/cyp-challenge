"""Fold machinery shared by UMTM (copied verbatim from run_regression.py so the
chemprop training env, which has no lightgbm, can import it).
Conventions: Murcko scaffold groups, __NONE__{i//40} bucketing, greedy
size-balanced k-way split. Regression track uses seed 0 (cache/folds.npy ==
ft_data.csv fold column), TDI track uses seed 7. UMTM master carries BOTH.

GOTCHA (2026-10): make_folds depends on np.argsort default-quicksort TIE-BREAK
ORDER, which differs between numpy 1.26.4 (cyp env, legacy) and 2.5.3 (chemeleon
env) - identical group vectors give 21.7% fold agreement across envs. ALWAYS run
fold GENERATION in the cyp env (numpy 1.26.4) so fold_reg is byte-identical to
ft_data.csv and fold_tdi identical to run_tdi.py. Training envs just consume the
stored columns, never recompute.
"""
import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
K = 5


def mol_of(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return m
    except Exception:
        return None


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
    order = order[np.argsort(-sizes[order])]
    fold = np.empty(len(group_keys), dtype=int)
    load = np.zeros(k)
    for g in order:
        f = int(np.argmin(load))
        fold[inv == g] = f
        load[f] += sizes[g]
    return fold
