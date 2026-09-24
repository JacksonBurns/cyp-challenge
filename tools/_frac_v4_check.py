import sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'src')
from tdi_fraction_opt import rank_conditional_curve, emcc, rescale_to_pi

Xtr = pd.read_parquet('cache/X_train.parquet').drop_duplicates('SMILES').reset_index(drop=True)
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv')
emx = pd.read_csv('data/cyp-challenge-TRAIN_Emax.csv')
lab = pd.concat([tdi[['SMILES', 'CYP2D6_is_TDI', 'CYP3A4_is_TDI']],
                 emx[['SMILES', 'CYP2D6_is_TDI', 'CYP3A4_is_TDI']]]).drop_duplicates('SMILES').set_index('SMILES')
Y = lab.reindex(Xtr['SMILES'])
pi_post = {"CYP3A4": {0.25: 0.08, 0.30: 0.15, 0.35: 0.22, 0.40: 0.28, 0.45: 0.17, 0.50: 0.10},
           "CYP2D6": {0.216: 0.15, 0.26: 0.22, 0.30: 0.26, 0.34: 0.22, 0.38: 0.10, 0.42: 0.05}}
for iso, cands in [("CYP2D6", [0.08, 0.20, 0.25, 0.30, 0.33, 0.35]),
                   ("CYP3A4", [0.32, 0.36, 0.40, 0.455, 0.50])]:
    p = np.load(f'cache/tdi_nested_pooled_oof_{iso}.npy')
    y = Y[f'{iso}_is_TDI'].values
    m = ~pd.isna(y)
    yl = np.array([1 if bool(v) else 0 for v in y[m]], dtype=int)
    rates, _ = rank_conditional_curve(p, yl)
    for f in cands:
        e = sum(pi_post[iso][pi] * emcc(rescale_to_pi(rates, pi), f) for pi in pi_post[iso])
        worst = min(emcc(rescale_to_pi(rates, pi), f) for pi in pi_post[iso])
        print(iso, f, round(e, 4), 'worst', round(worst, 4))
