import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
import run_tdi_ext as T
from lightgbm import LGBMClassifier

Xtr = pd.read_parquet('cache/X_train.parquet').drop_duplicates('SMILES').reset_index(drop=True)
Xte = pd.read_parquet('cache/X_test.parquet')
excl = set(c for c in (T.canon(s) for s in Xtr['SMILES']) if c) | set(c for c in (T.canon(s) for s in Xte['SMILES']) if c)
ext, Xe = T.load_external(excl)
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv')
emx = pd.read_csv('data/cyp-challenge-TRAIN_Emax.csv')
lab = pd.concat([tdi[['SMILES', 'CYP3A4_is_TDI']], emx[['SMILES', 'CYP3A4_is_TDI']]]).drop_duplicates('SMILES').set_index('SMILES').reindex(Xtr['SMILES'])
y = lab['CYP3A4_is_TDI'].values
m = ~pd.isna(y)
yl = np.zeros(len(y))
yl[m] = y[m].astype(bool).astype(float)
Btr = (Xtr[T.FP_COLS].values > 0).astype(np.float32)
Bex = (Xe[T.FP_COLS].values > 0).astype(np.float32)


def sim(A, Bb):
    d = A @ Bb.T
    ca = (A > 0).sum(1)[:, None]
    cb = (Bb > 0).sum(1)[None, :]
    return d / np.maximum(ca + cb - d, 1e-9)


Sex = sim(Bex, Btr)
elab = ~np.isnan(ext['CYP3A4_lab'].values)
pos = np.where(m & (yl > 0))[0]
Sp = Sex[elab][:, pos]
print('Sex ext-vs-train positives shape:', Sp.shape, 'mean maxsim:', Sp.max(1).mean())
t0 = time.time()
Xf = np.hstack([Xe.drop(columns=['SMILES']).values.astype(np.float32)[elab],
                np.column_stack([Sp.max(1), -np.partition(-Sp, 9, axis=1)[:, :10].mean(1), (Sp > 0.7).sum(1)]).astype(np.float32)])
mm = LGBMClassifier(n_estimators=50, n_jobs=4, verbosity=-1)
mm.fit(Xf, ext['CYP3A4_lab'].values[elab].astype(int))
print('fit ok', time.time() - t0, Xf.shape)
