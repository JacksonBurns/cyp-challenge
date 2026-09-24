import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

ft = pd.read_csv('cache/ft_data.csv')
d = pd.read_parquet('cache/emb_cpmed_all.parquet').set_index('SMILES')
cols = [c for c in d.columns if not str(c).startswith('CYP')]
E = d.reindex(ft['SMILES']).values.astype(float)
print('E', E.shape, 'nan', int(np.isnan(E).sum()))
folds = ft['fold'].values
for iso in ['CYP1A2', 'CYP3A4']:
    y = ft[iso].values
    m = ~np.isnan(y)
    sc = StandardScaler().fit(E[m])
    Xt = sc.transform(E)
    for al in [1e3, 1e4, 1e5, 1e6]:
        oof = np.full(len(y), np.nan)
        for f in range(5):
            trn = np.where(m & (folds != f))[0]
            va = np.where(m & (folds == f))[0]
            r = RidgeCV(alphas=[al])
            r.fit(Xt[trn], y[trn])
            oof[va] = r.predict(Xt[va])
        print(iso, al, round(pearsonr(y[m], oof[m]).statistic, 3))
