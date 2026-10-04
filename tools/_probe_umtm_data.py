import sys, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
import run_regression as R

ft = pd.read_csv('cache/ft_data.csv')
groups = R.scaffold_groups(ft['SMILES'].tolist())
for sd in (0, 7):
    f = R.make_folds(groups, seed=sd)
    print('rows', len(ft), 'seed', sd, 'match frac vs ft_data fold:', (f == ft['fold'].values).mean())
folds7 = R.make_folds(groups, seed=7)
try:
    fn = np.load('cache/folds.npy')
    print('folds.npy vs seed7:', (fn == folds7).mean(), 'vs ft_data:', (fn == ft['fold'].values).mean())
except Exception as e:
    print('folds.npy:', e)
tdi = pd.read_csv('data/cyp-challenge-TRAIN_TDI.csv')
inh = pd.read_csv('data/cyp-challenge-TRAIN_inhibition.csv')
sc = pd.read_csv('data/cyp-challenge-single-concentration-TRAIN.csv')
emax = pd.read_csv('data/cyp-challenge-TRAIN_Emax.csv')
test = pd.read_csv('data/cyp-challenge-TEST-BLINDED.csv')
tu, iu, scu = tdi.drop_duplicates('SMILES'), inh.drop_duplicates('SMILES'), sc.drop_duplicates('SMILES')
print('inh uniqs:', len(iu), 'tdi uniqs:', len(tu), 'sc uniqs:', len(scu), 'emax uniqs:', len(emax.drop_duplicates('SMILES')), 'test:', len(test), test.SMILES.nunique())
print('enzymes:', sorted(sc.enzyme.unique()))
print('inh subset of tdi:', iu.SMILES.isin(tdi.SMILES).all())
print('tdi-only:', (~tu.SMILES.isin(inh.SMILES)).sum())
print('sc not in tdi:', (~sc.SMILES.isin(tdi.SMILES)).sum(), 'sc not in inh:', (~sc.SMILES.isin(inh.SMILES)).sum())
print('ft labeled:', {i: int(ft[i].notna().sum()) for i in R.ISOFORMS}, 'fold nulls:', int(ft.fold.isna().sum()))
print('test cols sample:', list(test.columns)[:8])
print('is_TDI value sets:', {c: tdi[c].dropna().unique()[:5] for c in ['CYP2D6_is_TDI', 'CYP3A4_is_TDI']})
print('tdi pos counts:', {c: (tdi[c] == True).sum() for c in ['CYP2D6_is_TDI', 'CYP3A4_is_TDI']}, 'nan:', {c: int(tdi[c].isna().sum()) for c in ['CYP2D6_is_TDI', 'CYP3A4_is_TDI']})
