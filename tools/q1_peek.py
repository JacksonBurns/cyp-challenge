import pandas as pd
D = '/home/jackson/cyp-challenge/data/'
tdi = pd.read_csv(D + 'cyp-challenge-TRAIN_TDI.csv')
emx = pd.read_csv(D + 'cyp-challenge-TRAIN_Emax.csv')
print('TDI cols w/ is_TDI:', [c for c in tdi.columns if c.endswith('is_TDI')])
print('EMX cols w/ is_TDI:', [c for c in emx.columns if c.endswith('is_TDI')])
for df, nm in [(tdi, 'TDI'), (emx, 'EMX')]:
    for c in [c for c in df.columns if c.endswith('is_TDI')]:
        vc = df[c].value_counts(dropna=False)
        print(nm, c, 'n_rows=', len(df), 'counts=', dict(list(vc.items())[:5]))
X = pd.read_parquet('/home/jackson/cyp-challenge/cache/X_train.parquet')
print('X_train', X.shape, 'has SMILES:', 'SMILES' in X.columns,
      'nfeat=', X.shape[1] - 1)
