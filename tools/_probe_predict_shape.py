"""Diagnose the smoke shape mismatch: fold_inputs row counts + predict row coverage."""
import os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/cyp-challenge'))
sys.path.insert(0, 'src')
from umtm_module import fold_inputs, build_points, BATCH
from chemprop import data, featurizers

master = pd.read_csv('cache/umtm_master.csv')
ch = master[master.kind == 'challenge'].reset_index(drop=True)
print('challenge rows', len(ch), 'fold counts', np.bincount(ch.fold_reg.astype(int)))
fit_df, ev_df, va_idx = fold_inputs(master, 'plan', 'fold_reg', 0, 0)
print('va_idx n =', len(va_idx), '| ev_df rows =', len(ev_df))
print('fit_df rows', len(fit_df), 'by kind:', fit_df['kind'].value_counts().to_dict() if 'kind' in fit_df else '?')

# predict-side featurization: how many fold-0 SMILES survive?
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
smis = ch.iloc[va_idx]['SMILES'].tolist()
bad = 0
for s in smis:
    try:
        x = feat.featurize(s)[0]
        import torch
        if not torch.isfinite(x.x).all():
            bad += 1
    except Exception:
        bad += 1
print('fold0 smiles:', len(smis), 'featurize-fail/nonfinite:', bad)

# dataset-level filter check
pts = [data.MoleculeDatapoint.from_smi(s, [float('nan')] * 22) for s in smis]
dset = data.MoleculeDataset(pts, feat)
print('raw datapoints:', len(dset))
loader = data.build_dataloader(dset, batch_size=BATCH, num_workers=0, shuffle=False)
n = sum(b[0].shape[0] if hasattr(b[0], 'shape') else len(b[0]) for b in loader)
print('loader total rows:', n)
