import numpy as np
from chemprop import data, featurizers
rows = [[1.0, np.nan], [3.0, 1.0], [5.0, 0.0], [np.nan, 1.0]]
pts = [data.MoleculeDatapoint.from_smi("CCO", r) for r in rows]
ds = data.MoleculeDataset(pts, featurizers.SimpleMoleculeMolGraphFeaturizer())
sc = ds.normalize_targets()
print(type(sc).__name__, [a for a in dir(sc) if not a.startswith('_')])
print('mean_:', sc.mean_, 'std_:', sc.std_)
import torch
print('mean tensor?', type(sc.mean_))
