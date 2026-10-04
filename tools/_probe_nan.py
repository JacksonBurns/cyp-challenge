import numpy as np
from chemprop import data, featurizers
pts = [data.MoleculeDatapoint.from_smi('CCO', [1.0, np.nan]),
       data.MoleculeDatapoint.from_smi('CCCO', [3.0, 10.0]),
       data.MoleculeDatapoint.from_smi('CCCCO', [np.nan, np.nan])]
d = data.MoleculeDataset(pts, featurizers.SimpleMoleculeMolGraphFeaturizer())
y0 = np.array([p.y for p in pts])
sc = d.normalize_targets()
y1 = np.array([p.y for p in pts])
print('mean', sc.mean_, 'scale', sc.scale_)
print('before', y0.tolist())
print('after', y1.tolist())
