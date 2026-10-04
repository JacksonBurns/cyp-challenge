import inspect
import numpy as np, pandas as pd
from chemprop.nn.metrics import ChempropMetric
src = inspect.getsource(ChempropMetric)
print(src[src.index('def update'):src.index('def compute')])
print('#### compute')
print(src[src.index('def compute'):])
from chemprop.models import MPNN
print('#### MPNN configure_optimizers defined:', 'configure_optimizers' in MPNN.__dict__)
print('#### MLP sig:', inspect.signature(__import__('chemprop.nn', fromlist=['MLP']).MLP.__init__))

# dataset normalization: internal Y vs datapoint y
from chemprop import data, featurizers
pts = [data.MoleculeDatapoint.from_smi('CCO', [1.0, np.nan]),
       data.MoleculeDatapoint.from_smi('CCCO', [3.0, 10.0]),
       data.MoleculeDatapoint.from_smi('CCCCO', [np.nan, np.nan])]
d = data.MoleculeDataset(pts, featurizers.SimpleMoleculeMolGraphFeaturizer())
print('Y before', d.Y())
sc = d.normalize_targets()
print('Y after', d.Y())
print('pt y after', [p.y for p in pts])
from chemprop.nn import UnscaleTransform
t = UnscaleTransform.from_standard_scaler(sc)
import torch
print('unscale applied to normalized:', t(torch.tensor(d.Y()))[:, :2])
