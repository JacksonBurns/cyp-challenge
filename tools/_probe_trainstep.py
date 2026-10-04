import inspect
import numpy as np
import torch
from chemprop import data, featurizers, models, nn
from chemprop.nn.metrics import MSE

import chemprop.nn.predictors as P
mod = P.__name__
for cname in [c.__name__ for c in vars(P).values() if inspect.isclass(c) and c.__module__ == mod]:
    print("CLASS", cname)
ffn = nn.RegressionFFN(n_tasks=2, input_dim=8)
for m in ["forward", "train_step", "encode"]:
    owner = [k for k in type(ffn).__mro__ if m in vars(k)]
    if owner:
        print(f"--- {owner[0].__name__}.{m} ---")
        print(inspect.getsource(getattr(owner[0], m)))
# normalize_targets: does it mutate dataset targets?
rows = [[2.0, np.nan], [4.0, 1.0], [6.0, 0.0], [8.0, 1.0]]
pts = [data.MoleculeDatapoint.from_smi("CCO", list(r)) for r in rows]
ds = data.MoleculeDataset(pts, featurizers.SimpleMoleculeMolGraphFeaturizer())
sc = ds.normalize_targets()
print("post-normalize ds[0] targets:", [ds[i].targets.tolist() for i in range(4)])
