import inspect
from chemprop.data.datasets import MoleculeDataset
from chemprop.nn import UnscaleTransform
print("=== normalize_targets ===")
print(inspect.getsource(MoleculeDataset.normalize_targets))
print("=== from_standard_scaler ===")
print(inspect.getsource(UnscaleTransform.from_standard_scaler))
print("=== UnscaleTransform init/fwd ===")
print(inspect.getsource(UnscaleTransform.__init__))
print(inspect.getsource(UnscaleTransform.forward))
