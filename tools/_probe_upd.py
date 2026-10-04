import inspect
from chemprop.nn.metrics import ChempropMetric
src = inspect.getsource(ChempropMetric.update)
i = src.index('L = self._calc_unreduced_loss')
print(src[i:])
