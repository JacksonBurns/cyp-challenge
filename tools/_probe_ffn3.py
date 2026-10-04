import inspect
from chemprop.nn import predictors as P
ffn_mod = [c for c in vars(P).values() if inspect.isclass(c) and c.__module__ == P.__name__]
print("classes:", [c.__name__ for c in ffn_mod])
for name in ["RegressionFFN", "FFModule", "FFNModule"]:
    if hasattr(P, name):
        c = getattr(P, name)
        print("=" * 10, name, "MRO:", [k.__name__ for k in c.__mro__[:4]])
        for m in ["__init__", "forward", "train_step", "encode", "set_scaler"]:
            if m in vars(c) or any(m in vars(k) for k in c.__mro__):
                try:
                    owner = [k for k in c.__mro__ if m in vars(k)][0]
                    print(f"--- {owner.__name__}.{m} ---")
                    print(inspect.getsource(getattr(owner, m))[:1100])
                except Exception as e:
                    print(m, "ERR", e)
