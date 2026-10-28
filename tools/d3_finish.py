"""Finish d3 build for remaining molecules with per-molecule timeout.

SIGALRM guard inside workers (300s/molecule): pathological MMFF cases
return NaN features instead of hanging. Resumable; caches every 250.
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jackson/cyp-challenge/src")
import hinge_blend as H  # noqa: E402

OUTD = "/home/jackson/cyp-challenge/cache/adme4"
cpath = os.path.join(OUTD, "d3_cache.pkl")
TIMEOUT = 300


def get_todo(c):
    test = pd.read_csv("/home/jackson/cyp-challenge/data/cyp-challenge-TEST-BLINDED.csv")
    allsmi = list(H.Xtr["SMILES"]) + list(test["SMILES"])
    return [s for s in allsmi if s not in c], allsmi


def safe(smi):
    import signal

    def _to(signum, frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, _to)
    signal.alarm(TIMEOUT)
    try:
        from adme4_build import compute_one
        s, f = compute_one(smi)
        signal.alarm(0)
        return smi, f
    except (TimeoutError, Exception):
        signal.alarm(0)
        return smi, np.full(12, np.nan)


def main():
    with open(cpath, "rb") as fh:
        c = pickle.load(fh)
    todo, allsmi = get_todo(c)
    print("remaining", len(todo), flush=True)
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(12) as pool:
        for i, (smi, f) in enumerate(pool.imap(safe, todo, chunksize=4)):
            c[smi] = f
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(todo)} t={int(time.time()-t0)}s",
                      flush=True)
                with open(cpath, "wb") as fh:
                    pickle.dump(c, fh)
    with open(cpath, "wb") as fh:
        pickle.dump(c, fh)
    todo, allsmi = get_todo(c)
    print("DONE cached", len(c), "todo", len(todo), flush=True)


if __name__ == "__main__":
    main()
