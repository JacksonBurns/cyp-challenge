#!/usr/bin/env python3
"""Smoke test: dock dextromethorphan (canonical CYP2D6 substrate) + 3 others.
Validates the box (ligand should land near heme Fe) and the full pipeline.
"""
import sys, os, time
sys.path.insert(0, "/home/jackson/cyp-challenge/src")
from dock_features import _dock_one, _fingerprint, FE_POS, BOX_CENTER, BOX_SIZE

# canonical CYP2D6 substrates/inhibitors
test_smis = {
    "dextromethorphan": "CN1CC[C@@]2(c3ccccc3O2)C1",
    "tamoxifen": "COc1ccc(cc1)/C=C(/C)\\c2ccc(OC)c(O)c2",
    "metoprolol": "CNCCc1ccc(CC(O)CNC(C)C)cc1",
    "haloperidol": "CC1(CN(C)C)C(c2ccccc2Cl)CC(c3ccc(O)cc3F)N1C(=O)C",
}

print(f"Box center (heme Fe): {FE_POS}")
print(f"Box size: {BOX_SIZE}\n")
for name, smi in test_smis.items():
    t0 = time.time()
    res = _dock_one(smi, exh=8)
    el = time.time() - t0
    if res is None:
        print(f"{name:20s} [FAIL] {el:.1f}s")
        continue
    aff, pose, le, lt = res
    f = _fingerprint(aff, pose, le, lt)
    print(f"{name:20s} [ok] {el:.1f}s  score={f['vina_score']:.2f}  "
          f"heme_contact={f['n_heme_contact']}  hbond={f['n_hbond']}  "
          f"hyd={f['n_hydrophobic']}  cation_pi={f['n_cation_pi']}  salt={f['n_salt']}  "
          f"bN_Fe={f['basicN_Fe_dist']:.1f}  bN_acid={f['basicN_acid_dist']:.1f}  "
          f"inbox={f['pose_inbox_frac']:.2f}  n={f['n_heavy']}")
