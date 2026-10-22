# Docking Features: Post-Completion Plan

## RESULT: Avenue 3 (docking) is DEAD for both tasks. No new submission.

Docking completed: 6895 rows (6145 train + 750 test), 6679 docked (96.9%),
216 failed (3.1%). 13 features + dock_fail flag.

### Track A - Regression: FAIL
```
cp17  macro: 0.4682  (1A2 0.6305, 2C9 0.7370, 2D6 0.4923, 3A4 0.8306)
cp18  macro: 0.4666  (1A2 0.6305, 2C9 0.7339, 2D6 0.4905, 3A4 0.8306)
delta:     -0.0016   (2C9 -0.0032, 2D6 -0.0018, others flat)
|corr| vs pool: 1A2 0.298, 2C9 0.089, 2D6 0.340, 3A4 0.267
standalone macro R2: 0.0557  (1A2 0.195, 2C9 0.222, 2D6 0.181, 3A4 0.320)
```
Gate (delta > +0.004, no isoform > -0.01): **FAIL** - negative delta, redundant signal.

### Track B - TDI Classification: FAIL (inconsistent, below bar)
```
feature set          2D6     3A4     macro   vs no-dock
cpmed_umtm           0.1585  0.4281  0.2933  (best, no dock)
umtm+dock            0.1284  0.3649  0.2467  +0.0289
base_umtm+dock       0.0503  0.3635  0.2069  -0.0277
cpmed_umtm+dock      0.1531  0.3818  0.2675  -0.0259
```
- Docking helps only 1 of 3 configs (umtm+dock, 2D6-driven)
- With cpmed (strong base), docking is redundant/harmful
- Best overall (0.2933) < 0.3448 bar, << 0.5346 leaderboard best
**FAIL**

### Interpretation
The CYP2D6 heme-pocket docking features carry a SMALL, MOSTLY-REDUNDANT signal.
The "2D6 selectivity is heme-driven" hypothesis is weakly supported - there IS a
2D6 signal (standalone 2D6 R2 0.18, TDI 2D6 MCC lifts in umtm+dock), but it's too
small and too correlated with cpmed/UMTM to lift either task past its bar.

### Reusable infrastructure (kept)
`src/dock_features.py` is now production-grade: checkpointing (every N), resume
(skip done SMILES), per-child 2GB RLIMIT_AS cap (runaway ligand -> dock_fail,
not OOM-kill of the job), timeout handling. If a different pocket (3A4) or protocol
is ever worth trying, the pipeline is ready.

---
---

# (Original plan, for reference)

Once `cache/dockfeat_cyp2d6_full.csv` is complete (6895 rows: 6145 train + 750 test),
run the following in order.

## Task A: Regression Pool Member (Avenue 3)

**Goal:** Does adding 14 CYP2D6 docking features to the cp17 pool beat 0.4682?

### A1. Standalone honest nested eval
```bash
~/miniforge3/envs/cyp/bin/python src/dockfeat_eval.py
```
- Reads `cache/ft_data.csv` + `cache/dockfeat_cyp2d6_full.csv`
- Per-fold: impute + z-score (train stats), fit Ridge(alpha=1.0), predict val
- Outputs: `cache/ft_oof_dockfeat.csv` (blend-ready), `cache/dockfeat_eval.json`
- Key number: **standalone macro nested R²** (expect ~0.10-0.25 for 14 features)

### A2. Pool gate test
```bash
~/miniforge3/envs/cyp/bin/python src/new_member_pool_test.py dockfeat
```
- Loads cp17 (19 members) + `ft_oof_dockfeat.csv` as the 20th
- Runs the repo's exact honest greedy blend (per scaffold fold, Fisher-z nested)
- Reports: cp17 macro vs cp18 macro, per-isoform deltas, decorrelation
- **Gate: delta > +0.004 AND no isoform regresses > 0.01**
- Output: `cache/new_member_dockfeat_pooltest.json`

### A3. If PASS: update regression submission
- Add `ft_oof_dockfeat.csv` to the FAMS dict in `regression_final.py`
- Re-blend → new cp18 test predictions → `cache/regression_final_cp18_submission.csv`
- Verify with `src/verify_submissions.py`

### A4. If FAIL: document and move on
- Save the result, note the standalone R² and decorrelation
- Docking features may still help TDI (Task B) even if they don't lift regression

---

## Task B: TDI Classification Feature Boost

**Goal:** Does adding docking features to the TDI classifier beat the current bar?

### B1. Build the augmented feature matrix
The TDI classifier (`tdi_phase1_v2_classifier.py`) uses:
- `base` from `X_train.parquet` (Mordred 2D descriptors)
- `cpmed` from `emb_cpmed_all.parquet` (ChemProp embeddings)
- `umtm` from `umtm_oof_*.csv` (UMTM head predictions)

Create a new feature set that adds the 14 docking features:
```python
# In a new script or patch to tdi_phase1_v2_classifier.py:
dock = pd.read_csv("cache/dockfeat_cyp2d6_full.csv").set_index("SMILES")
dock_mat = dock.reindex(smi)[FEAT_COLS].values.astype(np.float32)
dock_mat = np.nan_to_num(dock_mat, nan=0.0)  # dock_fail rows → 0

feat_sets["cpmed_umtm_dock"] = np.hstack([cpmed_mat, umtm_avg, dock_mat])
feat_sets["base_umtm_dock"]  = np.hstack([base, umtm_avg, dock_mat])
```

### B2. Run the classifier with docking features
```bash
~/miniforge3/envs/cyp/bin/python src/tdi_phase1_v2_classifier.py
```
(After patching to include the new feature sets above.)

- LightGBM classifier, same params (n_est=800, lr=0.02, leaves=31)
- Gate: P(pIC50_TDI > 4.301), Shift: P(delta > 0.301)
- Score: nested MCC per isoform (CYP2D6, CYP3A4), macro = mean
- **Bar: macro product > 0.3448 (2D6 > 0.219, 3A4 > 0.470)**
- Leaderboard best: 0.5346 (TeamPozeSCAF) - that's the real competition

### B3. If PASS: update TDI submission
- Use the best feature set (with docking) for test predictions
- Generate `cache/tdi_submission_v6_candidate.csv`
- Run fraction optimization (`tdi_fraction_opt_v8.py`) to find the optimal
  selection fraction under the precision floor

### B4. If FAIL: document
- Docking features may be redundant with cpmed/umtm for the TDI task
- The heme-pocket pose signal may be more specific to continuous activity (regression)
  than to the binary TDI decision

---

## Task C: Test-Set Feature Extraction (both tasks)

The 750 test SMILES are already in the docking run. Once complete:
- `cache/dockfeat_cyp2d6_full.csv` has all 6895 rows
- Regression: merge test docking features into the cp18 blend → test preds
- TDI: merge test docking features into the classifier → test probs
- Both submissions use the same docking table, just different downstream models

---

## Execution Order & Dependencies

```
Docking completes (proc_d296f6e4872e)
    │
    ├──► A1: dockfeat_eval.py (standalone R²)
    │       │
    │       └──► A2: new_member_pool_test.py dockfeat (gate)
    │               │
    │               ├── PASS → A3: update regression submission
    │               └── FAIL → A4: document
    │
    ├──► B1: patch tdi_phase1_v2_classifier.py (add dock feature sets)
    │       │
    │       └──► B2: run TDI classifier (gate)
    │               │
    │               ├── PASS → B3: update TDI submission
    │               └── FAIL → B4: document
    │
    └──► C: test-set features ready for both submissions
```

A and B are independent - can run in parallel once docking is done.

## Key Files

| File | Role |
|------|------|
| `cache/dockfeat_cyp2d6_full.csv` | 6895 rows, 14 features + dock_fail |
| `src/dockfeat_eval.py` | Regression standalone eval (already written) |
| `src/new_member_pool_test.py` | Regression pool gate (already written) |
| `src/tdi_phase1_v2_classifier.py` | TDI classifier (needs dock feature patch) |
| `cache/ft_oof_dockfeat.csv` | Regression OOF (produced by A1) |
| `cache/new_member_dockfeat_pooltest.json` | Regression gate result (A2) |
| `cache/phase1_t1_v2_classifier.json` | TDI classifier result (B2) |

## Risks

1. **Docking features may be redundant** - cpmed (ChemProp) already captures
   3D-ish pharmacophore info. If decorrelation is high (>0.7), the pool gate
   will likely fail. The TDI classifier may show the same.
2. **Small signal** - 14 features on ~1500 labeled points per isoform is
   well-conditioned for Ridge, but the signal-to-noise may be low.
3. **CYP2D6-specific** - docking is into the 2D6 pocket only. For 1A2/2C9/3A4
   regression, the features may be noisy (wrong pocket). The TDI task only
   uses 2D6 + 3A4, so 3A4 will also be "wrong pocket" for docking features.
   This is a known limitation - the hypothesis is that 2D6 selectivity is
   heme-pocket-driven, not that docking into 2D6 predicts 3A4 activity.
