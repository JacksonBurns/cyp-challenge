# REPRODUCTION - final-push artifacts (verified Oct 22 2026)

Two shipped candidates. Both rebuild EXACTLY (md5-verified) from the cleaned
`src/` + committed `cache/` feature caches. Envs: `cyp`
(`~/miniforge3/envs/cyp/bin/python`, rdkit + lightgbm + pandas), `tabpfn`,
`tabicl-chemeleon` (embedding members only; their OOF/test preds are cached,
so plain rebuilds need only `cyp`).

## Regression: cp17 (nested macro R2 0.4682)

Pipeline (all from repo root, cyp env):
1. Pool weights (greedy nested over the 19 families in FAMS):
   `python src/blendN_all.py` -> `cache/blendN_cp17.json` (weights + OOF
   pearson per isoform). Member OOF files `cache/ft_oof_*.csv` are committed
   feature caches (each family built by its own `src/ft_*` / trainer script -
   see docstrings).
2. Submission builder (z-space mixture -> reveal placement; placement constants
   F_SPREAD / OOF_TO_BLIND / BLIND_MOMENTS / STRAE_MOMENTS at top of file):
   ```
   python src/regression_final.py --blend blendN_cp17.json \
       --out regression_final_cp17_submission.csv
   ```
   VERIFIED Oct 22: rebuild is byte-identical
   (md5 `11096b10097b30cc164a70b6c1e2b0aa`).
3. Validation:
   `python src/verify_submissions.py cache/regression_final_cp17_submission.csv`
   (official activity validator + row-for-row SMILES/name match vs blinded test).

## TDI: tdi_v5 (nested capped macro MCC 0.3404; leaderboard 0.3422)

Pipeline (cyp env, repo root):
1. Family-capped blend audit: `python src/tdi_blend_family_block4.py` ->
   `cache/tdi_family_block_audit4.json` (pool v6b, 0.5 family cap, fold-nested
   weights). Imports `tdi_blend_family_block2` (kept).
2. Fraction optimization under the board pi-prior:
   `python src/tdi_fraction_opt_v7.py` -> `cache/tdi_fraction_optima_v7.json`
   (2D6 plateau 0.15-0.19, 3A4 0.23-0.31; shipped 0.15 / 0.24).
3. Submission builder (z-avg of member test probs from audit4 v6b merged
   weights, threshold = target fraction):
   `python src/make_tdi_submission_v5.py` -> `cache/tdi_submission_v5_candidate.csv`
   VERIFIED Oct 22: rebuild = 0 label flips (112/180 positives, identical
   predictions; official TDI validator PASS).
4. Validation: built-in (`validation/tdi_validation.py`), prints `VALID: True`.

## L1 candidate (Oct 22, not yet submitted): cp18_hinge

1. `python src/hinge_blend.py --l4` -> `cache/hinge_blend_l1.json` (per-isoform
   gate report) + `cache/hinge_w_cp18.json` (fitted weights; 1A2/3A4 pass).
2. `python src/hinge_blend.py --emit` ->
   `cache/regression_cp18_hinge_submission.csv` (1A2/3A4 hinge-fit columns,
   moment-restore to cp17 placed mean/sd; 2C9/2D6 = cp17 columns).
   md5 `21e48ea3ac61086fda929f4c56d51c0a` (Oct 22 build).
3. `python src/verify_submissions.py cache/regression_cp18_hinge_submission.csv`
   -> official validator True.

## Honest-eval harness (the metric every gate uses)

- `src/new_member_pool_test.py <tag>`: adds `cache/ft_oof_<tag>*.csv` as a
  20th pool member, runs the exact greedy nested recipe (5 scaffold folds,
  seed 0, Fisher-z) + decorrelation. Gate used all cycle: delta > +0.004 and
  no isoform < -0.01.
- `src/eval_ft.py`: standalone honest nested eval of one ft family.
- Fold definition: `src/run_regression.py::make_folds/scaffold_groups`
  (Bemis-Murcko scaffold groups, seed in file; cyp env only - numpy-version
  tie-break GOTCHA, see src/cypfolds.py).

## What is NOT rebuildable from this repo (by design)

GPU retraining of pool members (chemprop FT/UMTM/D-MPNN runs, CheMeLeon
extract, TabPFN/TabICL inference) - long jobs whose OOF + test preds are the
committed caches above. Rebuild-from-raw-features requires the GPU box + the
trainer scripts (kept in src/, docstrings carry env + invocation).
