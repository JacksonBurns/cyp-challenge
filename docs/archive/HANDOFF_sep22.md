# HANDOFF - cyp-challenge (2026-09-22, ~17:30 local)

**SUPERSEDED PARTIALLY - SEE NOTES.md SECTION 9 FIRST (Sep 23 interim
reveal):** interim scored: regression 0.6766 MA-ST-RAE (rank 93/229),
TDI 0.3149 MA-MCC (rank 60/121). Diagnosis: placement works, ranking is
the gap; external data + deep embeddings now mandatory for top-40.
Updated agent handoff prompt lives at docs/NEXT_AGENT_PROMPT.md.

Interim-submission deadline: **Sep 24 2026**. Final closes Nov 3 2026.
12h cooldown between submissions; DO NOT SUBMIT without Jackson's explicit go-ahead
(needs his HF account on the Submit tab of https://huggingface.co/spaces/openadmet/cyp-challenge).

## What is done (verified)

### Regression track - MODEL COMPLETE, SUBMISSION FILE VALIDATED
- Env: `~/miniforge3/envs/cyp/bin/python` (rdkit, lightgbm 4.7, pandas, scipy). Repo `~/cyp-challenge`.
- Features cached: `cache/X_train.parquet`, `cache/X_test.parquet` (RDKit desc + Morgan2/3 counts + MACCS).
  **Gotcha: X_train.parquet has duplicate SMILES rows (concat of inhibition+TDI files); always
  `drop_duplicates("SMILES")` before indexing by SMILES.**
- `src/run_regression.py` - core library: `build_all()`, `nn_block()` (Tanimoto NN label features,
  **self-match exclusion via q_idx is mandatory or the model leaks**), `lgbm()`, `soft_rae()`,
  `make_folds()`, `scaffold_groups()`, `cv_config()`.
- `src/sweep.py` - 4 LGBM configs, scaffold-grouped 5-fold CV. Blend of all 4 is best:
  **scaffold-OOF MA-ST-RAE = 0.743** (1A2 0.821 / 2C9 0.692 / 2D6 0.947 / 3A4 0.512).
  Stacked multitask with TDI-arm aux was NOT better (0.770). Results: `cache/cv_sweep.json`
  (params re-injected manually into that json - sweep.py's own dump lacks "params" key).
- `src/final_submit.py` - retrains the 4-config x 3-seed blend on all labeled data per isoform,
  places predictions on the blind-half moments (BLIND_MOMENTS x OOF_TO_BLIND rho inflation,
  CYP2D6 uses ST-RAE-specific moments mean 3.57 sd 0.90), floors at 1.0.
  Output: **`cache/raw_regression_submission.csv` - PASSES official validator** (750 rows,
  correct cols, no NaN, std>=0.01). Calibration report: `cache/calibration_report.json`.
- OOF Pearson of blend: `cache/oof_pearson.json` (0.527/0.604/0.399/0.771).
- Expected blind ST-RAE ~0.5-0.62 by analogy to jeremy's public writeup (his comparable
  pipeline + same calibration scored 0.52; leaderboard top 0.38, raw-GBM field ~0.85-0.9).

### Classification track (TDI) - model good, SUBMISSION FILE NOT YET WRITTEN
- `src/run_tdi.py` - LightGBM classifier per scored isoform (2D6/3A4), scaffold-grouped CV,
  NN-positives features (self-exclusion fixed - **earlier collapse to MCC 0 was leakage**).
  Latest run (leak-free): 3A4 OOF MCC 0.376 at thr 0.15 (old grid floor; grid since extended
  to 0.05-0.85 but NOT rerun), 2D6 OOF MCC 0.153 at thr 0.27. Pearson 3A4 0.39 / 2D6 0.12.
- Test probabilities saved: `cache/tdi_test_probs.csv` (SMILES, CYP2D6_proba, CYP3A4_proba).
- `src/tdi_shift.py` - alternative approach (regress TDI-arm pIC50, apply the label rule
  shift>0.301 / arm>4.301). WRITTEN BUT NEVER COMPLETED A RUN (interrupted); decide whether
  to finish it or just go with classifier.
- Leaderboard context: top MA-MCC 0.494; ~0.44 cluster; baseline LGBM 0.288. Our OOF ~0.38-0.45
  blind-blend plausible.

## TODO (in order)

1. **Write `src/make_tdi_submission.py`**: read `cache/tdi_test_probs.csv` + thresholds
   (rerun `src/run_tdi.py` once with the extended grid to get final best_thr per isoform,
   or tune on existing OOF), emit 750-row CSV with columns
   `SMILES, Molecule_Name, CYP2D6_is_TDI, CYP3A4_is_TDI` (booleans True/False), row order
   matching `data/cyp-challenge-TEST-BLINDED.csv`, then validate with
   `validation/tdi_validation.py::validate_tdi_submission` from repo root (`sys.path.insert(0,".")`).
   2D6 signal is weak (OOF MCC 0.15): consider blending classifier proba with the shift-rule
   score, or for 2D6 use a moderate threshold near tuned value - do NOT predict all-negative
   (MCC punishes it).
2. Re-verify both submission files with both official validators, row counts, SMILES match
   blinded file exactly.
3. Update NOTES.md with final numbers; git add + commit everything (single commit, no push).
4. Report to Jackson: files ready, expected standing, ask permission + his HF login for the
   Submit tab (he must log in; upload both CSVs in the two track sections, tick open-code if asked).
5. Optional later (before Nov 3): external data (ChEMBL CYP, PubChem AID1851), Chemprop
   D-MPNN ensemble, Uni-Mol/frozen-encoder features, seed-bagged stacking. Interim reveal Sep 25.

## Key gotchas (all hit the hard way)

- Dedupe train features by SMILES before target reindex (`build_all` does it; run_tdi now too).
- `nn_block` self-exclusion (q_idx) mandatory in CV; test queries need none.
- is_TDI label columns are object dtype with NaN; convert `[1 if bool(v) else 0 for v in y[mask]]`.
- LightGBM fit must get only labeled rows for the target (mask NaN rows yourself, not just pass NaN).
- `validation/` and `evaluation/` are importable from repo root; final_submit's inline import of
  validation fails when run from /tmp scripts - run validation from repo root separately.
- The execute_code `terminal()` wrapper returns keys without "output" on failure - guard with .get.
- A background run of final_submit.py succeeded (see below) - its output file is the real one:
  `cache/raw_regression_submission.csv`.

## File map

- data/ raw CSVs; cache/ features+outputs; evaluation/ + validation/ official tutorial code;
  leaderboard/ scraped CSVs (2026-09-22); src/ all our code; NOTES.md full competition intel
  (metric formulas, BLIND_MOMENTS provenance, submission rules, timeline).
- Logs: /tmp/cyp_stage1.log /tmp/cyp_sweep.log /tmp/cyp_final.log
