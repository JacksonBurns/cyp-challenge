# OpenADMET CYP Inhibition Blind Challenge - Competition Notes

Compiled 2026-09-22 by Hermes agent for Jackson Burns.
Sources: challenge HF space (`openadmet/cyp-challenge`, config.py from space repo),
announcement blog post (openadmet.ghost.io, 29 Jul 2026), tutorial repo
(github.com/OpenADMET/CYP-Challenge-Tutorial), HF dataset
(`openadmet/cyp-challenge-train-test`).

## 1. Task summary

Predict Cytochrome P450 inhibition for a blinded 750-compound test set.
Two independent tracks, submitted as two separate CSV/parquet files:

**Track A - Direct Inhibition (regression).** Predict direct-inhibition pIC50
(-log10 IC50, higher = more inhibitory) for 4 isoforms: CYP1A2, CYP2C9, CYP2D6,
CYP3A4. Submission: exactly 750 rows, columns
`SMILES, Molecule_Name, CYP1A2_pIC50_direct_inhibition, CYP2C9_..., CYP2D6_..., CYP3A4_...`.

**Track B - TDI (classification).** Predict boolean time-dependent inhibition
for CYP3A4 and CYP2D6 only. Columns `CYP2D6_is_TDI, CYP3A4_is_TDI` (bool).

Rules: 1 submission per 12 hours; latest valid submission counts; predictions
must have std >= 0.01 per column (anti-gaming); no NaN/inf; external/pretrained
data allowed but proprietary data must be disclosed; one account per team.

Timeline: opened Aug 17 2026; interim-submission deadline **Sep 24 2026** (2 days
away!); interim leaderboard Sep 25 (one-time full-test reveal); closes Nov 3 2026.

## 2. Metrics (exact code in tutorial repo `evaluation/`)

**Regression primary: MA-ST-RAE** = macro-average across 4 endpoints of
Soft-Threshold Relative Absolute Error:

```
ST-RAE = sum(soft_err(y, yhat)) / sum(soft_err(y, mean(y)))
soft_err = max(0, yhat - conf_high) + max(0, conf_low - yhat)   # 0 inside CI band
```

- Denominator is the naive constant predictor `mean(y_test)` put through the
  same soft-thresholding. Lower is better; 1.0 = predict-the-mean.
- CI bands come from the *test-set* Bayesian DRC fits and are wide at low
  activity (conf_low floor ~1.03), so getting inactive compounds "close enough"
  is nearly free; the metric is really driven by correctly placing active
  compounds whose bands are tight.
- Strategy implication: predicting the posterior median with correct spread is
  what matters; systematic under/over-shrinkage on actives costs.
- Secondary: MAE, R2, Spearman, Kendall (bootstrap 1000x).

**Classification primary: MCC** on CYP3A4_is_TDI and CYP2D6_is_TDI (macro across
the two). Label rule: positive iff TDI-arm pIC50 exceeds direct-arm pIC50 by
>0.301 (2-fold shift); direct pIC50 < 4: inferred positive if TDI-arm > 4.301,
assigned negative if both arms < 4. Class imbalance likely; MCC punishes
all-one/all-zero predictions, threshold tuning matters.

## 3. Data files (in `data/`)

### cyp-challenge-TRAIN_inhibition.csv - PRIMARY TRAIN TARGETS (4,905 x 18)
Sparse direct-inhibition pIC50 per isoform + conf_high/conf_low + std.
Label counts: 1A2: 1,412 | 2C9: 1,285 | 2D6: 1,493 | 3A4: 2,335 (rows are mostly
single-isoform DRCs from the primary-screen hit expansion).
Value ranges ~1.9-7.9. CI width grows sharply below pIC50 ~4 (conf_low floored
~1.03 = assay LLOQ region). std column = posterior std of the fit.

### cyp-challenge-TEST-BLINDED.csv (750 x 2)
`Molecule_Name, SMILES` only. Dense: every compound has all 4 isoforms + TDI
labels (withheld). Zero SMILES/name overlap with any train file.

### cyp-challenge-TRAIN_TDI.csv (6,145 x 36)
TDI-arm pIC50 (both arms: `*_pIC50_TDI_condition` and
`*_pIC50_direct_inhibition`) for 6,145 compounds; includes ALL 4,905 of the
inhibition-train compounds plus 1,240 extra compounds with labels; boolean
`CYP2D6_is_TDI`, `CYP3A4_is_TDI` for classification track. NOTE: is_TDI here is
precomputed for training; use the same labeling rule consistently.

### cyp-challenge-TRAIN_Emax.csv (6,145 x 30)
Emax-vs-positive-control (log2-like fold scale, both arms) + `is_TDI` bool for
all 4 isoforms. Extra weak-activity signal (any inhibitor with high Emax), and
the Emax column is dense-ish (more non-NaN) - useful auxiliary target/features.

### cyp-challenge-single-concentration-TRAIN.csv (17,504 rows; 4,376 cpds x 4 enzymes)
Primary screen (TDI/+NADPH condition!) at 50 uM: `log2fc_estimate` (negative =
inhibition), std_error, p_value, FDR, cohens_d. Auxiliary multi-task signal,
but only for the screened subset; condition = TDI arm, so use carefully.

### Key structural facts (verified in data)
- Train label density per isoform ~1.3-2.3k; test is dense 750x4.
- Test set = top-25 hits x top-10 ECFP4-chemisimilar series per CYP (1A2/2C9/3A4
  only). => TEST IS CHEMISTRACTIVE: enriched in potent analogs of the very hits
  used to build the train set, but parents are NOT in train (no exact overlap).
  Nearest-neighbor features and scaffold-level SAR should transfer well; careful
  random CV will be over-optimistic - emulate series structure with grouped CV.
- 2D6 was assayed later (Echo-MS); 2D6 data mostly from TDI-file compounds.

## 4. Modeling plan (v1)

1. Featurize: RDKit descriptors (200+) + Morgan counts FP (2048, r=2/3) +
   chemisimilarity-to-train-centroids features.
2. Per-isoform multi-task LightGBM regression on pIC50 with NaN labels handled
   via native missing-label support (LGBM ignores NaN per row/col) + auxiliary
   targets (TDI-arm pIC50, Emax, single-conc log2fc) to break sparsity.
3. Validation: scaffold/grouped split per isoform + nearest-neighbor diagnostic
   (train-test similarity distributions) to predict leaderboard-relevant error.
   Score with actual ST-RAE from tutorial `evaluation/`.
4. TDI classifier: LightGBM classifier on 6,145-compound TDI labels, MCC
   threshold tuning under grouped CV.
5. Later iterations: pretrained molecular representations (e.g. Uni-Mol /
   Chemprop / ESM-style), conformer/3D features, stacking, uncertainty-aware
   prediction of the CI band, semi-supervised pseudo-labeling with
   single-concentration data, ensembling seed-diverse GBMs + NNs.

Hardware: Quadro RTX 6000 24GB, 31GB RAM, 12 cores.

## 5. Submission logistics

- Two files (regression + classification), 750 rows each, exact column names
  (case-sensitive), SMILES matching `cyp-challenge-TEST-BLINDED.csv`.
- Validate locally with tutorial `validation/activity_validation.py` +
  `tdi_validation.py` before uploading (12h cooldown between submissions).
- Submit via the Submit tab on the HF space while logged in; we will need
  Jackson's HF account. Open-source code checkbox: repo kept here, can go public.
- Do NOT submit without Jackson's explicit go-ahead.

## 6. Leaderboard status (scraped 2026-09-22, live track)

See `leaderboard_snapshot.md` (regenerated by tools/scrape_lb.py).
