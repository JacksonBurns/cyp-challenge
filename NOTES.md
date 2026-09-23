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

## 6. Leaderboard snapshot 2026-09-22 (files in `leaderboard/`)

Regression (MA-ST-RAE, lower better): top = wbot 0.3804, preheat-to-450 0.3814,
testcyp 0.3891, whippoorwill 0.402. Organizers' public baselines: LGBM 0.893,
XGB 0.897, TabICL 0.675, CheMeleon 0.834, chemprop 0.808. So the field is
stacked: raw-GBM-class ~0.85-0.9, decent ensembles 0.5-0.65, top 0.38.

Classification (MA-MCC, higher better): top = asparagine 0.494, TeamPozeSCAF
0.449 (public doc), then ~0.44 cluster. LGBM baseline ~0.288.

### Open-writeup intel (top-20 entries with public methods)
- **briford** (rank 10, 0.4378, blog supercowpowers.github.io): ensemble of 4
  Chemprop D-MPNNs trained on challenge data + ChEMBL 37 CYP + PubChem AID 1851
  (Veith qHTS), plus per-isoform affine "placement" correction.
- **jeremy** (rank 48, 0.5184 blind; repo jeremycheminf/openadmet_scripts):
  LightGBM ECFP4+RDKit, TabPFN on frozen chemprop embeddings (fine-tuning HURT;
  frozen-encoder-as-features wins), multitask chemprop w/ 17 heads incl. aux
  targets (single-conc log2fc, ChEMBL, AID1851), Caruana bagged ensemble
  selection, and the killer trick below.

### THE BIG ONE: blind-population moment calibration
`BLIND_MOMENTS` (mean/sd of the live scored half of the blind test set,
reverse-engineered publicly by SuperCowPowers from their own scored submissions
via R2 under affine transforms):
  CYP1A2 4.412/1.553, CYP2C9 4.830/1.101, CYP2D6 3.107/1.599, CYP3A4 4.880/1.272
vs OUR TRAIN label means ~4.96/4.58/4.78/4.10 (2D6!) and test-set design means
very different distributions (test enriched for 1A2/2C9/3A4 actives, 2D6 is
near-random). jeremy: applying moment-calibration flipped CYP2D6 R2 from -0.75
to +0.38 and improved blind ST-RAE 0.72 -> 0.52. We MUST z-score our test
predictions per isoform onto BLIND_MOMENTS (shrunk by OOF Pearson rho) before
submitting. This is public info from an open repo - legitimate to use.

### Verified data facts (2026-09-22, check_labels.py)
- TDI file's direct-inhibition columns are IDENTICAL to TRAIN_inhibition for
  overlapping compounds; adds 1,240 extra compounds with NO direct pIC50 but
  WITH TDI-arm pIC50 + is_TDI labels (3A4: 1238 labeled; 2D6: only 2).
- is_TDI rates (labeled): 2D6 21.6% pos (324/1497), 3A4 21.3% pos (764/3584).

## 7. FINAL SUBMISSION STATE (2026-09-22 evening)

Both files pass official validators, 750 rows, exact SMILES/Molecule_Name
row-for-row match to TEST-BLINDED (see `src/verify_submissions.py` output):
- `cache/raw_regression_submission.csv` - 4-config x 3-seed LGBM blend,
  BLIND_MOMENTS-calibrated (means 4.412/4.830/3.571/4.880; 2D6 uses ST-RAE
  moments 3.57/0.90). OOF MA-ST-RAE 0.743; expected blind ~0.5-0.62.
- `cache/tdi_submission.csv` - per-isoform fraction thresholds (see below).

### TDI threshold decision (threshold-transfer analysis)
- Raw OOF-argmax thr (3A4 0.08) gives 38.9% test positive rate vs 21.3% train
  - NOT suspicious: probas over-dispersed, and test genuinely enriched.
- KEY: label rule ties is_TDI to direct pIC50. Train TDI-pos rate by direct
  pIC50 decile: flat ~0.44-0.55 for deciles 3-9, ~0-5% for bottom 3. Our
  calibrated test predictions (enriched: 3A4 mean 4.88, floor ~2) imply blind
  positive rate ~0.35-0.45 for 3A4, ~0.3-0.38 for 2D6 - NOT the train 21%.
- Simulator lesson (wasted effort first pass): raw OOF probas are
  over-dispersed (raw top-8% mean prob 0.44 but observed rate 0.36); use
  observed rank-conditional curves, not raw probas, to simulate.
- Expected-MCC analysis (cache/tdi_fraction_optima.json):
  3A4: plateau f=0.35-0.50 across plausible pi; SHIPPED f=0.43 (thr 0.061,
  322/750 pos) - close to minimax-optimal, near OOF-argmax-equiv 0.39.
  Rate-matching to 21.3% would cost ~0.05-0.08 expected MCC.
  2D6: rank signal weak/flat (top-25% rate only 0.42); best f~0.08 everywhere
  regardless of pi; SHIPPED f=0.08 (thr 0.2726, 60/750) = quantile-matched to
  OOF argmax. Rate-matching (21.6%) costs ~0.03 MCC. Do NOT read 2D6 8% as
  "matched train operating point" luck - the OOF argmax fraction is simply
  the right choice for this isoform.
- OOF numbers (rerun, cache/tdi_cv.json): 3A4 argmax MCC 0.4095 @ 0.08; at
  our shipped f=0.43 the OOF top-k MCC is 0.394 (equiv thr 0.024) - barely
  worse than argmax even on OOF. 2D6: 0.1533 argmax @ 0.27; shipped f=0.08
  gives OOF 0.1495. Blended shipped OOF MCC ~0.27; expected blind MA-MCC
  maybe 0.25-0.45 (higher if enrichment thesis correct). Top LB 0.494.
- Scripts: src/tdi_threshold_analysis.py, src/tdi_fraction_opt.py,
  src/verify_submissions.py. cache/tdi_threshold_analysis.json,
  cache/tdi_fraction_optima.json.
- IMPORTANT for Nov 3: if interim reveal (Sep 25) shows actual test positive
  rates, refit thresholds directly (score-percentile inversion - no retrain).

### Gotcha added the hard way
- cache/tdi_oof.npz arrays are in X_train-dedup row order; TDI csv is a
  different row order. Always rebuild labels via the same
  concat-tdi+emx/drop_duplicates/reindex(Xtr SMILES) as run_tdi.py before
  slicing with a boolean mask, or MCC checks silently compute garbage.

## 8. WHAT WE BUILT + EXPECTATIONS FOR FUTURE ITERATIONS (2026-09-22)

### Models implemented (exact recipe)

**Features** (`src/featurize.py` -> cache/X_{train,test}.parquet, one row per
SMILES, 3455 cols): RDKit `Descriptors._descList` (nonfinite->0), Morgan
count FP r=2/2048, Morgan r=3/1024, MACCS 167. No scaling (tree models).

**Regression** (`src/run_regression.py` library; `src/sweep.py` CV;
`src/final_submit.py` final fit):
- One LightGBM regressor per isoform on pIC50 direct-inhibition labels.
  4 configs (differ in n_estimators/LR/leaves/min_child/regularization;
  canonical values re-injected in cache/cv_sweep.json "params"), 3 seeds
  each, equal-weight average of the 12 = per-isoform blend.
- Extra label-derived features: k-NN target features from Morgan-Tanimoto
  nearest labeled train rows (`nn_block` in run_regression.py) - top-1 NN
  label, top-10 mean label, Tanimoto sims. MUST self-exclude in CV (q_idx).
- Validation: scaffold-grouped 5-fold (`make_folds`/`scaffold_groups`,
  Murcko scaffolds; groups split so each fold gets whole scaffolds).
- Blend scaffold-OOF MA-ST-RAE 0.743 (per-isoform 1A2 0.821 / 2C9 0.692 /
  2D6 0.947 / 3A4 0.512). Stacked multitask with TDI-arm aux: worse (0.770).
- Calibration (THE thing that matters): final fit retrains on all labeled
  rows, z-scores each isoform's test preds onto BLIND_MOMENTS with
  target_sd = clip(rho_oof * inflation, 0.05, 0.95) * sd_blind, where
  rho_oof = OOF Pearson (0.527/0.604/0.399/0.771, cache/oof_pearson.json)
  and inflation OOF_TO_BLIND = 1.32/1.23/1.66/1.07 (per-isoform fudge,
  tuned so ST-RAE-optimal on the moment-matched assumption; CYP2D6 targets
  ST-RAE-specific moments mean 3.57 sd 0.90 instead of the BLIND_MOMENTS
  3.107/1.599 because 2D6 test is near-random; all preds floored at 1.0).

**TDI classification** (`src/run_tdi.py`): LightGBM classifier per scored
isoform (2D6 n=1497, 3A4 n=3584 labeled), same features, same scaffold CV.
Plus per-fold "NN-positive" features: max / top-10-mean / count>0.7
Tanimoto similarity from query to in-fold labeled POSITIVES (self-excluded).
Test preds = mean of 5 fold-models (cache/tdi_test_probs.csv). Threshold
decision is separate (section 7; src/tdi_threshold_analysis.py +
src/tdi_fraction_opt.py simulate expected MCC over base-rate hypotheses).
`src/tdi_shift.py` (regress TDI-arm, apply label rule directly) was written
but never finished - classifier won by default, may still be worth finishing
for 2D6 blending.

### Expectations (what to beat / how to know if you're doing better)
- Regression: our scaffold-OOF says 0.743 but BLIND_MOMENTS calibration
  means the blind number should land ~0.50-0.62 (jeremy's comparable
  pipeline + same calibration = 0.518). Leaderboard: top 0.38; good
  ensembles 0.5-0.65; raw GBMs 0.85-0.9. If your OOF improves but a new
  calibrated submission does not beat the interim reveal of this file,
  trust the blind number, not OOF.
- TDI: shipped OOF blend ~0.28; blind MA-MCC expectation 0.25-0.45 with
  real spread because the 3A4 enrichment thesis (blind pos rate ~0.40) is
  a modeled bet, not observed. Top 0.494. The interim reveal (Sep 25) will
  show per-column stats that mostly confirm/deny the thesis - check the
  actual positive counts before iterating on models.
- Biggest untried levers, roughly in expected-value order:
  1. External data: ChEMBL 37 CYP + PubChem AID1851 (briford's blog says
     this + Chemprop ensemble got 0.438; also good for TDI pretraining).
  2. Chemprop D-MPNN (or frozen pretrained embeddings as LGBM features -
     jeremy: frozen beats fine-tuned) ensembled with the GBM blend.
     CheMeLeon too: labels are only ~1.3-2.3k/isoform (small-data regime);
     see docs/CHEMPROP_CHEMELEON_PLAN.md for the staged experiment plan and
     local assets (weights + conda envs already on this box).
  3. Finish tdi_shift.py for 2D6 (classifier signal there is weak).
  4. Uncertainty-aware placement: predict INTO the per-compound CI band
     (ST-RAE is zero inside band) rather than z-matching moments.
  5. Re-tune OOF_TO_BLIND inflations against the Sep 25 reveal once we
     have our own scored blind result (one data point per isoform, but
     real, vs the analogy we used now).
- Submission protocol: verify with src/verify_submissions.py (both official
  validators + row-for-row SMILES/name match) before every upload; 12h
  cooldown; latest valid submission counts; NEVER submit without Jackson's
  explicit go-ahead (his HF login on the Submit tab of
  https://huggingface.co/spaces/openadmet/cyp-challenge).

## 9. INTERIM REVEAL RESULTS (Sep 23 2026) - READ THIS FIRST

> **STATUS UPDATE (Sep 23, post-reveal work session):** Section 9 diagnosis
> stands. Since then: CheMeLeon frozen embeddings extracted + gated (sec 10),
> candidate `cache/regression_v2_submission.csv` built (GBM x embeddings
> mixture + reveal-calibrated spread shrink) - NOT submitted, verifier-passing;
> external data downloaded (sec 10). Next: external-data features, TDI upgrade.

Files: leaderboard/regression_2026-09-23_interim_reveal.csv (229 entries),
leaderboard/classification_2026-09-23_interim_reveal.csv (121 entries).

### Our scored interim numbers
- Regression: JacksonBurns rank 93/229, MA-ST-RAE 0.6766, MAE 0.808,
  MA-R2 0.387, MA-Spearman 0.616.
- Classification: rank 60/121, MA-MCC 0.3149, precision 0.372, recall 0.556,
  accuracy 0.789.

### Diagnosis - regression (what the reveal actually tells us)
- The calibration thesis WORKED directionally: raw-GBM-class is ~0.85-0.9,
  our uncalibrated-scaffold-OOF equivalent sits near 0.74, and our scored
  blind is 0.677. But 0.5-0.62 was optimistic; we expected jeremy-with-our-
  ranking, got a mid-table rank.
- KEY DIAGNOSTIC: our blind MA-R2 (0.387) ~= our OOF Pearson^2 (mean of
  squared per-isoform OOF Pearson = 0.349). R2 under affine placement is
  maximized at corr^2, so (a) our BLIND corr ~ OOF corr (ratio ~1.05, NOT
  the 1.32 we assumed in OOF_TO_BLIND), and (b) our placement is already
  near R2-optimal. The remaining gap is 100% RANKING quality:
  top field Spearman 0.75-0.79 / R2 0.62-0.68 (preheat 0.374, wbot 0.380,
  testcyp 0.389); jeremy 0.734/0.602; us 0.616/0.387. No placement trick
  closes an 0.62 vs 0.75 correlation gap - only better models do.
- Concrete correction for next submission: OOF_TO_BLIND inflations are
  overestimated by ~1.2-1.3x on average (macro-averaged, so per-isoform
  caveat); our preds were slightly over-dispersed. Re-centering them down
  by that factor may win 0.02-0.05 ST-RAE cheaply, but ranking is the
  main event.
- Leaderboard pattern: top ~20 ALL have external data (briford 0.438 blog
  open; C_CYPher-cypext-* 0.443; CYPext-x-endpt 0.473) or non-fp deep
  features. 'random-forest' 0.447 and 'SVM' 0.464 show well-calibrated
  classic models + presumably external data reach 0.44-0.46.
  CheMeLeon evidence: KalenJosifovski 'CYP CheMeLeon MT+TDI TabPFN' scored
  0.631 (better than us, Spearman 0.689 vs our 0.616) - CheMeLeon
  embeddings add real signal on this task; baseline raw CheMeLeon is 0.834.

### Diagnosis - classification
- 0.3149 is mid-pack: above LGBM-baseline (0.288), ~= cypster's
  'LightGBM ECFP4+RDKit multi-seed TDI v2' (0.316, same family as ours),
  ~0.09 below jeremy (0.402, open code), 0.20 below nova (0.513).
- Our profile recall 0.556 / precision 0.372 vs nova 0.505/0.708:
  we over-predict positives relative to our ranking quality. The 3A4
  enrichment bet (43% fraction) was NOT confirmed as wrong nor right -
  recall/precision are macro, so true per-isoform positive rates are NOT
  recoverable from this CSV. But at our ROC the fraction is second-order;
  ranking dominates. Do not re-litigate 43% vs 21% until the classifier
  gets stronger; retune fraction via the same expected-MCC machinery
  (src/tdi_fraction_opt.py) with any new OOF.
- Public method docs to mine: jeremy github (jeremycheminf/openadmet_scripts
  /CYP_Challenge, regression+TDI), TeamPozeSCAF Google Doc (TDI 0.449),
  briford supercowpowers blog (regression 0.438), Lizard Wizard Gizzard
  METHOD.md (TDI 0.284 - what not to do: recall 0.73/precision 0.30).

### Revised expectations & priorities (Nov 3 final)
- Getting to 0.45 reg / 0.42 TDI requires BOTH: (1) external data
  (ChEMBL 37 CYP + PubChem AID1851) - this is now near-mandatory, every
  top-20 entry has it; (2) ranking upgrade from D-MPNN/CheMeLeon
  embeddings stacked on the GBM (KalenJosifovski proves CheMeLeon lifts
  ranking: Spearman +0.07 over our recipe with roughly TabPFN head).
- docs/CHEMPROP_CHEMELEON_PLAN.md stays the plan; its step 1 (frozen
  embeddings -> LGBM, must beat per-isoform OOF Pearson) is still first.
- Cheap wins to try in parallel: (a) recenter OOF_TO_BLIND by /1.25;
  (b) per-isoform placement search using our scored point + BLIND_MOMENTS;
  (c) TDI: retrain on train+external positives, keep fraction machinery.
- We get exactly ONE more scored data point per track at final. Make the
  last submission the best-calibrated one, keep a safe file.

## 10. POST-REVEAL ITERATION LOG (Sep 23)

### CheMeLeon frozen embeddings (plan step 1) - MIXED, useful only in mixture
- `src/extract_chemeleon.py` (chemeleon env): frozen BondMessagePassing from
  `~/chemeleon_nazarov/chemeleon_mp.pt`, mean-pooled 2048-d atom states,
  6895 compounds in ~6s GPU -> `cache/chemeleon_emb.parquet`. Chemprop 2.3.1
  API gotchas: `BatchMolGraph.to()` is in-place (returns None); `mp(bmg)`
  returns the node tensor directly; `agg(H, bmg.batch)`.
- Naive concat with FP features (`src/sweep_emb.py`): per-isoform OOF Pearson
  vs config-A baseline (0.523/0.598/0.397/0.767): 1A2 -0.009, 2C9 +0.008,
  2D6 +0.030, 3A4 -0.011. Emb-only is worse everywhere. Raw 2048-d too noisy
  for GBM on 1.3-2.3k rows (matches CheMeleon skill's descriptor-bank lesson).
- Ridge linear probe on embeddings alone: 0.491/0.598/0.363/0.738 - the
  embeddings alone nearly match the whole FP+desc GBM on 2C9/3A4. Signal is real.
- **OOF z-space mixture wins everywhere** (`src/emb_blend.py` -> 
  cache/emb_blend.json): best mix w_gbm in {1A2 0.6: 0.535, 2C9 0.4: 0.617,
  2D6 0.4: 0.436, 3A4 0.7: 0.773} vs gate 0.527/0.604/0.399/0.771. GATE PASSED
  on all 4 isoforms (vs shipped blend), biggest gain 2D6 (+0.037).
- Shipped-config comparison note: gate used the 4cfg x 3seed blend, my CV used
  config A only; mixture gain therefore understates the shipped-config blend.

### Candidate regression v2 - BUILT, VERIFIER-PASSING, NOT SUBMITTED
- `src/regression_v2.py`: FP-GBM blend (exact shipped 4cfg x 3seed) + same
  recipe with embeddings concatenated; z-space mixture per isoform (weights as
  above); placement = same moments but spread shrunk by F_SPREAD (below);
  raw per-model test preds saved to `cache/blend_test_preds.npz` (reusable).
- Output `cache/regression_v2_submission.csv` passes official validator +
  row-for-row SMILES/name match (`src/verify_submissions.py` now takes paths).
- OOF rho of the mixture (placement input): 0.535/0.617/0.436/0.773.

### Task 4 placement-shrink analysis (src/shrink_placement.py)
- Reveal-fit: MA-R2 0.387 with BLIND_MOMENTS exact + our placed sds implies
  k = rho_blind/rho_oof ~= 1.05-1.09 (incl. 2D6 mean-shift penalty). So
  OOF_TO_BLIND inflations were ~1.26/1.17/1.58/1.02x too big on average.
- ST-RAE sims (truth = train label dist + real DRC bands; both train-dist and
  chemistractive-enriched variants): optimal placement-spread factor is well
  below 1 for 1A2/2C9 (0.45-0.65), 0.7-0.85 for 3A4, ~1.0 for 2D6
  (STRAE_MOMENTS). Absolute sim levels do NOT transfer (train-dist OOF-equiv
  ~0.9-1.5 vs actual 0.677); use the SHAPE (direction+relativity) only.
- Leaderboard check: at our R2 band (0.35-0.42) entries span ST-RAE
  0.597-0.782 (n=10) - placement-only headroom ~0.08 at constant ranking.
- Adopted (in regression_v2): F_SPREAD = {1A2 0.6, 2C9 0.6, 2D6 1.0, 3A4 0.7}.
  Expected gain: sim shape ~0.05-0.10 isoform-average, discounted for sim
  transfer uncertainty -> ~0.03-0.06. Ranking gain from v2 mixture adds more.
- CAUTION: this is expectation-based; final submission should be re-checked
  against the one scored data point we get at the Nov 3 reveal.

### External data (plan step 4) - DOWNLOADED
- `external/chembl_cyp_ic50.csv`: 36,015 IC50 activities, 14,453 unique
  canonical SMILES (CHEMBL3356 1A2 5,924 / CHEMBL3397 2C9 7,520 / CHEMBL289
  2D6 8,684 / CHEMBL340 3A4 13,887) via paginated API (page cap 1000).
- `external/pubchem_cyp_qhts_aid1851.csv`: 14,496 CIDs, per-isoform
  Fit_LogAC50-derived pIC50 + binary Active/Inactive (Veith qHTS), SMILES via
  PUG REST batched POST with recursive splitting (server 503s are routine).
- RULE: these are NOT DRC-calibrated - use only for RANKING (pretraining /
  multitask auxiliary / similar-compound features), never placement moments.
- Jeremy's public repo cloned at /tmp/jeremyscripts (his AID1851 access
  pattern + Caruana ensemble + calibration code worth mining for TDI).

### TDI with embeddings - NOT a win by concat
- `src/run_tdi_emb.py` + `src/tdi_emb_blend.py`: emb-augmented OOF MCC
  3A4 0.382/2D6 0.105 vs base 0.4095/0.153; z-mixtures: 3A4 best 0.4136 @
  w_base=0.7 (small gain), 2D6 best is base-only. TDI keeps base classifier;
  rerun fraction machinery (src/tdi_fraction_opt.py) on any final OOF.
- Task-5 rerun on the blended OOF (`src/tdi_fraction_opt_v2.py` ->
  cache/tdi_fraction_optima_v2.json): 3A4 E[MCC] plateau 0.37-0.58 (shipped
  f=0.43 sits IN the plateau at 0.485); 2D6 optimum f~0.08-0.10 = shipped.
  No threshold change warranted; ranking is the only lever. Reveal profile
  (ours precision 0.372/recall 0.556 vs nova 0.708/0.505) = ROC-curve
  quality, not operating point.

### CheMeLeon FINE-TUNES (plan step 2) - BIG WIN, beats gate standalone
- `src/ft_chemeleon.py` (chemeleon env): multitask 4-head FFN on CheMeLeon MP,
  our scaffold folds (seed 0), NaN-masked, targets normalized (UnscaleTransform),
  10% internal val for EarlyStopping, ~4-9 min/run. Modes: frozen MP (8.4M
  trainable) and --full-ft (17.1M). OOF -> cache/ft_oof_<tag>.csv,
  test preds -> cache/ft_test_preds_<tag>.csv (eval via src/eval_ft.py).
- OOF Pearson (frozen / fullft) vs gate 0.527/0.604/0.399/0.771:
  frozen 0.449/0.603/0.340/0.745; **fullft 0.527/0.640/0.383/0.766**.
  Full fine-tune BEATS the gate standalone on 2C9 (+0.036) - contradicts
  jeremy's "FT hurts" on OUR folds; frozen wins only as mixture filler.
- **4-way N-mixture** (`src/blendN.py` -> cache/blendN.json; z-space greedy
  w/ replacement over gbm/emb/ft_frozen/ft_fullft): per-isoform Pearson
  0.566/0.668/0.442/0.799 vs gate 0.527/0.604/0.399/0.771. Macro rho^2
  0.400 vs shipped-implied 0.349. Weights: 1A2 {gbm .4, emb .1, ft_full .5};
  2C9 {gbm .2, emb .15, ft_frz .15, ft_full .5}; 2D6 {gbm .25, emb .5,
  ft_full .25}; 3A4 {gbm .45, ft_frz .15, ft_full .4}.
- `src/regression_v3.py` builds **cache/regression_v3_submission.csv** from
  these weights + reveal placement (moments, F_SPREAD, OOF_TO_BLIND k-adjust
  folded into rho assumption). PASSES official validator + row-for-row match.
  Raw test preds all cached (blend_test_preds.npz + ft_test_preds_*.csv) so
  placement knobs are cheap to re-tune without retraining.

### External features into GBM - NO tabular win (deep route is the way)
- `src/external_features.py` -> cache/ext_feats_{train,test}.parquet: per-iso
  external-NN features (ChEMBL median IC50 pIC50 + AID1851 Fit_LogAC50/binary;
  top1sim/wavg/top10/count7/binrate5). Overlap audit: only 11 train SMILES in
  ChEMBL, 0 test (src/ext_overlap_audit.py).
- CV (`src/sweep_ext.py`): ext 0.524/0.601/0.398/0.764 and extemb
  0.519/0.608/0.431/0.758 - FP+NN features already saturate that signal
  tabularly. External data should enter via deep multitask (jeremy 06_ script,
  briford blog) - next lever if time allows.
- Jeremy's README (cloned /tmp/jeremyscripts): kit OOF 0.614->blind 0.5214
  WITH blind-moments calibration (matches our mechanism); TDI uses TabICL on
  frozen chemprop embeddings (TabICL > TabPFN there); AID1851 chosen over
  ChEMBL pretrain because ChEMBL population is ~1-1.65 log more potent.

### Session 3 (Sep 23, continued from crashed session) - seed averaging + ext multitask
- Seed-1 full-FT OOF (cache/ft_oof_fullft_s1.csv, eval_ft.py fullft_s1):
  0.530/0.655/0.367/0.761 - consistent with seed 0 (0.527/0.640/0.383/0.766).
- **Seed-averaged full-FT** (z-mean of both seeds; src/blendN_seedavg.py ->
  cache/blendN_seedavg.json): standalone avg singles 0.540/0.659/0.387/0.772;
  4-way greedy 0.570/0.676/0.441/0.799, macro R2 0.404 (vs single-seed blendN
  0.566/0.668/0.442/0.799, macro 0.400). Small but free gain; adopted.
- `src/regression_final.py`: generalized submission builder - resolves blend
  npz keys, per-tag ft_test_preds CSVs, and "<tag>_avg" = z-mean across all
  matching seed files. Built cache/regression_final_submission.csv from
  blendN_seedavg: PASSES official validators + row-for-row SMILES/name match.
  Placement knobs identical to regression_v3 (F_SPREAD/OOF_TO_BLIND/moments).
- `src/ft_ext.py` (fixed placeholder bugs, smoke-tested 1-fold + 2-epoch):
  12-head multitask full-FT - 4 challenge primaries (inverse-count task wts)
  + 4 ChEMBL IC50 + 4 AID1851 aux heads (flat 0.3 wt). 29,130-row table
  (6,145 challenge + 9,159 ChEMBL + 13,826 PubChem); verified label
  separation (challenge rows never carry aux labels and vice versa). External
  subsampled to EXT_CAP=9,000/fold for throughput. Queue:
  tools/queue_ext_tabicl.sh (ft_ext then tdi_tabicl on GPU).
- `src/tdi_tabicl.py` (jeremy pattern): TabICL classifier on 2048-d CheMeLeon
  embeddings, scaffold folds, per-iso 2D6/3A4; writes tdi_tabicl_oof.npz +
  test probs; evaluated vs base classifier OOF + fraction machinery next.
