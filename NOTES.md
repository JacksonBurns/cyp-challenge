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

### CheMeLeon external-aux multitask ft_ext - WIN, new best blend
- src/ft_ext.py run completed (full-FT, 12 heads): inf crash root cause was 5
  ChEMBL standard_value=0 rows -> log10 -> inf (now filtered; pyarrow also
  installed into tabicl-chemeleon env which was missing parquet support).
- OOF (src/eval_ft.py ext): 0.534/0.630/0.378/0.761 standalone - weaker than
  seed-avg full-FT on 2C9/3A4 but complements it better.
- 5-way blend (src/blendN_ext.py -> cache/blendN_ext.json):
  **0.578/0.681/0.444/0.803, macro R2 0.410** (seed-avg 4-way was 0.404,
  single-seed 0.400, shipped-implied 0.349). ft_ext takes 20-35% weight on
  every isoform.
- cache/regression_final_ext_submission.csv BUILT (regression_final.py
  --blend blendN_ext.json) - PASSES official validators + row-for-row match.
- GPU queue lesson: background wrapper got SIGTERM'd mid-queue once (fold OOF
  survived because oof.to_csv happens before the all-data model; added
  --skip-folds reuse flag to avoid redoing 5 folds).

### TDI TabICL + 3-way blend - WIN (both isoforms)
- src/tdi_tabicl.py OOM-killed by memory cgroup at 2048-d embeddings (kernel
  log: "Memory cgroup out of memory"); fixed with unsupervised PCA-256
  (whitened, fit on train+test embeddings only) + n_estimators=4. Runs ~35 min.
- TabICL standalone OOF MCC: 2D6 0.141, 3A4 0.372 (both below base).
- 3-way z-blend (src/tdi_blend3.py -> cache/tdi_blend3.json):
  2D6 0.9*base+0.1*emb -> MCC 0.161 (base 0.150) @ f=0.08
  3A4 0.6*base+0.2*emb+0.2*tabicl -> MCC 0.427 (base 0.407) @ f=0.36
- Fraction machinery rerun on the blended score (src/tdi_fraction_opt_v3.py):
  2D6 f=0.08 confirmed (E[MCC] 0.156); 3A4 plateau 0.36-0.50, minimax 0.38 ->
  chose 0.36 (OOF-argmax of blend also 0.36; shipped v1 used 0.43).
- cache/tdi_submission_v2.csv BUILT (src/make_tdi_submission_v2.py: same
  weights, test z-space) - PASSES official validator + row-for-row match.
  Note: blend weight selection is OOF-argmax on ~600/2400 labeled rows;
  weight gains are small (+0.01-0.02 MCC) so treated as mild, direction is
  robust (blends beat standalone everywhere).

### Nested blend check (selection-overfit audit)
- src/blend_nested_check.py: per-fold greedy weight selection scored on the
  held-out fold (Fisher-averaged). Honest macro R2: 4way_seedavg 0.4063 vs
  5way_ext 0.4106 - matches the in-sample blendN_ext numbers (0.404/0.410):
  the greedy weights barely overfit, ext gain is real.

### CANDIDATE FILES (all verified, NOT submitted - need Jackson's go-ahead)
- Regression: cache/regression_final_ext_submission.csv (5-way, macro R2
  0.410 honest) [best]; regression_final_submission.csv (4-way seed-avg,
  0.404); regression_v3_submission.csv (single-seed, 0.400); interim safe
  file raw_regression_submission.csv (scored 0.677).
- TDI: cache/tdi_submission_v2.csv (3-way blend) [best]; tdi_submission.csv
  (shipped 0.315 baseline).

### ft_ext seed-1 + final candidate refresh
- ft_ext seed-1 done (auto-suffixed outputs _s1; blendN_ext + regression_final
  now seed-average ext automatically via glob). ext seed-avg singles
  0.545/0.653/0.407/0.773; 5-way blend 0.578/0.683/0.449/0.804,
  **macro R2 0.412**; nested-honest 0.4127 (selection-overfit-free).
- regression_final_ext_submission.csv rebuilt on the 2-seed ext (verified).
- GPU lesson (Hermes): foreground terminal caps at ~420s and SIGTERM's the
  child; long GPU jobs MUST use terminal(background=true) - wrappers that
  exit also SIGTERM queued children, so one script per tracked process.

## 11. SECOND SCORED POINT - the v3 blend was submitted (Sep 23 evening)

Jackson submitted cache/regression_final_ext_submission.csv and
cache/tdi_submission_v2.csv at 19:41 UTC (leaderboard "Submitted" timestamps;
the live entries confirm our blend is what is on the board, Open Code = Yes,
repo pinned at ee5ab9a). New leaderboards saved:
leaderboard/{regression,classification}_2026-09-23_final_reveal.csv.

### Our scored numbers (238 reg / 125 TDI entries now)
- Regression: rank 73/238, MA-ST-RAE 0.5870, MAE 0.760, R2 0.4472,
  Spearman 0.6799, Kendall 0.5078. (was 93/229: ST-RAE 0.6766, R2 0.387)
- TDI: rank 39/125, MA-MCC 0.3419, acc 0.8115, prec 0.4079, rec 0.5368,
  F1 0.4613. (was 60/121: MCC 0.3149)
- Gains delivered: regression ST-RAE -0.090, R2 +0.060; TDI MCC +0.027.
  Both slightly UNDER the OOF prediction (blend macro R2 0.412 vs realized
  R2 0.4472 -> k = sqrt(0.4472/0.412) = 1.042, consistent with the first
  scored point k=1.053; but TDI OOF blend MCC 0.398 -> blind 0.342, so TDI
  transfer ratio ~0.86 - blend gains do NOT fully transfer on classification).

### Regression diagnosis (2 scored points now)
- ST-RAE vs R2 field fit (R2 in [0.40,0.50]): ST-RAE ~ 0.695 - 0.210*R2;
  at our R2 the fit predicts 0.601, we score 0.587 - PLACEMENT IS NOW
  SLIGHTLY BETTER THAN FIELD-AVERAGE. Placement headroom is exhausted
  (~0.01-0.02); do not spend more time on shrink knobs. F_SPREAD as shipped
  was validated empirically.
- Top-12 all have Spearman 0.75-0.79 / R2 0.58-0.68. We are 0.680/0.447.
  The entire remaining gap (0.587 -> ~0.43 = top-12 band) is RANKING:
  need per-isoform Pearson ~+0.08-0.10 (Spearman 0.68 -> 0.75+).
- OOF says the current blend reaches 0.578/0.683/0.449/0.804 (macro rho^2
  0.412). Field top implies test-equivalent Pearson ~0.76-0.80 macro.
  Gap to close: 1A2 +0.14, 2C9 +0.10, 2D6 +0.14, 3A4 +0.08 (2D6 is the
  weakest head, both abs and relative to field).

### TDI diagnosis
- Top band MCC 0.41-0.51 (nova 0.513). We are 0.342 - still mid-pack,
  above baselines (LGBM 0.288, TabICL-baseline 0.325). Profile prec 0.408/
  rec 0.537 vs top-10 ~0.55-0.71 prec: our ranking quality at the chosen
  operating point still over-predicts positives. With better ranking,
  retune fraction (tdi_fraction_opt_v3) - fraction stays second-order.
- The blend gains (OOF +0.01-0.02) were real but small; the step function
  on TDI is what the top entries exploit (nova: high precision at f~0.5;
  public reports to mine: TeamPozeSCAF Google doc, jeremy repo - he now
  scores 0.402 TDI / 0.518 reg, both open code).

### Nov 3 game plan (final submission = latest valid counts; 1/12h)
Current submitted files are OUR BEST and are ON THE BOARD. Everything below
must beat them on OOF before swapping in; keep them safe otherwise.
1. Regression ranking (priority 1, biggest measurable headroom):
   - ft_ext seeds 3-5 (cheap, proven; seed-avg gain still open).
   - ChEMBL pretrain-then-finetune (the big untried lever; pretrain the
     CheMeLeon MP on the 36k ChEMBL CYP pIC50 multitask, then fine-tune
     heads on challenge data; population-shift caveat per jeremy README).
   - D-MPNN (chemprop) with external multitask aux as a 6th blend member.
   - Target: blend macro rho^2 >= 0.45 (per-iso ~0.62/0.71/0.52/0.82).
2. TDI ranking (priority 2): retrain base classifier on train+external
   positives (AID1851 binaries - near-mandatory, all top entries have it);
   add TabICL-on-emb as full blend member with per-fold nested weights;
   consider jeremy's TabICL-on-chemprop-emb route directly (his 0.402).
3. Placement: leave regression placement machinery AS IS unless OOF rho
   moves >0.03 (then re-fit k on the two scored points: k ~ 1.04-1.05,
   NOT the original 1.2-1.6 OOF_TO_BLIND; recompute F_SPREAD from
   shrink sims at the new rho). Re-verify with official validators +
   src/verify_submissions.py, 12h cadence, final submit >= Nov 1.
4. Decision rule at Nov 3: submit the highest-OOF verified candidate;
   if nothing beats current, do not resubmit (latest counts - the two
   files on the board are already the best candidates built so far).

## 12. SESSION 4 (Sep 23 night -> Sep 24) - seeds 2-4, pretrain test, jeremy ckpts mined
### Stage 1 DONE: ft_ext seeds 2-4 (4-seed ext family)
- ft_ext seeds 2/3/4 added (~30 min each; tools/run_ext_seeds234.sh).
- 4-seed ext singles 0.558/0.663/0.417/0.780; ext2 blend (gbm,emb,ft_frozen,
  ft_fullft,ft_ext): 0.582/0.684/0.450/0.804, macro R2 0.4141 (was 0.412 with
  2 ext seeds; nested-honest 0.4155 vs 0.4127). Gain small but real.
- cache/regression_final_ext4_seed_submission.csv BUILT from blendN_ext2.json
  + verified (official validator + row-for-row). NOT submitted (pending;
  marginal +0.002, and 12h cooldown from 19:41 UTC means earliest slot is
  07:41 UTC Sep 24 anyway).
### Stage 2 IN PROGRESS: CheMeLeon pretrain-then-finetune - looking WEAK
- src/pretrain_mp.py: MP pretrained on 20k rows ChEMBL+AID1851 union (4 CYP
  heads, 10 epochs, 4 min) -> cache/mp_pretrained_cyp.pt.
- src/ft_ext.py now takes --pretrained + --prefix (outputs ft_oof_pre*.csv,
  disjoint from the ext seed glob).
- pre seed-0 standalone OOF: 0.528/0.602/0.390/0.745 vs 4-seed ext singles
  0.558/0.663/0.417/0.780 - WORSE on every isoform. Matches jeremy's
  population-shift warning (his fix was AID1851-weighted pretraining; we
  mixed both scales already, didn't help). Seed-1 running; will test
  pre6 pool for blend value anyway (decorrelation could still pay).
### TDI external-positives base classifier - FAILED on both weights tested
- src/run_tdi_ext.py adds 13,447 AID1851 rows (challenge-excluded, deduped)
  to the LightGBM base with sample_weight W_EXT; OOF scored on challenge only.
- W_EXT=0.3: 2D6 OOF MCC 0.060 (base 0.150), 3A4 0.285 (base 0.410). The
  qHTS population DRASTICALLY shifts the decision function - naive pooling
  hurts exactly like jeremy warned for fine-tuning. W_EXT=0.1 running.
- CONCLUSION so far: jeremy's TDI gain comes from a DIFFERENT mechanism:
  TabICL on frozen adme_pretrain embeddings (checkpoints chemprop_medium.pt
  / chemprop_chemeleon.pt shipped in /tmp/jeremyscripts/CYP_Challenge/
  checkpoints/, loadable via chemprop models.MPNN.load_from_file; 600-d
  pre-FFN output). src/cp_embeddings.py (regression ridge probes, blend-ready
  outputs ft_oof_cpmed/cpchm) + src/tdi_tabicl_cp.py (TabICL TDI on cpmed,
  optional --ext 1 to add AID1851 binaries as in-context rows) queued.

### Session 4 RESULTS (Sep 24 early) - NEW CANDIDATES, both beats scored points
#### Regression: cp7 pool = gbm+emb+ft_frozen+ft_fullft+ft_ext+ft_pre+ft_dmpnn+ft_cpmed+ft_cpchm
- WINNERS: jeremy's FROZEN adme_pretrain checkpoints (his public kit at
  /tmp/jeremyscripts/CYP_Challenge/checkpoints/) + ridge probe (alpha grid
  100/1000/10000; small alphas overfit and gave -0.04 garbage, big win with
  correct grid): cpmed 0.550/0.701/0.407/0.800, cpchm 0.548/0.639/0.419/0.764
  standalone scaffold-OOF Pearson.
- dmpnn from-scratch (chemprop-dev, MultiHot v1 atoms, 12-head ext multitask,
  seed 7, 10 min total - far faster than CheMeLeon FT): weak standalone
  (0.462/0.570/0.368/0.738) but adds blend weight on every isoform.
- pre (ChEMBL-pretrained CheMeLeon FT): WEAK standalone (0.524-0.528 avg per
  iso below ext), consistent with jeremy's population-shift warning; still
  gets 15-25% greedy weight on 1A2/2D6 via decorrelation.
- blendN cp7: 0.596/0.716/0.464/0.821, macro R2 0.4391 (nested-honest 0.4385;
  all7 without cp members: 0.4184/0.4192; ext2 4-seed: 0.4141/0.4155; the
  two scored files' blend: 0.412/0.4127). OOF rho up +0.014..+0.033 per iso
  vs submitted -> within the 0.03 rule, placement UNCHANGED (F_SPREAD/
  OOF_TO_BLIND/BLIND_MOMENTS as shipped, validated by both scored points).
- NEW CANDIDATE cache/regression_final_cp7_submission.csv - official validator
  + row-for-row verified. (regression_final_ext4_seed_submission.csv = 0.4141
  interim candidate, superseded.)
#### TDI: TabICL-on-chemprop_medium embeddings is the win jeremy's repo advertised
- src/tdi_tabicl_cp.py: TabICL n_est=4 on PCA-256 of frozen chemprop_medium
  embeddings (same scaffold folds seed 7). Standalone OOF MCC: 2D6 0.2057
  @f0.22 (vs base 0.1495!), 3A4 0.4363 @f0.35 (vs base 0.4066).
- --ext 1 (AID1851 binaries as extra in-context rows) HURTS: 2D6 0.097,
  3A4 0.325 - qHTS population shift poisons ICL context; confirms the naive
  external-pool failure found in run_tdi_ext.py (w=0.3: 0.060/0.285,
  w=0.1: 0.058/0.314 - monotone in weight, direction: qHTS != TDI population).
- tdi_blend_nested.py (per-fold greedy w/ replacement, MCC-at-best-fraction
  score on held-out fold) over {base,emb,tab,extbase,tabcp,tabcpext}:
  nested-honest 2D6 0.1857 @f0.35, 3A4 0.4663 @f0.32 (v2 pool = these without
  tabcp: nested 0.121/0.4156). Macro 0.326 vs 0.268.
- Fraction v4 (posterior E[MCC] on nested pooled OOF): 2D6 E[MCC] 0.139@f0.08
  vs 0.203@f0.33 - the 2D6 optimum MOVED from ~0.08 to 0.25-0.35 because the
  blend fixes the ranking; shipped 0.33/0.36 (0.455/0.33 argmax left alone;
  0.33 chosen for 2D6 as it dominates 0.25 on both E and worst-case).
- NEW CANDIDATE cache/tdi_submission_v3.csv - official validator + row-for-row
  verified; test rates 2D6 0.331, 3A4 0.360 (v2 on board: 0.08/0.36).
- TDI expected blind: nested-blend macro +0.058 over v2's nested; with the
  observed OOF->blind blend transfer (v2: 0.294 nested -> 0.342 blind) the
  v3 expectation is ~0.39-0.40 macro MCC vs 0.342 on the board.
#### Decision guidance for Nov 3 (or earlier slot after 07:41 UTC Sep 24)
- Submit pair: cache/regression_final_cp7_submission.csv + cache/tdi_submission_v3.csv
  (needs Jackson's HF login + explicit go-ahead; latest valid submission counts).
- Both beat the current on-board files on honest nested OOF by +0.026 macro R2
  and +0.058 macro MCC respectively.
- Still open levers if time: 2nd seed for cpmed ridge is moot (ridge is
  deterministic); ft_ext 5th seed (+~0.001); cpchm+cpmed blend weight is
  already in; a 3rd checkpoint (adme_pretrain variant) would be the analogous
  next win. TDI: jeremy says his full pool has "more genuinely diverse
  candidates" - TabPFN on cpchm embeddings could add a 2D6 point.
