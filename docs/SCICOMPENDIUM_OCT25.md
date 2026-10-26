# CYP Challenge - Oct 25 science push: results compendium

Directive (Oct 25): PCGrad multi-objective balancing, imbalance learning
for TDI, uncertainty quantification (loss + latent + detection), Q2 CV-to-
blind discount fit, and the Q1-Q6 open questions. All executed Oct 25;
every gate pre-registered, results reported honestly including failures.

## Gate scorecard

| Arm | Gate | Result |
|---|---|---|
| Q2 discount fit | calibrate CV->blind | PASS: blind = cv - 0.099 (era-B), gain slope 0.66, R2 delta +0.067 |
| Q1 is_TDI dedicated + imbalance | MCC > 0 | PARTIAL: 3A4 +0.061 (E[MCC] .168), 2D6 +0.034, 1A2/2C9 dead |
| PCGrad (mvx 32-head) | 2D6 unstarved + pool gain | TDI PASS (3A4 MCC .000->.340), 2D6 not fixed, pool weight ~0 |
| UQ detection (cross-family) | AUC >= .62 on >=2 iso | PASS 4/4 (.623-.661), lift 1.32-1.52x |
| UQ loss reweighting | macroST gain > .005 | FAIL: 0.7513 vs 0.7437 ctl |
| UQ latent-embedding OOD | corr with residual | FAIL: ~0, 2D6 = -0.25 (inverse) |
| Q3 pool-role predictor | Spearman <= -0.5 | FAIL: -0.069 (significant, tiny) |
| Q4 2D6 noise floor | sigma ratio | DONE: 1.19 - resistance is feature-side, not label noise |
| Q5 interval supervision | coverage calibration | PARTIAL: dissociation on accuracy axis, NOT coverage |
| Q6 homolog direction | per-direction gain | RUNNING (base/A/B, GPU) |

## Headline findings

1. **The joint-head TDI collapse was an optimization pathology, not label
   absence.** Every plain joint arm gave is_TDI MCC 0.000 (all-negative).
   PCGrad gradient surgery (prim-anchored projections) lifted 3A4 to MCC
   0.3403 (best-threshold 0.4086). Dedicated LGBM + imbalance methods
   (balanced weights, logit adjustment, threshold calibration) also break
   0 (3A4 0.061). The label carries real signal; the multi-task optimizer
   was killing it. v4_candidate (E[MCC] 3A4 .409) remains the ship.
2. **CV-to-blind discount is measurable and tight on the level offset:**
   era-B boards score blind = CV - 0.099 (ST-RAE), and blind R2 EXCEEDS
   nested CV R2 by +0.067 (placement effect). Gain propagation is ~0.66
   (CI .16-.95); cp18's observed 8.8% transfer rate bounds expected cp21
   gain at <= 0.014, i.e. at/below the 0.012 board noise floor. Implication:
   pool-swap slots are EV-neutral; slots should buy probes/placement.
3. **Behavioral UQ detects issue-molecules; representation UQ and loss-side
   UQ do not work.** Cross-family OOF disagreement flags top-quartile
   residual molecules at AUC .62-.66 (decile lift 1.32-1.52x) on all four
   isoforms. Seed-spread is much weaker (model-FORM diversity carries the
   signal - same lesson as the loss zoo). Down-weighting flagged rows in
   training does not improve ST-RAE: flagged rows are irreducibly hard,
   not mislabeled. Frozen-encoder latent kNN distance is uninformative
   (corr ~0; 2D6 inverse -0.25: OOD-in-latent = MORE predictable).
4. **2D6 is not noise-limited.** Inter-assay pIC50 replicate sigma for 2D6
   (0.65) ~ 1A2/3A4 (0.62-0.63); 3A4 scores best with the same noise. 2D6
   resistance is representation-side (pharmacophore saturation), so
   headroom exists; the winning lever so far is homolog transfer
   (pt_homarm +0.165 pool weight), direction measured in Q6.
5. **Pool role is not predictable a priori.** Member-residual geometry
   (corr with incumbent residual) has tiny predictive power for fitted
   pool weights (Spearman -0.069, n=1286). dscst-as-bias-corrector is a
   recipe-identity effect, not a transferable statistic. Family screening
   stays empirical.
6. **Interval supervision's pool value is decorrelation, not calibration.**
   Positive-weight families have WORSE standalone RMSE (0.796 vs 0.757)
   - the standalone-bad/pool-good dissociation is real on the accuracy
   axis - but bound-coverage is identical across arms (0.24-0.34): the
   hinge/interval loss geometry decorrelates errors, it does not teach
   calibrated intervals (at least not measurably on point predictions).

## Ship-relevant decisions (Oct 25)
- cp21 (zoo4) remains regression ship candidate; zoo7 (cp21+pcgrad) gates
  = zoo6 unchanged, pcgrad pool weight ~0 -> no rebuild.
- TDI v4_candidate remains TDI slot owner.
- Q2 says: spend remaining slots on probes (per-isoform R2 -> placement
  solve via blind_moment_solve.py), not pool swaps.

## Artifacts
src/: q2_discount_fit.py, q1_tdi_imbalance.py, pcgrad_arm.py, uq_detect.py,
uq_weight_arm.py, uq_ood_latent.py, q3_poolrole.py, q4_noise_floor.py,
q5_interval.py, q6_direction.py, zoo7_gate.py, tdi_pcgrad_eval.py
cache/: q2_discount.json, q1_tdi_imb.json, uq_cross_spread.csv,
uq_seed_spread.csv, uq_detect.json, uqw_*.json, uq_ood_test.csv,
q3_poolrole.{csv,json}, q4_noise.json, q5_interval.csv, hinge_blend_zoo7.json,
ft_oof_pcgrad.csv, tdi_probs_pcgrad.csv, joint_oof_tdi_pcgrad.csv
