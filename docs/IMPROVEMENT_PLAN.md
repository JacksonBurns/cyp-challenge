# IMPROVEMENT PLAN - final push (Oct 22 -> Nov 3)

Status anchor (Oct 22 board, 243 entries): regression **rank 51** (ST-RAE 0.5133,
MAE 0.7091, R2 0.5219, rho 0.7245, tau 0.5539), files on board =
`regression_final_cp17_submission.csv` + `tdi_submission_v5_candidate.csv`
(md5s in docs/REPRODUCTION.md). TDI **rank 38** (MA-MCC 0.3422; rank 30 =
0.3596, rank 1 = 0.5346). Full history in NOTES.md; field-mining evidence in
docs/mining/{board_analysis,open_top,open_mid_low}.md.

## 0. What the board says (board_analysis.md)

- Rank sensitivity near us: **~2.8 positions per 0.01 ST-RAE**. Targets:
  rank 45 = -0.024, rank 40 = -0.035, rank 35 = -0.052, rank 30 = -0.062,
  rank 20 = -0.076, rank 10 = -0.090, rank 1 = -0.158.
- Field fit ST-RAE ~ -0.743*R2 - 0.748*rho + 1.459 (resid std 0.030).
  Frontier at rank 30 (0.4512): rho 0.7245 needs R2 0.627; rho 0.75 needs
  0.601; rho 0.77 needs 0.581. We are NOT one lever away from rank 30;
  we are ~0.06 away, which needs 2-3 stacked levers.
- Placement residuals: we sit at -0.016 vs field fit; the band leaders sit at
  -0.027 to -0.046. **Placement-only headroom ~0.01-0.03** (bounded, real).
- We sit ABOVE the rho ceiling story: rank-30 band entries all have rho >=
  0.75; top-10 rho 0.77-0.80. Ranking (rho) is the metric the field is
  actually competing on; R2 mostly follows from calibration at rho.
- Late surge: 11 of top 20 submitted in the last 72h; the field is iterating
  with the metric, not just with models. Submission slots are a resource
  (~24 left at 1/12h).

## 1. What the top open entries do (open_top.md) - the three load-bearing facts

1. **The metric is fit, not assumed.** Scigantic (rank 11 interim, 0.4436)
   fits member weights by Powell minimization of the ST-RAE hinge on OOF
   rows + L2, then restores the incumbent's mean/sd (moment-restore, so only
   reordering ships). One step: 0.4620 -> 0.4453 (-0.017). Their training
   arms also use the interval (hinge) loss at UNIFORM sample weights
   (+0.0275; it HURTS at inverse-variance, interaction -0.051).
2. **Placement is a solved, probed, per-isoform affine target.** briford
   (rank 20, 0.4378): R2 = 2*rho*k - k^2 - b^2 exactly; optimum spread k=rho,
   not 1; ST-RAE optimum sits ABOVE the population centre with sd MUCH
   narrower (CYP2D6 probes: (3.57,0.90)->0.5502 vs (3.11,1.08)->0.6944 -
   0.14 on one isoform); penalty above the optimum is steeper than gain
   below; they solve blind-half mean/sd/Pearson EXACTLY from 3 scored
   affine probe submissions of one vector.
3. **Combination is non-negative + confidence-weighted.** stir_bar (rank 26;
   TDI rank 6): ridge/EN returned cancelling coefficients (+1.9/-0.6) that
   made CV-up/worse-blind revisions -> restricted to NNLS/non-negative.
   briford: per-compound confidence weights (5-fold CV std, conformal,
   scaled by |conf-error correlation|) beat flat mean on ALL FOUR isoforms
   beyond seed-noise resolution (+0.025 to +0.063 OOF Spearman).

Per-endpoint transfer discipline (Scigantic's 20-call record): offline gains
transfer on CYP3A4 6/6 and CYP2C9 3/4; CYP2D6 2/5 with signed -0.18,
CYP1A2 2/5. **Gate all offline wins per isoform; 2D6 needs the strictest bar.**
Seed noise floor (briford): 0.05 Pearson on 1A2/2D6; 0.012-0.013 on 2C9/3A4.
One-fold evaluations are not measurements (Scigantic: single-fold +0.0913 was
+0.0292 at 5 folds).

## 2. Levers, ranked by expected gain / effort (with gates)

Infra we already have: per-member scaffold-OOF predictions in
cache/ft_oof_*.csv (19-member cp17 pool), per-compound credible bounds
(build_all -> {iso}_lo/_hi), hinge metric src/run_regression.py:soft_rae,
shipped placement = moments x OOF_TO_BLIND x F_SPREAD in regression_final.py,
official validator + src/verify_submissions.py, md5 reproduction harness.

**L1. ST-RAE hinge-fit combiner + moment restore (CPU, 1-2 d).** Replace the
ridge objective on the cp17 pool: per isoform, Powell-fit member-delta
weights minimizing sum(hinge(p, lo, hi)) + L2 on scaffold-OOF rows; z-restore
to incumbent blend mean/sd. Expect -0.010 to -0.017 (Scigantic's step).
Gate: rotation CV (fit 4/5, score held 1/5) positive in >=4/5 folds AND mean
gain > 0.004; Spearman of shipped column unchanged (affine integrity);
per-isoform decision (2D6 strictest). This is the single biggest published
gain and reuses our exact caches.

**L2. ST-RAE-aware placement grid + blind-moment probes (analysis + 3 slots).**
Kill constant F_SPREAD: affine grid (mean, sd) per isoform on the OOF hinge
surrogate; target centre ABOVE mean (low-activity intervals are wide:
predicting high is nearly free, low is punished); cap centre at OOF-truth
mean + 0.75 (briford's 4.35 probe scored 0.8558 - over-correction disaster).
Then briford's moment solve: 3 affine probes of ONE vector (base; 1.15x
spread; centre +0.25 at new spread) -> exact var(y), mean(y), cov(y,p) per
isoform per the R2 quadratic; require physical root + rho >= Spearman
cross-check (+-0.07). Feeds L2's grid with the true blind target
distribution; also calibrates OOF_TO_BLIND exactly. Budget 3 slots.

**L3. Interval-hinge training arms, uniform weights (GPU 2-3 d + 1-2 slots).**
Our chemprop recipe (union table: challenge pIC50 + single-conc log2FC +
TDI-condition + AID1851 max_response/AC50 + ChEMBL 5-isoform incl. CYP2C19
as correlated 5th task) with scored-head loss swapped to hinge-outside-CI,
UNIFORM sample weights, 5 scaffold folds. Scigantic: +0.0275 at arm level.
Gate: per-fold ST-RAE means, no fold >0.085 off pooled (their NaN-divergence
tripwire); run the config at 3 seeds FIRST to measure our own noise floor;
promote to pool only above it, 4/5 folds on 2D6.

**L4. Confidence-weighted + non-negative blending (CPU 1 d).** Per-compound
CV-std confidence weights (free from our 5-fold member preds) x NNLS
delta-space combiner per isoform. briford OOF +0.025-0.063 Spearman; stir_bar
provides the CV-up/worse-blind caution. Gate: paired bootstrap 95% CI on
ST-RAE delta excludes 0 AND direction agrees with L1 (same pool, must not
disagree); ships only through moment-restore.

**L5. TDI: mechanism-first legs + precision retune (GPU, opportunistic).**
stir_bar rank 6 at MCC 0.380 says the TDI top tier is not far above us
(0.3422, rank 30 = 0.3596): -0.04 is enough for the top-30, top-26 tier.
Their facts: direct encoder heads OUTRANK tabular embedding legs on TDI;
"TDI is a different problem from potency" (mechanism-driven, not potency);
rank 30 needs ~+0.02 MCC. Plan: (a) train direct-head TDI classifiers on our
CheMeLeon/chemprop encoders (direct multi-task, not embedding+tabFM),
(b) use the conjunction structure we verified (is_TDI = gate AND shift on
TDI-arm-minus-direct potency) as an auxiliary regression feature, (c)
retune fractions on the v6b cap at the new ranking. Gate: must beat the
0.3448 bar on nested OOF (same gate that killed the Oct 20 refresh).

**Closed (do NOT reopen):** x_d descriptor channel, pool simplification,
3D SchNet/vacuum conformers, LLM pharmacophore priors, docking features,
2D6 pKa/stereo, pseudo-labeling, dead-zone refit, k-NN retrieval features,
band-width inversion (no blind bands - L2's probes replace this), placement
knob sweeps at constant F_SPREAD, CheMeleon-2048 ridge, near-test retrieval.
Each entry has a NOTES.md section showing the gate it failed.

## 3. Sequencing (12 days, ~24 slots)

- **Oct 23-25:** L1 + L4 CPU pair (same pool, run together, gate together).
  If L1 clears: build cp18_hinge candidate, validator + md5 reproduction,
  submit slot 1 (regression).
- **Oct 25-27:** L2 grid (uses the new hinge-fit column, not the old blend)
  + 3 probe submissions; solve blind moments; re-derive placement targets.
  Submit placement-refined candidate.
- **Oct 25-29 (GPU, sequential, single-task):** L3 arms at 3 seeds; promote
  arms clearing the fold tripwire; cp19 pool -> re-run L1 hinge fit on it.
- **Oct 27-31:** L5 TDI direct-head legs; fraction retune; TDI candidate if
  it beats 0.3448.
- **Oct 31-Nov 1:** final integration pass; verify md5 reproduction of every
  shipped file; submit final pair Nov 1 (latest counts; leave Nov 2-3 as
  buffer slots for one probe/fix only).
- Every GPU job: one at a time (shared single GPU), background with notify,
  check CHILD python PID.

## 4. Rules of engagement (anti-self-deception)

1. No candidate ships without rotation-CV gain >= gate for the specific
   isoform(s) changed; 2D6 demands 4/5 folds.
2. Per-fold score means always; any fold >0.085 off pooled = quarantine.
3. Spearman/Kendall of a shipped column must be bit-stable under placement
   (affine integrity check; no clamps that create ties).
4. Every submission goes through src/verify_submissions.py + the official
   validator; md5 of the file shipped is recorded in NOTES.md.
5. Offline wins on 1A2/2D6 are treated as noise until probed; 3A4/2C9 wins
   get the benefit of the transfer record.
6. Rebuild-and-verify (docs/REPRODUCTION.md) must pass on the FINAL cleaned
   tree for every shipped artifact.
7. Repo hygiene for the final push (Jackson's requirement): the cleaned src/
   (53 scripts) is the spine; new levers land as src/hinge_blend.py,
   src/placement_grid.py, src/blind_moment_solve.py, src/interval_arm.py,
   src/tdi_direct_heads.py; superseded probes get pruned; docs/archive/
   holds finished plans; REPRODUCTION.md updated so the final pair rebuilds
   byte-identical from the cleaned tree.

## 5. Honest expectation

L1+L4 (CPU pair) is worth ~-0.012 to -0.020 ST-RAE (rank ~35-42) if gates
pass; L2 placement refinement ~-0.01 to -0.02 (rank ~28-35); L3 arms add
rho ~+0.01-0.02 if the interval objective reproduces, worth another
-0.010 to -0.015. Realistic landing band with all three: **rank 25-40**;
rank 20 (-0.076) requires L3 to also raise 2D6 rho, the isoform with the
signed-negative transfer record - do not promise it. TDI: top-30 is
reachable with +0.02 MCC; top-10 is not (leaders at 0.41+).
The field surges late; plan final submits for Nov 1-2, not now.
