# LEDGER2 - Oct 25 final-week science push (compaction anchor)

## STANDING DIRECTIVE (user, Oct 25 - authoritative)
Dedicate final CYP efforts in the "CYP Challenge" Telegram channel. User acts as
scientific collaborator. Execute ALL of these autonomously until done, then report:
1. **Q2 (do first, CPU):** fit the CV->blind ST-RAE discount from shipped-model
   history (board scores in leaderboard/*.csv vs CV macro in NOTES.md/JSON).
   Deliverable: per-isoform (or global) discount constant + CI; use it to rank
   cp21/cp22 offline without slots.
2. **Q1 (GPU):** dedicated is_TDI classifier - imbalance-learning methods
   (class weights, focal loss, logit adjustment, threshold calibration,
   SMOTE-free recompute). Gate: OOF MCC > 0 (prior joint arms = 0.000
   all-negative). If PASS -> feeds TDI track (board 0.3422, rank30=0.3566).
3. **PCGrad** (multi-objective gradient surgery) on mv/mvx multi-head trainer:
   the mv arms starve 2D6 (0.97-1.01 ST-RAE) under auxw=3; cls-weight 0.3 did
   NOT fix it. PCGrad projects conflicting task gradients. Gate: 2D6 improves
   AND pool CV gain positive.
4. **UQ arm:** per-molecule uncertainty (5-fold CV pred std + latent-embedding
   distance-to-train/OOD score) to DETECT issue-causing molecules; optionally
   UQ-weighted loss. Deliverable: ranked issue-molecule list + correlation of
   UQ vs OOF error (Spearman; target rho > 0.3).
5. Keep probe chain (cp21 submit -> probe B -> probe C) as leaderboard critical
   path; user handles submissions (12h cooldown, ~24 slots).

## GPU rules (memory-backed)
ONE GPU (RTX6000 24GB shared w/ desktop). ONE heavy job at a time. Use
terminal(background=true, notify_on_complete=true) EXPLICITLY - foreground
caps ~420s and SIGTERMs children. Watch CHILD python PID, not wrapper.
Subagents: max 1 at a time, blocking, shared GPU. Memory cgroup OOM: avoid
2048-dim embeddings without PCA.

## Context state
- Ship lineage: cp17 on board; cp21 primary candidate (zoo4 pool), cp22 reserve
  (zoo6 first all-4-PASS). Probes B/C built (cache/probes/).
- Closed avenues (do NOT reopen): rank heads, masked-atom PT, pKa/stereo,
  pseudo-labeling, is_TDI BCE-in-joint (MCC 0.00), per-row confidence blend.
- Key files: src/loss_zoo.py (multi-head trainer, --mv/--tdic/--emx/--clsw),
  src/hinge_blend.py (L1 combiner), src/blind_moment_solve.py, src/run_regression.py.
- Env: ~/miniforge3/envs/cyp/bin/python. Data dir: NEVER touch data/.
- Commit after every result. NOTES.md sections 30-34 = Oct 23-25 history.

## Results log (append below)

## Results log

### Oct 25 ~03:00 - execution started
- Q2 (CV->blind discount fit): DELEGATED to subagent (CPU-only, per GPU rule).
  Writes src/q1... no: src/q2_discount_fit.py + cache/q2_discount.json.
- Q1 (TDI imbalance): src/q1_tdi_imbalance.py - dedicated per-iso is_TDI LGBM,
  scaffold-5CV, arms m0 base / m1 balanced / m2 logit-adjust, MCC-best-thresh +
  E[MCC] at shipped fractions w/ board prior. LABEL INVENTORY (new): is_TDI
  positives 2D6 324/1497 (21.6%), 3A4 764/3584 (21.3%), 1A2 23/1413 (1.6%),
  2C9 38/1285 (3.0%) = the 1A2/2C9 MCC-0 collapse is partly extreme imbalance;
  2D6/3A4 imbalance is mild -> imbalance methods target is 1A2/2C9, and the
  2D6/3A4 joint-collapse points to training pathology, not imbalance. Running.
- PCGrad: src/pcgrad_arm.py - manual trainer, task groups prim(hinge)/aux(MSE)/
  cls(BCE), shared-FFN gradient surgery (anchor=prim, project aux+cls when
  dot<0), per-epoch macro soft_rae val ES, fold-avg test preds, TDI head MCC
  export. Diagnostics: cos(prim,aux), cos(prim,cls) per batch = the conflict
  evidence Q-PCGrad asked for. SMOKE running on GPU (1 job).
  Control: --nopcgrad plain-sum. Gate: 2D6 standalone < mv (0.97) AND macro
  <=0.84; then hinge_blend pool CV + 3 seeds.
- GPU: smoke test only; full 5-fold pcgrad + control queued after.

### Q2 (Oct 25) - CV->blind ST-RAE discount fit (src/q2_discount_fit.py, cache/q2_discount.json)
Five scored points paired CV-side (hinge/soft_rae surrogate on the exact SHIPPED
columns, raw family OOF + assay bounds; R2 = documented nested-honest) with blind
board scores. Boards stratified: era A = Sep23 interim+final, era B = Sep26/Oct22
(+Oct23 cp18 re-score 0.5119, NOTES 30b; Oct3 = cp7 repeat, board noise 0.012).
Era-A blind scores drift-adjusted to era-B scale (paired field +0.073 R2 /
-0.058 ST-RAE / +0.017 rho, NOTES 14); unadjusted sensitivity reported.

CV ship columns (macro over 4 iso): cp17-blend 0.6172, cp18 0.6013,
cp21 0.5969 (1A2 .6734+3A4 .3694 zoo4-fit, 2C9/2D6 cp18 bytes), cp22 0.5961
(1A2 .6702+3A4 .3694 zoo6-fit). cp21/22 = CV-noise twins (delta 0.0008).

Fits (blind = a + b*cv, bootstrap over points 10k):
- ST-RAE adjusted 5pts: a=0.058 b=0.739 (b CI [0.155,0.954], fit R2 0.88)
  -> pred cp21 0.499 [0.383,0.503], cp22 0.499.
- ST-RAE unadjusted 5pts: a=-0.238 b=1.228 (fit R2 0.986) -> cp21 0.494 [0.339,0.445].
- era-B only (n=3, the same-blind triple): blind - cv = -0.099 ST-RAE (tight:
  -0.089..-0.104); gain slope b=0.66 [0.16,0.95]. cp21 point 0.498 [0.492,0.512],
  cp22 0.498. Anchor propagation: from cp17 0.5133 -0.66*0.0203 -> 0.500; from
  cp18 0.5119 -0.66*0.0044 -> 0.509; at cp18's observed 8.8-13% transfer rate -> 0.511.
- R2: blind - cv (nested) = +0.067 era-B (NEGATIVE discount, placement; k~1.05
  history). Adjusted line pred cp21 0.60 [0.51,0.70] - EXTRAPOLATION, do not trust.
- rho: blind - cv = +0.064 era-B; line pred cp21 0.731 [0.721,0.742] (fit R2 0.96).

DECISION READ: predicted blind MA-ST-RAE cp21 0.499-0.509, cp22 0.499-0.511 vs
cp18 on board 0.5119. Expected gain 0.001-0.014 = at or below board-side noise
(0.012 from the cp7 same-file rescore). The discount is a LEVEL offset of
-0.099 ST-RAE (blind set is chemistractively enriched, easier than CV rows),
NOT a multiplicative gain factor: era-B gain slope 0.66 upper CI 0.95 excludes
the ~0.13x rate implied by the single cp18 event. Calibrated: absolute blind
level (+/-0.01). NOT calibrated: sub-0.013 deltas, cp21-vs-cp22 ordering
(CV twins), R2 transfer line (n=4, extrapolated CV range, nested-vs-placed
metric mismatch).
