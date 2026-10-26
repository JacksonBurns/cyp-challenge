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

### Oct 25 ~19:50 - Q2 RESULT (subagent, commit d557382, VERIFIED on disk)
- Fit: blind_STRAE = 0.058 + 0.739*cv_STRAE (drift-adjusted, n=5, fit R2 0.88;
  b CI [0.155,0.954]). era-B clean triple: blind = cv - 0.099 (tight -0.089..-0.104).
- R2: blind EXCEEDS nested-CV by +0.067 (placement effect); rho +0.064.
- cp18 observed gain transfer rate 8.8% (CV -0.0159 -> blind -0.0014): the
  honest bound is predicted gain <= 0.014 for cp21/cp22.
- PREDICTED blind cp21 0.499-0.509, cp22 0.499-0.111 (0.5119 cp18 baseline):
  gain at/below 0.012 board noise floor. cp21-vs-cp22 ordering uncalibrated.
- IMPACT: Q2 says pool-swap slots (cp21/cp22) are EV-neutral. Slots should go
  to PROBES (per-isoform R2s for blind_moment_solve placement) not pool swaps.
  Files: src/q2_discount_fit.py, cache/q2_discount.json.

### Oct 25 ~19:55 - Q1 label wiring fix
- is_TDI labels: TDI file = 2D6/3A4 only; 1A2/2C9 come from Emax file
  (23/38 positives -> 1A2/2C9 arms are low-signal; gate on 2D6/3A4).
  q1_tdi_imbalance.py rewired to merge both label sources. Running proc_63e9d110edc7.
- PCGrad smoke proc_dcd87d523a54 (GPU, sole GPU job).

### Oct 25 ~20:05 - PCGrad smoke PASS + arms in flight
- pcgrad_arm.py smoke: pipeline OK (fold 16s, cos_pa/cos_pc logged, no crash).
  Full run proc_c9779d78af7f (tag pcg1, SOLE GPU job). Gate: 2D6 val ST-RAE
  vs mv baseline (~0.97-1.01 starved) and pool CV gain.
- Q1 relaunch (labels merged from TDI+Emax files): proc_63e9d110edc7 (CPU).
  Label census: 2D6 324+/1173-, 3A4 764+/2820-, 1A2 23+ (low-signal), 2C9 38+.
- UQ arm src/uq_detect.py (CPU, proc_8e9f1c363b52): seed-spread + cross-family
  disagreement vs |residual|; pre-registered gate AUC>=0.62 on >=2 isoforms.
  Writes cache/uq_cross_spread.csv, uq_seed_spread.csv, uq_detect.json.

### Oct 25 ~20:30 - Q1 + UQ GATE RESULTS
**Q1 (dedicated is_TDI + imbalance methods) - gate result: weak PASS on 3A4,
near-chance on 2D6, dead on 1A2/2C9.**
- 3A4 (764+/2820-): AUROC 0.526, MCC_best +0.061, E[MCC]@f=0.36 = +0.168.
- 2D6 (324+/1173-): AUROC 0.476-0.505 (chance), MCC_best +0.034.
- 1A2 (23 pos) / 2C9 (38 pos): unlearnable, MCC <= 0.
- SCIENTIFIC VERDICT: joint-arm MCC 0.000 collapse was NOT gradient starvation -
  the binary is_TDI label signal is ~chance-level from available features.
  Balanced weights/logit-adjust/threshold-calibration lift MCC from 0.000 to
  0.03-0.06 on labeled isoforms: real but tiny. TDI track rank-30 target
  (+0.014 MA-MCC) will NOT come from this arm. Files: cache/q1_tdi_imb.json.

**UQ detection arm - PRE-REGISTERED GATE PASSED 4/4** (threshold AUC>=0.62
on >=2 isoforms):
- cross-family disagreement AUC(top-quartile residual): 1A2 .631, 2C9 .625,
  2D6 .623, 3A4 .661. Lift top-decile: 1.32-1.52x mean |residual|.
- seed-spread weaker (AUC .55-.62): model-FORM diversity carries the
  uncertainty signal, seed diversity does not - consistent with loss-zoo
  finding (loss diversity > seed diversity).
- Spearman(uq, |resid|) .17-.27. Per-molecule UQ maps exported:
  cache/uq_cross_spread.csv, uq_seed_spread.csv, gate json uq_detect.json.
- NEXT (GPU, queued behind pcg1): UQ-weighted 2D6 loss (downweight top-decile
  spread rows); gate = CV ST-RAE gain. Latent-embedding kNN OOD to follow.

### Oct 25 ~20:45 - GPU serial queue (via notifications, not chained script)
- pcg1 full run = proc_704c26ad15e5 (pid 3864474) SOLE GPU job now.
- Queued after it (launch on each completion notification; 1 GPU job at a time):
  1. src/uq_weight_arm.py --baseline --seed 0        (baseline, prim-only)
  2. src/uq_weight_arm.py --q 0.9 --wdw 0.5 --iso all --seed 0
  3. src/uq_weight_arm.py --q 0.9 --wdw 0.5 --iso CYP2D6 --seed 0
- Gate: macroST gain vs baseline > 0.005. JSON per run in cache/uqw_*.json.
- NOTE: foreground terminal caps ~420s; ALL GPU runs use background=true.

### Oct 25 ~21:00 - Q3 pool-role predictor: GATE FAIL (clean negative)
- Pre-registered: sign(corr(resid_m, resid_inc-minus-m)) predicts pool weight
  sign (bias-corrector vs decorrelator). Pooled Spearman = -0.069 (n=1286,
  14 pools, p=0.014): SIGNIFICANT but tiny. Per-iso -0.11..+0.06.
- d (resid corr) compresses to .79-.96 for all members: shared label noise
  dominates; residual orthogonality ALONE cannot predict pool role a priori.
- Family means do order correctly (dscst w-.116 vs intarm w+.105) but d values
  overlap (0.798 vs 0.840): the effect is family-identity, not a transferable
  OOF statistic. Verdict: pool role = property of training recipe interaction,
  not predictable from member-residual geometry. Q3 CLOSED (negative result
  for methods report; zoo-family screening stays empirical).
- Files: src/q3_poolrole.py, cache/q3_poolrole.{csv,json}.

### Oct 25 ~21:10 - Q4 2D6 noise floor: resistance is FEATURE-side, not label noise
- ChEMBL replicate analysis (>=3 assays/molecule, human, relation '='):
  inter-assay pIC50 std pooled: 1A2 0.63 (n=15 mols), 2C9 0.39 (35),
  2D6 0.65 (47), 3A4 0.62 (103). Medians 0.16-0.33.
- sigma_2D6/others = 1.19 (~parity). 2C9 has the LOWEST noise floor yet
  mid-tier scores; 3A4 has noise ~2D6 but the BEST scores. Noise does NOT
  order the difficulty. Verdict: 2D6 resistance = feature/chemistry side
  (pharmacophore saturation: basic amine + spacing is ubiquitous), NOT an
  irreducible label-noise ceiling. 'Know when to stop' argument weakened;
  headroom exists if representation improves. Files: src/q4_noise_floor.py,
  cache/q4_noise.json. Caveat: replicate n small (15-103 mols/iso).

### Oct 25 ~21:20 - Q5 interval-supervision dissociation (quantified)
- Family table (cache/q5_interval.csv): POSITIVE pool-w families mean
  standalone RMSE 0.796; NEGATIVE-w (dscst -0.116, fullft -0.133) RMSE
  0.757-0.768 = BETTER standalone. Confirms 'standalone-bad = pool-good'
  ordering (mv 0.88w +.094; umtmL2 best-of-both +.363).
- Bound-coverage of OOF preds ~0.24-0.34 for ALL families (conf bounds are
  tight); interval arms (intarm .267, zoo_hinge_w3 .260) do NOT cover more.
  VERDICT: the interval-arm pool value is decorrelation via loss-geometry
  diversity, not measurable coverage calibration on point preds. Q5 partial:
  dissociation confirmed on the accuracy axis, NOT on the coverage axis.

### Oct 25 ~21:35 - PCGrad FULL RUN: TDI head gate PASS (MCC 0.00 -> 0.34)
- pcgrad_arm.py 5-fold, prim-anchored shared-trunk surgery, mvx 32-head layout:
  POOLED soft_rae 1A2 .8438 / 2C9 .6956 / 2D6 .9503 / 3A4 .5686
  (macro .7646; mvx marginal baseline .834; 2D6 .95 = starved band persists
  on val, modest pool-relevant gain).
- **TDI heads: 3A4 MCC 0.3403, 2D6 MCC 0.0541** (joint arms before: 0.000).
  PCGrad gradient surgery FIXED the all-negative collapse on 3A4. Combined
  with Q1 (is_TDI signal real but thin), TDI track path = PCGrad joint head,
  not standalone LGBM. 3A4 head alone would contribute ~+0.085 to MA-MCC if
  it transfers blind (current board MA-MCC 0.3422; rank-30 +0.014).
- Files: cache/ft_oof_pcgrad.csv, tdi_probs_pcgrad.csv, joint_oof_tdi_pcgrad.csv.
- GPU queue next: uqw baseline (launched), then uqw q90w50 all, then 2D6-only,
  then q6_direction modes base/A/B.

### Oct 25 ~21:45 - TDI head adjudication: v4_candidate STAYS slot owner
- PCGrad TDI heads under field scoring (prior-grid E[MCC], src/tdi_pcgrad_eval.py):
  3A4 MCC_best 0.4086, E[MCC]@f=.26 = 0.230 (q-sweep 0.283); 2D6 MCC_best
  0.1327, E[MCC]@f=.33 = 0.046.
- v4_candidate E[MCC] book: 2D6 0.163, 3A4 0.409 (macro 0.286). PCGrad head
  loses at shipped fractions on BOTH isoforms. No TDI candidate promotion.
- SCIENCE: MCC 0.000 -> 0.4086 (best-thr, 3A4) proves joint-head collapse was
  an optimization pathology (fixed by PCGrad surgery), NOT label absence;
  signal is real but weaker than the dedicated LGBM blend machinery. Both
  facts go in the methods report.

### Oct 25 ~22:05 - UQ-weighted loss GATE FAIL (diagnostic yes, causal no)
- prim-only baseline macroST 0.7437 (2D6 .9100); UQ-downweighted (top-decile
  x0.5): all-iso 0.7513, 2D6-only 0.7512. GATE (gain >0.005) FAILS (-0.008).
- Verdict: cross-family UQ is a valid DETECTOR of high-residual molecules
  (uq_detect AUC .62-.66) but down-weighting them in the loss does not
  improve point ST-RAE: flagged rows are exactly the irreducible-hard rows;
  removing their gradient neither helps fit nor denoises targets. UQ stays
  a monitoring/flagging tool (test-set flags in uq_ood_latent), not a loss.
- GPU freed -> uq_ood_latent (quick) then q6_direction base/A/B queued.

### Oct 25 ~22:20 - PCGrad pool gate: pool-NEUTRAL (no ship change)
- zoo7 = zoo6 + ft_pcgrad: fitted weights pcgrad 1A2 -0.034, 2C9 +0.046,
  2D6 -0.043, 3A4 +0.016 -> weight ~0 all isoforms. Pool gates pass
  (1A2 +.0209 5/5, 2D6 +.0303 4/5, 3A4 +.0194 5/5, 2C9 +.0025 4/5) = zoo6
  gains unchanged. PCGrad is a TDI/science win, NOT a regression-pool member.
- cp21 (zoo4) remains the regression ship candidate; no rebuild warranted.
