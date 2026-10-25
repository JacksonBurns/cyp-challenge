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
