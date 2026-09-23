# CYP CHEMPROP / CHEMELEON EXPERIMENT PLAN (post-interim, before Nov 3 final)

Status: NOT STARTED. Written 2026-09-22 after interim submission freeze.
Do not run before the Sep 24 submission unless Jackson says otherwise; GPU is
shared with llama-server - check nvidia-smi and never kill running processes.

## Why both are good bets HERE (reasoning, don't lose this)

- Regression labels per isoform are only ~1.3-2.3k rows - small-data regime,
  which is exactly where pretrained encoders beat hand-featured GBMs.
- CheMeLeon leaderboard baseline = 0.834, but that is a *raw uncalibrated*
  submission (same as raw LGBM 0.893). Our thesis: CheMeLeon's advantage is
  RANKING quality; feed it through the same BLIND_MOMENTS placement that took
  our GBM blend from OOF 0.743 to expected ~0.5-0.62 blind. Nobody gets to a
  top-10 score (0.38-0.44) with one raw model - stacking is table stakes.
- jeremy (0.518 blind): frozen chemprop embeddings AS FEATURES for TabPFN
  beat fine-tuning; fine-tuning HURT. briford (0.438): 4x Chemprop D-MPNN
  ensemble + external data (ChEMBL 37 CYP, PubChem AID1851) + per-isoform
  affine placement. Two independent confirmations that D-MPNN adds signal on
  top of GBM+FP.

## Assets already on this box (verified 2026-09-22)

- `~/chemeleon_nazarov/chemeleon_mp.pt` (40 MB, weights_only-loadable) and
  `~/.chemprop/chemeleon_mp.pt` (same file).
- Conda envs ready: `chemeleon`, `chemprop-dev` (has chemprop+torch+lightning;
  cyp env has NO torch - do not pollute it, it only runs LightGBM/rdkit).
- Skill `chemeleon-fine-tuning` has the working code patterns (frozen MP +
  FFN head, x_d descriptor pass-through, checkpoint-cleanup pitfall, and the
  critical warning that generic 200+ RDKit descriptor banks HURT small-data
  CheMeLeon fits - our submission features stay LGBM-side only).
- Quadro RTX 6000 24GB; CheMeLeon full FT is ~2 min/fold, frozen-encoder
  extraction minutes for 6k compounds. Single heavy GPU task at a time.

## Experiment order (cheap -> expensive, each independently shippable)

1. **CheMeLeon frozen embeddings -> LGBM** (~half a day, lowest risk)
   - Extract MP embeddings (1024-d, pre-FFN) for all train + test compounds
     via frozen `BondMessagePassing` + `MeanAggregation`, batch_size 64.
   - Concatenate with existing X (or use standalone - test both) and rerun
     `sweep.py` scaffold CV. Ship only if scaffold-OOF blend MA-ST-RAE < 0.743
     AND per-isoform OOF Pearson does not regress on 2C9/3A4 (the two that
     carry the metric; 2D6 is placement-dominated).
   - Same features feed the TDI classifier - OOF MCC 0.41 (3A4) is the bar;
     2D6 0.15 has the most headroom.
2. **CheMeLeon full FT, scaffold CV, regression** (~half day)
   - Per-isoform: fine-tune whole MP on labeled rows of that isoform,
     5 folds by our existing scaffold_groups. Keep authors' recipe defaults;
     early stopping needs internal val split (10% of train fold).
   - Stack fold-OOF -> per-isoform OOF Pearson + ST-RAE. If a single FT model
     beats the 12-model GBM blend OOF, add it to the ensemble (weighted by
     OOF, or simple average after z-scoring per isoform).
3. **Chemprop D-MPNN regression** (~day; env `chemprop-dev`)
   - `chemprop train` per isoform, `--config` json, scaffold split matching
     our scaffold_groups if expressible, else their scaffold split + re-do
     OOF via cross-validation flags. 2-3 seeds per isoform.
   - briford-style ensemble; then same placement pipeline.
4. **External data pretraining/multitask** (the biggest lever per briford;
   save for last because it changes the training distribution)
   - ChEMBL 37 CYP activities + PubChem AID1851 (Veith qHTS, 1A2/2C9/2D6/3A4
     percent-activity, thresholded at 35%). Pretrain/fine-tune on external +
     challenge data jointly; validate that it does not poison calibration
     (external assays are not DRC-fitted; only ranking may transfer).
5. Stacking: Caruana-style greedy selection over {GBM-blend, CheMeLeon, D-MPNN}
   OOF preds per isoform, then placement, then `src/verify_submissions.py`.

## Non-negotiables (from section 7/8 gotchas)

- Any new model that uses label-derived NN features must self-exclude in CV.
- All CV via scaffold_groups from run_regression.py - random CV is inflated on
  this chemistractive test design.
- Judge final changes on calibrated blind expectations + the Sep 25 interim
  reveal, not raw OOF alone; the reveal gives per-isoform truth to recalibrate
  OOF_TO_BLIND inflations.
- New deps go in a fresh conda env or the existing chemeleon/chemprop-dev
  envs; NEVER pip install into `cyp`.
- Re-run src/verify_submissions.py on every candidate file before showing it
  to Jackson; NEVER submit without his explicit go-ahead.
