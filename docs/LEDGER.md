# CYP challenge - autonomous science ledger (Oct 23, 12:30 EDT)

Durable work-state for context compaction. Update this file as work completes.
Deadline: Nov 1. Board slots: 12h cooldown. Current standing: cp18 = 0.5119.

## Standing facts
- Repo /home/jackson/cyp-challenge; main env ~/miniforge3/envs/cyp/bin/python (CPU
  blending), ~/miniforge3/envs/chemeleon/bin/python (GPU chemprop, NO lightgbm -
  never import run_regression there).
- ONE heavy GPU job at a time (RTX6000 shared with desktop). Long jobs:
  terminal(background=True) + notify_on_complete.
- Gate = rotation-CV in src/hinge_blend.py (pool cp18i = cp17 families +
  ft_intarm). Ship-set = 5/5-clean passes only.
- L3 intarm seed-averaged: standalone WORSE (0.6886 vs cpmed 0.6443) but flips
  2C9/2D6 gates -> decorrelation member. Weights 0.046-0.168.
- cp19 built+validated (md5 ca2aa784...), NOT submitted. Probes B/C built in
  cache/probes/. L2 blind-moment solve = src/blind_moment_solve.py (needs
  per-isoform R2 rows from user's submissions: cp19, B, C).
- USER BOARD ACTIONS QUEUED: submit cp19 -> probe B -> probe C, send me
  per-isoform ST-RAE numbers after each. Cooldown expired ~8:52 AM EDT.

## Theory driving current work
Pool's binding constraint is DECORRELATION, not accuracy. Loss-axis diversity >
seed-axis diversity (intarm evidence). Queue:

1. LOSS ZOO (GPU, ~4h serial): same union-table trainer, new primary loss:
   - mae: plain L1 z-space (loss-diversity control)            [arm tag zoo_mae]
   - rank: squared hinge outside interval + pairwise logistic
     ranking on labeled pairs within each isoform (metric-native:
     zero cost inside + ranking pressure inside)                [zoo_rank]
   - quantile: 3 pinball heads (10/50/90) per isoform as extra cols,
     primary = median head (native interval-estimation task)   [zoo_qtl]
   - auxw sweep {0.0, 1.0, 3.0} of intarm loss (AUX_W knob, effect size of
     aux regularization on the gate)                            [zoo_auxw0/1/3]
   Gate: add family to cp18i pool, hinge_blend --pool, marginal CV gain.
   One seed for screening; 2-3 seeds ONLY for survivors (seed floor known:
   intarm standalone pooled 0.436/0.501/~0.44 across seeds).
2. CHEMBL HOMOLOG AUX HEADS (CPU download then GPU arm): add 2C19/2C8/2B6/2A6
   ChEMBL IC50 heads to the ext table (same pipeline as 1A2/2C9/2D6/3A4 aux).
   Shared trunk on 8 homolog heads = real transfer, unexplored diversity axis.
   Target IDs: 1A2=CHEMBL3356 2C9=CHEMBL3397 2D6=CHEMBL289 3A4=CHEMBL340;
   query API for 2C19/2C8/2B6/2A6 (human, single protein target).
3. PER-ROW CONFIDENCE BLENDING (CPU): interval width = per-compound
   uncertainty -> per-row weights in the hinge-fit objective + per-row blend
   weights. stir_bar's unfinished lesson. Test via L1 gate on cp18i.
4. TRANSDUCTIVE PRETRAINING (GPU, last): masked-atom PT on 750 test SMILES +
   unlabeled external, init encoder from PT, fine-tune. Only if time.
5. Post-deadline: methods report (interval supervision + honest-eval protocol +
   measured CV->blind transfer discount). Outlined in NOTES.md.

## Done (do not redo)
- L1 hinge combiner (hinge_blend.py, --pool/--ship/--incumbent/--out CLI;
  smoke artifacts excluded from family globs).
- intarm seeds 0/1/2 trained (ft_oof_intarm{,_s1,_s2}.csv + test preds).
- cp18i seed-avg gate: ALL 4 PASS (1A2 +0.026 5/5, 2C9 +0.0046 4/5,
  2D6 +0.0135 4/5, 3A4 +0.0188 5/5). 2C9/2D6 promoted only w/ probe evidence.
- cp19 = 1A2/3A4 refit, 2C9/2D6 = cp18 bytes. Validator PASS. REPRODUCTION.md
  documents rebuild. NOTES sec 30/30b/30c/30d.
- L2 placement grid src/placement_grid.py (sd k in [0.6rho,1.1], centre cap);
  probes B/C built; blind_moment_solve.py --emit-probes works.

## Log
- 12:30 ledger created; loss-zoo trainer being written; ChEMBL homolog
  download queued on CPU.
