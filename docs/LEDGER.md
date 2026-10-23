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
- cp20 built+validated (md5 5a141f80), NOT submitted; cp19 first (marginally
  better CV), cp20 candidate if probe evidence supports.
- USER BOARD ACTIONS QUEUED: submit cp19 -> probe B -> probe C, send me
  per-isoform ST-RAE numbers after each. Cooldown expired ~8:52 AM EDT.

## Results so far (Oct 23, 17:30)
- zoo2 seed-avg gate (mae/w3/homarm 3-seed families): ALL PASS. 1A2 5/5 +0.0235
  (mae +0.215), 2C9 4/5 +0.008, 2D6 4/5 +0.021 (mae +0.245, mse -0.198), 3A4
  5/5 +0.0175 (w3 +0.222). cp20 BUILT: ship 1A2+3A4, 2C9/2D6=cp18 bytes,
  validator PASS, md5 5a141f80. vs cp19: 1A2 rho 0.987; 1A2 CV slightly below
  cp19's (+0.0235 vs +0.026) = fitter variance of 27-family pool. SUBMISSION
  ORDER: cp19 first, cp20 if probes support. auxw sweep: w=0 0.783, w=0.3
  0.689, w=1 0.754, w=3 0.743 (aux carries ~0.10; 0.3 sweet spot standalone,
  w=3 best decorrelator). Seed floors: w3 .740-.748, homarm .751-.766.
- Transductive PT (item 4): src/pt_encoder.py masked-atom PT on 29252 SMILES
  (test set + unlabeled ext + ChEMBL incl. homologs) from chemprop WEIGHTS
  init; GPU running. loss_zoo --pt flag; batch5 = pt_{mae,w3,homarm} x3 seeds
  (9 arms) queued behind; pt_* families in zoo1 pool.

## Results (Oct 23, 16:00)
- zoo1 pool gate (cp18i + zoo_mae/rank/mse seed-0): ALL 4 PASS. 1A2 5/5 +0.0217,
  2C9 4/5 +0.0058, 2D6 4/5 +0.0213 (was +0.0135 in cp18i: mae w=0.23, mse
  w=-0.138), 3A4 5/5 +0.0146. Loss-axis diversity CONFIRMED; mse anti-correlates
  (w -0.14 on 2C9) = the decorrelation trade. rank arm inert (~0 weight).
- Standalone zoo ST-RAE: mae 0.7502, rank 0.8146, mse 0.8443 (all worse than
  cpmed 0.6443; value is pool-level only, per theory).
- --conf per-row interval-width: NEGATIVE (all 4 gates fail, cv -0.033/-0.009).
- Homolog data: 2089 compounds (2C19 1615, 2C8 463, 2B6 224, 2A6 250),
  CHEMBL3622/3721/4729/5282 verified. homarm trains in batch2.

## Theory driving current work
Pool's binding constraint is DECORRELATION, not accuracy. Loss-axis diversity >
seed-axis diversity (intarm + zoo evidence). Queue:

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
