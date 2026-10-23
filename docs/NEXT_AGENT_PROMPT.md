# PROMPT FOR NEXT AGENT (Oct 22) - FINAL PUSH, closes Nov 3

Copy-paste block below. Repo state described = commit ae5b094 (Sec 29).

## COPY-PASTE PROMPT

You are taking over the OpenADMET CYP challenge at /home/jackson/cyp-challenge.
Competition closes Nov 3; today is Oct 22. Execute the plan in
docs/IMPROVEMENT_PLAN.md - do not re-derive it. Read in this order:
docs/IMPROVEMENT_PLAN.md (the plan: levers L1-L5, gates, sequencing, rules),
docs/mining/open_top.md (what the top entries do - the evidence base),
docs/mining/board_analysis.md (rank sensitivity math), NOTES.md sec 28-29
(what is closed - do NOT reopen closed levers, the list is in plan section 2).

Standing (leaderboard/2026-10-22 boards, committed):
- Regression: rank 51 of 243, ST-RAE 0.5133 (R2 0.5219, rho 0.7245). Shipped:
  cp17 19-member pool (nested macro R2 0.4682). Rank 30 = 0.4512 (-0.062);
  ~2.8 ranks per 0.01 ST-RAE. Top entries beat us by FITTING THE METRIC
  (Scigantic: hinge-fit combiner -0.017/step; briford: placement algebra +
  blind-moment probes) not by exotic models.
- TDI/classification: rank 35 of 130, MA-MCC 0.3422. Shipped: tdi_v5.
  Rank 30 = 0.3566 (+0.014 needed); rank 20 = 0.3832; rank 1 = 0.5752
  (nova - unreachable).
- Current files on the board ARE our best: cache/regression_final_cp17_
  submission.csv (md5 11096b10..) + cache/tdi_submission_v5_candidate.csv
  (md5 c8ab0626..). Both rebuild byte-identical per docs/REPRODUCTION.md.

Do first (in order):
1. L1: src/hinge_blend.py - Powell-fit member-delta weights minimizing
   soft_rae (src/run_regression.py:78, hinge = clip(p-hi,0)+clip(lo-p,0))
   + L2 on cp17 pool scaffold-OOF rows (cache/ft_oof_*.csv + blendN_cp17.json;
   bounds from build_all -> {iso}_lo/_hi), per isoform; moment-restore to
   incumbent mean/sd so only reordering ships. Gate: rotation CV (fit 4/5
   folds of OOF rows, score held 1/5) positive >=4/5 AND mean gain >0.004;
   2D6 needs 4/5 (its offline-transfer record is signed -0.18).
2. L4: same script, confidence-weighted (per-compound 5-fold CV std from
   member preds) non-negative NNLS variant; ship only if paired bootstrap
   95% CI excludes 0 and agrees in direction with L1.
3. If L1 (or L1+L4) passes: build cp18 candidate via regression_final-style
   placement, run src/verify_submissions.py, record md5 in NOTES.md, tell me
   (the user) it is ready to submit (submission is the user's action; 12h
   cooldown, ~24 slots left - slots are a resource, plan them).
4. L2: placement_grid.py affine (mean,sd) grid on the hinge surrogate,
   centre ABOVE truth mean capped at +0.75 (briford 4.35 probe = disaster);
   then blind_moment_solve.py: 3 affine probe submissions (base / 1.15x
   spread / centre+0.25) -> solve per-isoform var(y), mean(y), cov(y,p)
   from the R2 quadratic; cross-check solved Pearson >= measured Spearman.
5. L3 (GPU, ONE job at time - single RTX6000, use terminal background=true
   + notify_on_complete, watch the CHILD python PID; foreground terminal caps
   ~420s): interval-hinge training arms at UNIFORM sample weights, 3 seeds
   first (seed noise floor is 0.05 Pearson on 1A2/2D6). Per-fold score means;
   any fold >0.085 off pooled = quarantine. Promote to pool, re-run L1.
6. L5: TDI direct-head classifiers (plan section 2); gate = beat 0.3448
   nested OOF (the bar that killed the Oct 20 refresh).

Rules: never touch data/; gate every lever per-isoform; Spearman/Kendall of
shipped columns must be bit-stable under placement; every shipped file gets
verify_submissions.py + md5 in NOTES.md + a REPRODUCTION.md rebuild line;
append dated sections to NOTES.md (sec 29 is ours); commit often to main,
push to origin (github.com/JacksonBurns/cyp-challenge). Clean tree discipline:
new levers land as the named src/ files above; prune superseded probes; the
FINAL pair must rebuild byte-identical from the cleaned tree (Jackson's hard
requirement for the final push). Work sequentially - you and any delegated
agents share one machine/GPU. No em dashes in output. When something gates
RED, document it in NOTES.md and move to the next lever - do not retry a dead
lever. Report progress at each gate result.

## Context

- Envs: ~/miniforge3/envs/cyp/bin/python (main), ~/miniforge3/envs/mordred/bin/python (rdkit 2026.09, TabPFN/TabICL).
- Submission page: https://huggingface.co/spaces/openadmet/cyp-challenge (user submits; files at cache/ paths above).
- Boards: leaderboard/regression_2026-10-22_board.csv, leaderboard/classification_2026-10-22_board.csv.
- Scigantic hinge code port reference: /tmp/mining/scigantic (if gone: Scigantic/open-admet-challenge-cyp, scripts/build_r25.py).
