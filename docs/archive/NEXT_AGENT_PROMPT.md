# PROMPT FOR NEXT AGENT (Oct 15) - Descriptor-augmented Chemprop backbone

You are continuing Jackson's CYP competition work in ~/cyp-challenge. Execute
docs/NEXT_ITER_PLAN.md top-to-bottom. It is the single source of truth for this
iteration. Read NOTES.md sections 20+ and docs/RANK_UP_PLAN.md (for the dead-
lever list) FIRST, then NEXT_ITER_PLAN.md.

Context in one line: we are rank 63 regression (ST-RAE 0.5133, R2 0.5219) and
rank 38 TDI (MA-MCC 0.3422). The rank limiter on regression is R2, not
placement. The plan is a MODEL-ACCURACY iteration built on Jackson's two
observations: (O1) simplify the 19-member pool to ~3-5 strong members, and
(O2) feed mordred molecular descriptors into the chemprop backbone as x_d.

Run the phases in order, stopping at every GATE and reporting to Jackson:
  - Phase 0 (CPU, no GPU): build the mordred descriptor matrix
    (cache/emb_mordred_all.parquet, anti-leak NaN policy) + a tabular ridge
    probe (embedding vs embedding+mordred vs mordred-only). Gate 0.
  - Phase 1 (GPU, one job at a time): chemprop D-MPNN 4 variants
    {V1 control, V2 aux, V3 x_d, V4 aux+x_d} on the SAME scaffold folds +
    ft_ext external-aux table; winner also on Butina folds. Gate 1.
  - Phase 2 (CPU): simplified pool (<=5 members, each must pass LOFO /
    paranoid-merge), greedy nested blend vs cp17 (0.4682), placement UNCHANGED.
    Gate 2.
  - Phase 3: build cache/regression_final_cp18_submission.csv ONLY if Gate 2
    passed; verify_submissions.py + official validators; md5; commit+push;
    report to Jackson. DO NOT submit.
  - Phase 7 (TDI, only if time remains and regression is done): x_d into the
    TDI backbone, must beat tdi_v5 (0.3422) nested to ship.

Hard rules (non-negotiable, see NEXT_ITER_PLAN.md sec 9):
  - NEVER submit without Jackson.
  - One GPU job at a time; background long runs; never touch llama-server;
    never kill a running process without permission.
  - New conda env `mordred` for the descriptor step (never system python).
  - verify_submissions.py + official validators before showing ANY file.
  - Family-block / paranoid-merge audit before any member enters the pool.
  - Anti-leak: external data = aux/ranking only, never placement; descriptor
    stats fit on TRAIN only; label separation in the aux table.
  - Log honest numbers in NOTES.md (new section 23+) and commit+push at each
    gate.
  - A no-op (gates fail in sequence -> "cp17 stands, no resubmit") is a valid,
    logged result. Do not force a submission.

Report to Jackson at each gate with the actual numbers, not summaries of intent.
