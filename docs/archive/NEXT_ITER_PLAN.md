# NEXT ITERATION PLAN (Oct 15) - Descriptor-augmented Chemprop backbone

Replaces RANK_UP_PLAN (that program is exhausted: 1 win / 4 dead levers, all
closed) and supersedes the stale NEXT_AGENT_PROMPT (which still pointed at the
finished FINAL_PUSH_PLAN / CYP-UMTM spec).

Owner: Jackson. Execution: a fresh agent session runs this top-to-bottom,
reporting at every gate. NEVER submit without Jackson. This file is the single
source of truth for the next model iteration.

---

## 0. WHERE WE STAND (the diagnosis that shapes the plan)

Live leaderboard (Oct 14 23:06 UTC submission, cp17 / tdi_v5):

  REGRESSION  rank 63   ST-RAE 0.5133   R2 0.5219   Spearman 0.7245
  TDI         rank 38   MA-MCC 0.3422   Precision 0.4329   Recall 0.5013

The binding constraint is R2, NOT placement.
  - Our placement is already strong for our R2: nested 0.4682 -> blind 0.5219
    (ratio ~1.11). We are not leaving rank on the table via the placement knob.
  - briford (supercowpowers blog, reg rank 15) runs ST-RAE 0.4378 at R2 0.6181.
  - jeremy (reg rank 66) runs R2 0.6024 but ST-RAE 0.5185: high accuracy,
    placement that under-expresses it. We are AHEAD of jeremy on placement and
    BEHIND on R2.

Conclusion: the next iteration is a MODEL-ACCURACY problem (backbone + features),
not a blend-size or placement problem. This lines up exactly with Jackson's two
observations below. Target: lift nested R2 from 0.4682 toward 0.50-0.54, which at
our placement ratio maps to blind R2 ~0.55-0.59, i.e. reg rank 25-45.

## 1. THE TWO OBSERVATIONS (Jackson, Oct 15)

(O1) SIMPLIFY. The supercowpowers / briford entry at ~0.4 ST-RAE is only a
    Chemprop ensemble with auxiliary tasks + post-processing. Implication: our
    19-member pool is over-diversified. The heavy lifting is a SMALL number of
    STRONG Chemprop D-MPNNs (each with external aux heads), not 19 weakly-
    decorrelated members. Cut the pool to ~3-5 strong, genuinely-decorrelated
    members.

(O2) DESCRIPTORS INTO THE BACKBONE. Concatenate hand-computed molecular
    descriptors (mordred) to the learned graph representation by passing them as
    chemprop's dense features (x_d / the Data `x` field). This is a NEW input
    channel to the PRIMARY model, not a separate pool member - so it can raise
    the backbone's own R2 instead of just adding a weak redundant member (the
    failure mode that killed our qHTS-DA and SMILES pool members).

## 2. STRATEGY IN ONE PARAGRAPH

Build a clean, small, strong regression stack: 2-4 Chemprop D-MPNNs, each with
external multitask aux heads (ChEMBL + AID1851, the proven briford/jeremy recipe
we already run in ft_ext), and the NEW lever of mordred descriptors fed as x_d
into the graph backbone. Validate the descriptor channel cheaply first (Phase 0,
CPU, tabular probe on cached embeddings) before spending GPU on x_d chemprop.
Then rebuild a SIMPLIFIED pool around the winning backbone, keep placement knobs
unchanged, and report a verified candidate to Jackson.

## 3. PHASE 0 - Descriptor pipeline + tabular probe (CPU, no GPU, no submits)

Goal: (a) get a robust mordred descriptor matrix, (b) cheaply test whether
descriptors add honest signal ON TOP OF a learned embedding, before any GPU.

Steps:
  1. New conda env `mordred` (python 3.11, rdkit, mordred). NEVER install into
     the system python or an existing env.
  2. `src/mordred_features.py`: compute mordred descriptors for ALL molecules
     (train ~6,145 + test ~750, using the same SMILES the rest of the repo uses).
     - Use the NON-3D descriptor set (no conformers; 2D pipeline). If mordred's
       full calc is too slow/fragile, use a curated subset (the "low"/stable
       descriptors) - log exactly which set and how many.
     - NaN policy (ANTI-LEAK): for each descriptor column, if NaN fraction on
       TRAIN > 0.15, DROP the column. Remaining NaNs imputed with the TRAIN
       column median. Standardize (z-score) using TRAIN mean/std ONLY.
     - Save `cache/emb_mordred_all.parquet` (index = Molecule_Name, aligned 1:1
       with the train/test tables) + `cache/mordred_meta.json` (n_descriptors,
       dropped list, imputation/standardization stats, mordred version).
     - Verify 1:1 row alignment on Molecule_Name and SMILES-identical to the
       existing tables (the repo invariant).
  3. `src/mordred_probe.py` (CPU): the cheap pre-filter. Take an EXISTING cached
     learned embedding (chemprop_medium 600-d -> cache/emb_* / ft_oof_cpmed, and
     CheMeLeon) and compare three ridge models on the SAME scaffold folds:
        (A) embedding only
        (B) embedding + mordred (concat)
        (C) mordred only
     Report per-isoform OOF Pearson and nested macro R2 for each.
  GATE 0 (go to Phase 1 x_d work only if):
     (B) beats (A) on nested macro R2 by >= +0.003 AND on >= 2 of 4 isoforms.
     If (B) does NOT clear the gate: the tabular signal is weak, but x_d into a
     TRAINED FFN can still do what ridge can't (nonlinear interaction), so do NOT
     kill the lever - instead go to Phase 1 and let the trained x_d chemprop be
     the real test. Log the probe numbers either way.
  Report to Jackson: descriptor count, NaN/drop stats, probe table, gate result.

## 4. PHASE 1 - Chemprop D-MPNN with x_d + external aux heads (GPU)

Goal: the core accuracy lever. A small set of STRONG chemprop D-MPNNs, tested
with and without the x_d descriptor channel and with external aux heads.

Env: reuse the existing chemprop env (the one that built ft_cpmed/ft_dmpnn/ft_ext).
One GPU job at a time (the local-llama-server rule: never run heavy GPU in
parallel; background the runs, one tracked process each).

Build 4 model variants, all on the SAME scaffold folds, same external-aux data
table as ft_ext (challenge + ChEMBL + AID1851, label-separated, EXT_CAP as in
ft_ext.py):
  V1  control:      challenge-only D-MPNN, no x_d            (reproduce ft_dmpnn)
  V2  aux:          challenge + external aux heads, no x_d    (= our ft_ext recipe)
  V3  x_d:          challenge-only D-MPNN + mordred as x_d    (the NEW lever, clean)
  V4  aux + x_d:    challenge + external aux heads + mordred as x_d  (the bet)

x_d mechanism: pass the mordred matrix as the chemprop `Data.x` dense-feature
field so it is concatenated to the MPNN graph embedding before the FFN. Verify
the model input dim accounts for it (embedding_dim + n_mordred). If the installed
chemprop version's CLI does not expose x_d cleanly, wire it via the Data object
in a small training script (`src/chemprop_xd.py`) reusing ft_ext.py's data
loading. Keep the FFN head, LR schedule, and early-stopping IDENTICAL across
V1-V4 so the only variables are aux heads and x_d.

Eval: per-isoform OOF Pearson + nested macro R2 for each variant, on scaffold
folds. For the WINNER only, also compute on Butina cluster-disjoint folds (A5 -
our scaffold CV is optimistically biased; a winner that only wins on scaffold is
suspect).

GATE 1:
  - V4 (or V3) must beat V2 (best existing aux recipe) on nested macro R2 by
    >= +0.003 to adopt the x_d channel.
  - Log which of {aux, x_d, both} contributed, per isoform.
  Report to Jackson: the 4-variant table, the winner, scaffold vs Butina for the
  winner, gate result.

## 5. PHASE 2 - Simplified pool + placement (CPU)

Goal: apply O1. Replace the 19-member pool with a small, strong, decorrelated set
centered on the Phase 1 winner.

Steps:
  1. Start from the Phase 1 winning variant(s). Build a candidate pool of at most
     ~5 members, each of which must add HONEST nested R2 (LOFO / paranoid-merge
     test, same recipe as the r1_paranoid check that killed qHTS-DA and SMILES):
       - the winning chemprop (aux + x_d)
       - frozen chemprop_medium ridge (cpmed) - our proven decorrelated anchor
       - TabPFN-on-cpmed (the R2 win from last cycle, keep if it still adds)
       - optionally one 2D6 specialist (briford uses two CYP2D6-SLICE heads -
         2D6 is our weakest isoform, blind 2D6 is the laggard)
     DROP every member that is pool-redundant (we already know qHTS-DA, SMILES,
     CheMeLeon-ext, UMTM-regression are redundant - do not re-add them).
  2. Greedy nested blend over the survivors. Compare to cp17 (nested 0.4682).
  3. Placement knobs UNCHANGED (F_SPREAD / OOF_TO_BLIND / moments) - they are
     validated and the rank limiter is R2, not placement. Only re-fit placement
     if nested R2 moves > 0.03.
  GATE 2: the simplified pool must beat cp17 (0.4682) on nested macro R2 to be
     the new candidate. If it does not, the iteration is a no-op - do NOT ship a
     regression resubmit; keep cp17.
  Report to Jackson: final member list, per-member LOFO gain, nested R2, expected
  blind R2 at the 1.11 ratio, gate result.

## 6. PHASE 3 - Build + verify + report (no submit)

  1. `src/regression_final.py` -> build `cache/regression_final_cp18_submission.csv`
     from the Phase 2 pool (only if GATE 2 passed).
  2. `src/verify_submissions.py` + official validators: PASS, 750 rows, 1:1
     Molecule_Name join, SMILES-identical, zero nulls, std >= 0.01 per column.
  3. md5sum, commit + push to origin/main.
  4. Report to Jackson with the verified file + md5 + expected rank. DO NOT
     submit - Jackson submits.

## 7. TDI (SECONDARY - only if regression Phase 0-2 are done and time remains)

Our TDI rank (38) is already better than our regression rank (63). The x_d
descriptor channel is the one untried TDI lever: feed mordred as x_d into the
TDI backbone (the UMTM / chemprop TDI gate+shift heads) and test whether it lifts
the 2D6 TDI head (our 0.219 laggard) and the macro precision (the binding
constraint - top-12 macro precision 0.55-0.85 vs our 0.43). Same GATE logic:
must beat tdi_v5 (MA-MCC 0.3422) on nested to ship. Do NOT resubmit TDI on a
no-op. This is explicitly lower priority than regression.

## 8. DEAD LEVERS - do NOT retry (logged, verified dead)

  - R1 qHTS-only domain adaptation: standalone GO, POOL-REDUNDANT.
  - R3 SMILES-sequence transformer (ChemBERTa): standalone NO-GO, POOL-REDUNDANT.
  - R4 dead-zone ST-RAE refit: neutral.
  - R5 band-width placement: infeasible (no blind DRC bands).
  - SimCLR / 3D-conformer / pseudo-label / AutoGluon / cpmed-cpchm twins: dead.
  - External data as TABULAR GBM features (external-NN): no tabular win.
  - TDI external-positives naive pooling (AID1851 into LightGBM base): population
    shift, failed at both weights.
  - TDI T1 conjunction as a MACRO lever: it is a 3A4-level lever, not macro.
  NOTE: these are dead as POOL MEMBERS / as specified. The x_d descriptor channel
  is different: it goes INTO the backbone, so it is NOT one of these.

## 9. CONVENTIONS / CADENCE (carry over, non-negotiable)

  - Never submit without Jackson.
  - One GPU job at a time; background long runs (terminal background=true);
    never touch llama-server; never kill a running process without permission.
  - verify_submissions.py + official validators before showing ANY file.
  - Family-block / paranoid-merge audit before any member enters the pool.
  - Anti-leak: external data = ranking/pretrain/aux ONLY, never placement;
    descriptor stats fit on TRAIN only; label separation in the aux table.
  - Log honest numbers in NOTES.md (new section 23+) and commit + push at each
    phase gate.
  - Iterate with incremental logging; validate with short runs before full runs.

## 10. HONEST EXPECTATIONS

  - Realistic win: nested R2 0.4682 -> 0.50-0.53 (descriptor + clean aux
    chemprop), blind R2 ~0.55-0.58, reg rank 25-45.
  - This is NOT a top-12 cliff jump (that needs R2 ~0.62+). It is a meaningful
    20-30 rank climb on the regression board.
  - If GATE 0/1/2 fail in sequence, the honest output is "cp17 stands, no
    resubmit" - a no-op is a valid, logged result. Do not force a submission.

## APPENDIX - FRESH SESSION PROMPT (paste into a new session)

See docs/NEXT_AGENT_PROMPT.md for the paste-ready version.
