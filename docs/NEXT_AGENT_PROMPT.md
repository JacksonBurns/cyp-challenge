# PROMPT FOR NEXT AGENT (written 2026-09-24, post third scored point)

Paste-ready handoff for the next Hermes agent session on the OpenADMET CYP
inhibition competition (repo ~/cyp-challenge, final deadline Nov 3 2026).
The previous version of this file described the Sep 23 interim state and is
obsolete; NOTES.md section 13 is the current source of truth.

---

Continue work on the OpenADMET CYP inhibition competition in the repo
~/cyp-challenge (git repo, branch main, remote origin = JacksonBurns/cyp-challenge;
commit AND push at every stage boundary).

READ FIRST, in order: NOTES.md section 13 (third scored point, Sep 24, and
the failure analysis of the TDI v3 submission), then section 12 (what was
built), then section 11 (transfer-ratio history and the placement rule).
docs/CHEMPROP_CHEMELEON_PLAN.md is historical background only.

CURRENT STATE ON THE BOARD (submitted 2026-09-24 14:17 UTC, repo pinned 0fa9dd4):
- Regression: cache/regression_final_cp7_submission.csv scored MA-ST-RAE
  0.5415 / MA-R2 0.4924 / Spearman 0.708, rank 61/238. This IMPROVED on the
  previous point (0.5870 / 0.4472, rank 73). Keep it.
- TDI: cache/tdi_submission_v3.csv scored MA-MCC 0.3057, rank 54/125. This
  REGRESSED against cache/tdi_submission_v2.csv on the previous board
  (0.3419, rank 39). The v3 blend did not transfer: nested OOF macro MCC
  0.326 predicted an improvement, blind came out worse. Root cause: the
  per-fold greedy nested audit shares fold structure across the whole
  chemprop-embedding member family (tabcp 2D6 weight 0.448 + tabcpext
  0.216 = 0.66), so it prices within-family variance but not family-level
  variance; the 2D6 fraction move 0.08 -> 0.33 then amplified bad top-of-list
  ranking (blind precision 0.408 -> 0.346, recall 0.537 -> 0.657).

THREE SCORED POINTS NOW (use for calibration, details in sec 13):
regression OOF->blind k = 1.053 / 1.042 / 1.060 - placement machinery
(F_SPREAD, OOF_TO_BLIND, BLIND_MOMENTS) is validated and stays AS SHIPPED
unless OOF rho moves by >0.03. TDI transfer ratio was 1.28 for v2 but 0.94
for v3 - do not trust TDI nested gains until they pass the family audit
below.

EXECUTE IN THIS ORDER:

0. IMMEDIATE SAFE REPAIR (needs Jackson's go-ahead + HF login; 12h cooldown
   from 14:17 UTC, earliest slot ~Sep 25 02:17 UTC): re-submit the existing
   VERIFIED files as the pair cache/regression_final_cp7_submission.csv
   (identical content to what is on the board, so zero regression risk) +
   cache/tdi_submission_v2.csv (the 0.342 MCC file). This alone should
   recover ~15 TDI rank places. Do not rebuild anything for this - just
   re-verify both files with src/verify_submissions.py and ask Jackson.
   If Jackson declines an early slot, fold this into step 3's first submit.

1. TDI REBUILD - PRIORITY 1 (the only regressed track).
   a. Family-block audit FIRST: rewrite the nested selection audit
      (src/tdi_blend_nested.py pattern) so entire MEMBER FAMILIES are held
      out together (family = {base}, {emb,tab}, {tabcp,tabcpext}, ...).
      Report both fold-nested and family-block numbers for every candidate
      pool. Calibrate expectations on the two scored TDI points: v2-style
      pools transferred at 1.28x, the v3 pool at 0.94x. A new pool must beat
      v2's fold-nested 0.268 on the FAMILY-BLOCK audit, not just the
      fold-nested one, before it is a submission candidate.
   b. Diversify the member pool: the v3 pool's failure mode was single-family
      dominance. Add genuinely diverse members: TabICL (or TabPFN) on the
       chemprop_chemeleon checkpoint embeddings (cpchm, cache exists),
       TabICL on the frozen CheMeLeon 2048-d embeddings (cache/chemeleon_emb.
       parquet, PCA-256 first - the memory cgroup OOM-kills raw 2048-d),
       D-MPNN-derived classifier heads. Same scaffold folds (seed 7), same
       OOF conventions as src/tdi_tabicl_cp.py.
   c. Cap per-family blend weight at ~0.5 in the greedy selection so no
      single embedding family can carry an isoform head again.
   d. Fractions: rerun the expected-MCC posterior machinery
      (src/tdi_fraction_opt_v4.py pattern) on the NEW nested OOF, and pick
      the fraction that is good on BOTH E[MCC] and the family-block variant.
      Remember 2D6: shipped v2 f=0.08 blind-scores fine, v3 f=0.33 was
      built on the poisoned audit - treat any big fraction move as guilty
      until the family audit clears it.
   e. Known dead ends (do not retry): pooling PubChem AID1851 qHTS binaries
      into the GBM base or as TabICL in-context rows (both fail hard,
      population shift); ChEMBL pretrain-then-finetune of CheMeLeon (weak,
      population shift).

2. REGRESSION RANKING - PRIORITY 2 (biggest remaining headroom: top-12 band
   is R2 0.58-0.68 / Spearman 0.75-0.79; we are 0.492/0.708 and the gap is
   100% ranking now, placement is validated better-than-field-average).
   Weakest heads per-iso OOF: 1A2 0.596 and 2D6 0.464.
   a. Cheap proven lever: more heterogeneous frozen-embedding ridge probes
      into the blendN pool (this lever produced the last +0.026 nested and
      +0.045 blind R2). Candidates: frozen CheMeLeon MP mean-pooled 2048-d
      ridge (PCA or big alpha grid only - small alphas overfit, see sec 12),
      Uni-Mol or any other pretrained encoder embeddings available on box,
      chemprop checkpoints from jeremy's public kit (/tmp/jeremyscripts/
      CYP_Challenge/checkpoints/, re-pull if the clone is gone - see NOTES
      for repo name). Apply the cp_embeddings.py ridge pattern with the
      corrected alpha grid (100/1000/10000).
   b. Apply the same FAMILY-BLOCK audit discipline from step 1a to the cp7
      pool before betting a resubmission on further blend gains: cpmed and
      cpchm share the adme_pretrain lineage; verify the cp7 gain survives
      holding out the whole frozen-ridge family. If it does not, note it
      (it still scored well blind, so the risk may be bounded - but know
      which way it cuts before Nov 3).
   c. Modest extras: 2nd D-MPNN seed, ft_ext 5th seed (each ~+0.001-0.003,
      only worth running as filler on GPU idle time - never displace priority
      work).
   d. Blend via src/blendN_ext.py / blendN_all pattern + nested check; new
      candidate ships only if family-block-honest macro R2 >= ~0.45 (vs
      cp7's 0.4385 fold-nested). Placement knobs UNCHANGED.

3. SUBMISSION CADENCE (competition closes Nov 3, latest valid submission
   counts, 12h cooldown, NEVER submit without Jackson's explicit go-ahead +
   his HF login on the Submit tab of https://huggingface.co/spaces/openadmet/
   cyp-challenge): after each verified beating candidate exists, ask Jackson
   for a slot; report expected blind numbers using k=1.05 (regression) and
   the family-block-calibrated TDI ratio, never raw OOF. In the final week:
   stop model work by Nov 1, re-verify, submit the best pair by Nov 1, and
   only the final submission matters - keep the known-good files safe and
   give them new filenames (never overwrite files already submitted).

CONSTRAINTS (all still in force):
- Single Quadro RTX 6000 shared with llama-server: one heavy GPU job at a
  time, check nvidia-smi, NEVER kill running processes (especially
  llama-server) without asking Jackson.
- Foreground terminal caps ~420s and SIGTERMs children - long GPU jobs MUST
  use terminal(background=true), ONE script per tracked process (a wrapper
  that exits also kills its queued children).
- All CV uses scaffold folds (seed 7 conventions in src/); label-derived NN
  features must self-exclude (q_idx).
- External data (ChEMBL, PubChem AID1851) allowed for RANKING only, never
  for placement moments (not DRC-calibrated).
- New deps go into conda envs (cyp = LightGBM/rdkit only, chemeleon,
  chemprop-dev, tabicl-chemeleon); never pip install into cyp.
- Every candidate passes BOTH official validators and src/verify_submissions.py
  (row-for-row SMILES/Molecule_Name match vs data/cyp-challenge-TEST-BLINDED.csv)
  before it is shown to Jackson.
- Append a section-11/12/13-style log entry to NOTES.md and commit with
  honest numbers at every stage boundary; when context runs low, append a
  handoff section to NOTES.md before stopping. No em dashes in output.
- Report to Jackson after each stage: honest nested + family-block deltas vs
  what is on the board, whether a new verified candidate exists, and a
  recommendation.
