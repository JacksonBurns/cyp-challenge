# PROMPT FOR NEXT AGENT (written 2026-09-26, post fourth scored point / Sep 26 interim board)

Paste-ready handoff for the next Hermes agent session on the OpenADMET CYP
inhibition competition (repo ~/cyp-challenge, final deadline Nov 3 2026).
The previous version (written Sep 24, post third scored point) is obsolete:
its step 0 (safe TDI repair) has been EXECUTED and scored. NOTES.md section
14 is the current source of truth.

---

Continue work on the OpenADMET CYP inhibition competition in the repo
~/cyp-challenge (git repo, branch main, remote origin = JacksonBurns/cyp-challenge;
commit AND push at every stage boundary).

READ FIRST, in order: NOTES.md section 14 (fourth scored point, Sep 26
board, paired field-drift analysis), then section 13 (third scored point and
the TDI v3 failure), then section 12 (what was built), then section 11
(transfer-ratio history and the placement rule). docs/CHEMPROP_CHEMELEON_PLAN.md
is historical background only.

CURRENT STATE ON THE BOARD (Sep 26 interim reveal; files in
leaderboard/*_2026-09-26_interim_reveal2.csv):
- Pair on board = cache/regression_final_cp7_submission.csv +
  cache/tdi_submission_v2.csv. Step 0 of the old plan was executed: the v2
  TDI file recovered the score (MCC 0.321 vs v3's 0.306 on the prior board).
- Regression: rank 57/219, ST-RAE 0.5354, MA-R2 0.5192, Spearman 0.7251,
  Kendall 0.5522 (74th percentile). Top-1: R2 0.677 / rho 0.805 / ST-RAE 0.372.
- TDI: rank 39/114, MA-MCC 0.3209, acc 0.7252, prec 0.3590, rec 0.6716
  (66th percentile, Tier 2; Tier 1 starts at rank 27 = MCC ~0.341).

WHAT THE NEW BOARD CHANGED IN THE STRATEGY (sec 14 has the numbers):
1. The field moved. Paired drift of unchanged submissions between the two
   boards: regression got EASIER everywhere (median +0.073 R2) - our cp7
   gained only +0.027, i.e. we LOST ~0.05 of relative ground even while our
   rank improved. TDI got HARDER in our region (median drift negative at
   MCC >= 0.28) - our v2 beat its band by ~+0.04. Cross-board rank and
   absolute comparisons are unreliable (the blind set changed); use the
   paired drift tables or absolute nested OOF gates only.
2. TDI v3 deserves a softer verdict: on a same-blind estimate its ranking
   machinery was ~+0.04 better than v2's; what killed it was the operating
   point (2D6 fraction 0.33) and possibly selection noise. Re-open the v3
   pool in the family-block audit with fresh eyes, but keep v2's fractions
   until the audit clears any move.
3. New cheapest TDI gap = operating point. Our macro accuracy is 0.725
   while the entire top 10 sits at 0.839-0.859 with precision >= 0.44; we
   are at prec 0.359 / rec 0.672, i.e. over-predicting positives. v2's own
   shipped fractions may be worse than assumed - retune BOTH files.
4. Regression placement has a bounded surplus: at our Spearman, competitive
   neighbors carry R2/rho^2 = 1.02-1.15; we are at 0.988. A spread/calibration
   bump is worth maybe +0.04 blind R2. Ranking remains the main event.

EXECUTE IN THIS ORDER:

1. TDI - PRIORITY 1 (furthest from promotion bands; Tier 1 needs ~0.341,
   top-10 needs ~0.371).
   a. Family-block audit FIRST (unchanged mission from sec 13): rewrite the
      nested selection audit (src/tdi_blend_nested.py pattern) so entire
      MEMBER FAMILIES are held out together (family = {base}, {emb,tab},
      {tabcp,tabcpext}, ...). Report fold-nested AND family-block numbers
      for every candidate pool. A new pool must beat v2's fold-nested 0.268
      on the FAMILY-BLOCK audit before it is a submission candidate. Score
      candidates on absolute nested MCC, NOT cross-board deltas.
   b. Re-open the v3 pool ({base,emb,tab,tabcp,tabcpext}) inside that audit
      - the Sep 26 drift evidence says its ranking was probably real.
      Diversify further: TabICL/TabPFN on the chemprop_chemeleon checkpoint
      embeddings (cpchm, cache exists), TabICL on frozen CheMeLeon 2048-d
      embeddings (cache/chemeleon_emb.parquet, PCA-256 first - the memory
      cgroup OOM-kills raw 2048-d), D-MPNN classifier heads. Same scaffold
      folds (seed 7), same OOF conventions as src/tdi_tabicl_cp.py.
      Cap per-family blend weight at ~0.5 (v3's fatal 2D6 concentration was
      0.66 on one chemprop_medium family).
   c. FRACTIONS - NEW FIRST-CITIZEN WORK ITEM: run the expected-MCC
      posterior (src/tdi_fraction_opt_v4.py pattern) on the v2 pool's nested
      OOF FIRST, to see whether v2's shipped 0.08/0.36 itself beat a better
      fraction on OOF. The board profile (acc 0.725, prec 0.359 vs top-10
      acc >= 0.839, prec >= 0.44) says at least one isoform is badly
      mis-tuned. Then re-run on the best new pool. Ship only a fraction set
      good on BOTH E[MCC] and the family-block variant.
   d. Known dead ends (do not retry): pooling PubChem AID1851 qHTS binaries
      into the GBM base or as TabICL in-context rows (population shift);
      ChEMBL pretrain-then-finetune of CheMeLeon (weak, population shift).

2. REGRESSION RANKING - PRIORITY 2. We are 57; the rank-34 band needs
   R2 ~0.596 at our rho, i.e. macro R2 +0.08, i.e. per-isoform Pearson
   +0.05-0.07. Note the field drift (+0.073 median R2 for UNCHANGED files):
   expect to need more than +0.08 OOF by the Nov 3 reveal, so start now.
   a. Cheap proven lever: more heterogeneous frozen-embedding ridge probes
      into the blendN pool (this lever produced the last +0.026 nested /
      +0.045 blind R2). Candidates: frozen CheMeLeon MP mean-pooled 2048-d
      ridge (PCA or big alpha grid only - small alphas overfit), Uni-Mol or
      any other pretrained encoder embeddings on box, and the NOW-PUBLIC
      method reports to mine: rasayan-labs github.com/whis9/rasayan-cyp
      (rank 14, R2 0.627), stir_bar github.com/lachrymator/openadmet-cyp-challenge-public
      (rank 28 reg + rank 6 TDI - read both), briford
      supercowpowers.github.io/workbench/blogs/cyp_challenge/ (rank 15).
      jeremy's kit is already mined (cpmed/cpchm). Apply the
      cp_embeddings.py ridge pattern with the corrected alpha grid
      (100/1000/10000).
   b. Apply the FAMILY-BLOCK audit from 1a to the cp7 pool before betting a
      resubmission on further blend gains: cpmed and cpchm share the
      adme_pretrain lineage; verify the cp7 gain survives holding out the
      whole frozen-ridge family. cp7 transferred fine (+0.045 blind), so
      the risk is likely bounded - but know which way it cuts before Nov 3.
   c. Modest extras: 2nd D-MPNN seed, ft_ext 5th seed (~+0.001-0.003 each;
      filler for GPU idle time only, never displace priority work).
   d. Blend via src/blendN_ext.py / blendN_all pattern + nested check; new
      candidate ships only if family-block-honest macro R2 >= ~0.47 (cp7's
      fold-nested was 0.4385 and it blind-scored 0.5192; require a clear
      jump, not noise). Placement knobs UNCHANGED by default.
   e. OPTIONAL last-mile placement probe (only after a new ranking candidate
      is verified, and only if a submission slot is imminent anyway): the
      R2/rho^2 neighbor table in sec 14 suggests a modest spread INCREASE
      (~1.05-1.10 on the F_SPREAD shape, or calibrating the z-space moments
      on the two same-file scored points) is worth ~+0.04 blind R2. Test it
      in simulation only, keep it in a separate candidate file, and never
      let it displace the known-good cp7 file.

3. SUBMISSION CADENCE (competition closes Nov 3, latest valid submission
   counts, 12h cooldown, NEVER submit without Jackson's explicit go-ahead +
   his HF login on the Submit tab of https://huggingface.co/spaces/openadmet/
   cyp-challenge): after each verified beating candidate exists, ask Jackson
   for a slot; report expected blind numbers using k=1.05 (regression) and
   ABSOLUTE nested OOF gates (the TDI cross-board drift shows blind-to-blind
   extrapolation is not trustworthy), never raw OOF. In the final week: stop
   model work by Nov 1, re-verify, submit the best pair by Nov 1, and only
   the final submission matters - keep the known-good files safe and give
   them new filenames (never overwrite files already submitted:
   regression_final_cp7_submission.csv and tdi_submission_v2.csv are the
   incumbents).

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
- Append a NOTES.md log entry (sec 15+) and commit with honest numbers at
  every stage boundary; when context runs low, append a handoff section to
  NOTES.md before stopping. No em dashes in output.
- Report to Jackson after each stage: honest nested + family-block deltas vs
  what is on the board (cp7 R2 0.5192 / tdi v2 MCC 0.3209 - but prefer
  absolute OOF comparisons over board deltas, see above), whether a new
  verified candidate exists, and a recommendation.
