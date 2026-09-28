# PROMPT FOR NEXT AGENT (written 2026-09-27, "FULL SEND" - GPU work order)

Paste-ready handoff for the next Hermes agent session on the OpenADMET CYP
inhibition competition (repo ~/cyp-challenge, branch main, remote origin =
JacksonBurns/cyp-challenge; commit AND push at every stage boundary). Final
deadline Nov 3 2026. The previous version of this file (Sep 26, post interim
board) had steps 1 and 2 EXECUTED and logged - NOTES.md sec 16 (TDI audits +
v4_candidate) and sec 17 (regression step 2). Read this file, then NOTES.md
sec 17, then sec 16, then sec 14. Do not re-do executed steps.

## BOARD STATE (unchanged since Sep 26 reveal; do not trust older numbers)
- Regression: cp7 on board, rank 57/219, MA-R2 0.5192, ST-RAE 0.5354.
  New candidate cache/regression_final_cp8d_submission.csv is built, verified,
  dominates cp7 on every honest gate, but by only +0.002 - it is the Nov 1
  default, NOT a mid-course slot claim.
- TDI: v2 on board, rank 39/114, MA-MCC 0.3209 (Tier 2; Tier 1 ~0.341).
  Candidate cache/tdi_submission_v4_candidate.csv (capped v3 pool @ fractions
  0.33/0.26) is built + verified; honest blind range ~0.30-0.36, a genuine
  promotion bet. It holds the next submission slot if Jackson approves one.
- Field drift warning (sec 14): the field is improving on unchanged files
  (regression median +0.073 R2 between boards). Absolute OOF gates only;
  never compare across boards.

## WHAT SEC 17 PROVED IS SPENT (do not retry)
- Frozen CheMeLeon 2048-d ridge probe (chmridge): zero blend weight, dead.
- Near-test retrieval of external unlabeled rows (ft_ext_near): +0.0002, noise.
- cp-family family-block audit: cp7's gain is honest (LOFO +0.019 survives);
  the whole CheMeLeon lineage drop costs 0.0017. Family-block audit exists:
  src/regression_family_block.py - reuse it for any new pool.
- rasayan-labs repo (rank 14) is EMPTY; stir_bar + briford reports already
  mined (sec 17) - their remaining levers need their whole stack, not cheap.
- The cheap members are done: cp8d is +0.002, not +0.08. The rank-34 band
  needs ~+0.08 blind R2. Only a genuinely NEW encoder family can get there.

## THE MISSION: fine-tune / extract from NEW encoders (GPU queue, in order)

CheMeLeon fine-tune status (sec 18 audit - answer to "have we tried it"):
YES for regression (ft_frozen / ft_fullft / ft_ext all in the blend pool;
pretrain-then-ft was weak/population-shift). NOT TRIED: TDI binary task,
adme_pretrain full-tune, non-CheMeLeon encoders. That is the queue below.

### Task 1 (TDI priority, ~40 min): CheMeLeon fine-tuned on the TDI binary task
- New script src/tdi_ft_chemeleon.py: copy src/tdi_dmpnn.py (2-head BCE
  is_TDI, challenge rows ONLY, TDI+Emax label union, scaffold folds seed 7)
  but swap the encoder: nn.BondMessagePassing loaded from
  ~/chemeleon_nazarov/chemeleon_mp.pt via ft_chemeleon.py's pattern
  (weights_only=False load, hyper_parameters, 2-head BinaryClassificationFFN,
  d_h=300). Full-FT and frozen-head both, seed 0 first.
- Output cache/tdi_ft_chemeleon_oof.npz (same schema as tdi_dmpnn_oof.npz).
- Evaluate standalone best-fraction MCC (tdi_blend_nested.py machinery), then
  add to the pool {base,emb,tab,tabcp,tabcpext} + ft_chemeleon in
  src/tdi_blend_family_block2.py pattern as family 'ft_chemeleon' (a NEW
  lineage distinct from the frozen cp ridge family). GATE: candidate pool must
  beat v4_candidate's capped-nested macro (+0.018 over v2's nested 0.2685 is
  the bar v4 set at E[MCC] posterior 0.286) AND pass LOFO with the cp family
  capped at 0.5. Only then build a new TDI candidate file (new filename,
  never overwrite incumbents or v4_candidate).
### Task 2 (regression, ~2 h): adme_pretrain checkpoints FULL-TUNED
- jeremy's /tmp/jeremyscripts/CYP_Challenge/checkpoints/chemprop_chemeleon.pt
  (and chemprop_medium.pt) were only ever FROZEN + ridge-probed (cpmed/cpchm
  = the cp family that carried cp7). Fine-tune them on the ft_ext 12-head
  recipe (src/ft_ext.py: --pretrained flag exists, + --prefix to keep tags
  disjoint): e.g. --pretrained .../chemprop_chemeleon.pt --full-ft
  --prefix admecd, seeds 0-1. Expectation: better than frozen ridge singles
  (0.548-0.550 macro region) if population-shift behavior follows jeremy's
  kit, but verify vs the sec 12 pretrain-warning before believing it.
- Blend as tag admecd* via blendN_all/blend_nested_all pool copies; family
  map must keep admecd SEPARATE from cpridge {cpmed,cpchm} (different
  training: tuned vs frozen) but SAME corpus lineage - run family-block both
  ways (merged vs split) like tdi_blend_family_block2.py does.
### Task 3 (the real +0.08 bet, ~half day incl. debug): Uni-Mol 3D encoder
- stir_bar's write-up (rank 28 reg / 6 TDI) explicitly credits 3D
  representation diversity. Install uni-core/Uni-Mol in a NEW conda env
  (python=3.11, per standing rule never pip into cyp), download the
  pretrained conformation-generation + molecular property checkpoints.
- GO/NO-GO GATE FIRST: verify the weights actually download on this box
  (TabPFN died with a 401 here - check before investing hours). If blocked,
  log it as a dead end and move to Task 4.
- If go: extract 256-d molecular embeddings for all 6,895 train+test SMILES
  (conformer gen ~minutes per thousand; cache parquet like
  cache/chemeleon_emb.parquet), then ridge probe (big alpha grid 100/1000/
  10000, cp_embeddings.py pattern) AND a GBM member. Blend + nested +
  family-block as a NEW family. This is the one lever all top reports agree
  on: representation diversity.
### Task 4 (filler, ONLY when the GPU is otherwise idle): seed filler
- ft_ext seed 5 (tools/run_ext_seeds234.sh pattern), d2d6 seed 10. ~+0.001
  each; never displace Tasks 1-3.
### Task 5 (optional, medium effort): ft_ext intermediate embeddings
- ft_ext/fold models are not persisted (grep state_dict, ft_ext.py). Add
  save-state or same-run hook to extract 2048-d pre-FFN states from the
  FINE-TUNED models, then ridge/tabFM members. Only if Tasks 1-3 finish early.

## CADENCE, GATES, CONSTRAINTS (all still in force)
- Submission cadence: NEVER submit without Jackson's explicit go-ahead + his
  HF login at the Submit tab of https://huggingface.co/spaces/openadmet/
  cyp-challenge. Latest valid submission counts; 12h cooldown. When a task
  produces a candidate that passes its gate, report honest nested +
  family-block numbers and ASK for a slot; recommend the best single file
  (TDI slot competition is real: v4_candidate may already have spent it).
- Final week rule: stop model work by Nov 1, re-verify everything, submit the
  best pair by Nov 1. Nov 1 defaults right now: regression cp8d, TDI
  v4_candidate - both verified, both dominate incumbents honestly.
- Single Quadro RTX 6000 shared with llama-server: ONE heavy GPU job at a
  time, check nvidia-smi first, NEVER kill running processes (especially
  llama-server) without asking Jackson.
- Foreground terminal caps ~420s and SIGTERMs children: long GPU jobs MUST
  use terminal(background=true) with notify_on_complete, ONE tracked script
  per chain (tools/run_ext_seeds234.sh pattern), poll via process tool.
- Memory cgroup OOM-kills TabICL on raw 2048-d - PCA-reduce to 256 first.
- Envs: cyp = LightGBM/rdkit/blends; chemeleon = torch+chemprop (CheMeLeon
  ft_ext/ft_chemeleon); chemprop-dev = D-MPNN scripts; tabicl-chemeleon.
  New deps go into new/existing conda envs, never into cyp.
- All CV: scaffold folds seed 7 conventions (src/run_regression.py +
  tdi_dmpnn.py inline copies); NaN-masked multi-task; label-NN features
  self-exclude (q_idx); external data for RANKING only, never placement
  moments (F_SPREAD/OOF_TO_BLIND/BLIND_MOMENTS knobs stay UNCHANGED - sec 17
  placement probe showed R2 vs ST-RAE are near-zero-sum around f=1).
- Every candidate passes BOTH official validators + src/verify_submissions.py
  (row-for-row vs data/cyp-challenge-TEST-BLINDED.csv); new filenames only
  (incumbents regression_final_cp7_submission.csv + tdi_submission_v2.csv and
  candidates v4/cp8d must never be overwritten).
- Append NOTES.md sections (19+) with honest numbers at every stage boundary;
  commit AND push (origin main) per stage; when context runs low, append a
  handoff section to NOTES.md before stopping. No em dashes in output.
- Report to Jackson after each task: honest nested + family-block deltas
  (absolute OOF only), whether a verified candidate exists, recommendation.

## FIRST COMMANDS (copy-paste start)
cd ~/cyp-challenge && git pull origin main
sed -n '/## 18/,/EOF/p' NOTES.md   # sec 18 fine-tune audit
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
# then Task 1: write src/tdi_ft_chemeleon.py from tdi_dmpnn.py +
# ft_chemeleon.py's encoder-load pattern, run seed 0 (~40 min, background):
~/miniforge3/envs/chemeleon/bin/python src/tdi_ft_chemeleon.py --seed 0
