# PROMPT FOR NEXT AGENT (written 2026-09-23, post interim reveal)

Paste-ready handoff for the next Hermes agent session on the OpenADMET CYP
inhibition competition (repo ~/cyp-challenge, deadline Nov 3 2026 final).

---

Continue work on the OpenADMET CYP inhibition competition in the repo
~/cyp-challenge (my home directory). Read ~/cyp-challenge/NOTES.md —
SECTION 9 IS THE MOST IMPORTANT THING IN THE REPO (interim reveal results
and diagnosis, Sep 23) — then HANDOFF.md for history, then
docs/CHEMPROP_CHEMELEON_PLAN.md for the staged model-improvement plan.
Env: ~/miniforge3/envs/cyp/bin/python (LightGBM/rdkit only, no torch —
never pip install into it; torch work goes in the existing chemeleon /
chemprop-dev conda envs). Repo is a git checkout on main; commit, never
push.

Where we stand (interim reveal, scored): regression MA-ST-RAE 0.6766
(rank 93/229; top 0.374), TDI MA-MCC 0.3149 (rank 60/121; top 0.513).
Both submission files passed official validators (cache/raw_regression_-
submission.csv, cache/tdi_submission.csv; recipe + verification script
src/verify_submissions.py). The reveal's key diagnostic: our blind MA-R2
(0.387) matches our OOF Pearson^2 (0.349), so our moment-calibrated
PLACEMENT IS ESSENTIALLY OPTIMAL ALREADY — every remaining point of
ST-RAE must come from RANKING quality (our Spearman 0.616 vs top field
0.75-0.79). The blind test is chemistractive, only ~1.3-2.3k labels per
isoform, and every top-20 entry uses external data (ChEMBL 37 CYP +
PubChem AID1851) and/or pretrained encoders. A CheMeLeon+TabPFN entry
scored 0.631 with Spearman 0.689 — better features than our FP+RDKit
GBM, which is the clearest upgrade path.

YOUR TASK for the Nov 3 final, in priority order (details in the plan
file): (1) CheMeLeon frozen-encoder embeddings as features into the
existing LightGBM pipeline — weights already on box at
~/chemeleon_nazarov/chemeleon_mp.pt, skill 'chemeleon-fine-tuning' has
working code patterns; ship only if per-isoform scaffold-CV OOF Pearson
beats 0.527/0.604/0.399/0.771 (cache/oof_pearson.json). (2) External
data: ChEMBL 37 CYP activities + PubChem AID1851 qHTS — pretrain or
multitask, use for RANKING only (external assays are not DRC-calibrated,
do not let them move your placement moments). (3) Chemprop D-MPNN and/or
CheMeLeon fine-tune ensembled with the GBM blend on OOF; same for the TDI
classifier (top is 0.513; jeremy's open repo documents a 0.402 TDI recipe
worth reading). (4) Cheap recalibration: our OOF-to-blind correlation
inflation (OOF_TO_BLIND in src/final_submit.py) measured ~1.05 at reveal
vs assumed 1.32/1.23/1.66/1.07 — shrinking the placement spread by roughly
that factor on the current predictions is worth ~0.02-0.05 ST-RAE with no
retraining; validate the idea against evaluation/ ST-RAE logic before
trusting it. (5) TDI threshold fraction: rerun the expected-MCC machinery
(src/tdi_fraction_opt.py) on any new OOF; do not blindly keep 43%/8% or
revert to train-rate matching without the analysis.

Single GPU (Quadro RTX 6000) shared with llama-server: check nvidia-smi,
one heavy job at a time, NEVER kill running processes. All CV must use
scaffold_groups from src/run_regression.py (random CV is inflated on this
test design). Label-derived NN features must self-exclude (q_idx) in CV.
cache/tdi_oof.npz is in X_train-dedup row order, not TDI-CSV order —
rebuild labels exactly as run_tdi.py does. Raw LightGBM probas are
over-dispersed; use observed rank-conditional curves for MCC-threshold
simulations. Judge every candidate on calibrated expectations; the
interim file that scored 0.677 is your baseline to beat.

Every candidate submission file must pass src/verify_submissions.py
(official validators + row-for-row SMILES/Molecule_Name match against
data/cyp-challenge-TEST-BLINDED.csv) before you show it to me. Update
NOTES.md and commit at every stage boundary. Report to me (Jackson) at
stage boundaries. NEVER submit anything without my explicit go-ahead —
submissions need my HF login on the Submit tab of
https://huggingface.co/spaces/openadmet/cyp-challenge, there is a 12h
cooldown, and the LATEST valid submission counts (so the final days are:
test early, keep a safe file, submit the best last).
