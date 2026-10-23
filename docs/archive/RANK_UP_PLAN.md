# CYP-CHALLENGE RANK-UP PLAN (Oct 4)

Supersedes `docs/FINAL_PUSH_PLAN.md` (CYP-UMTM, now complete: cp15/cp16 regression
+ tdi_v5 shipped, board reg rank 50 / TDI rank 40). This is the next program:
get materially higher on both boards before the Nov 3 close. Anything is on the
table. Written for a FRESH session with no prior context - read this top to
bottom, then read NOTES.md sec 9-22 for the full history before running GPU.

## 0. WHERE WE STAND (Oct 4 board, post-final-push)

| track | our rank | our score | top | top score | gap (scored metric) |
|---|---|---|---|---|---|
| Regression (MA-ST-RAE, lower=better) | 50/258 | 0.5181 | preheat-to-450 | 0.3772 | 0.141 |
| Classification/TDI (MA-MCC, higher=better) | 40/148 | 0.3422 | TeamPozeSCAF | 0.5346 | 0.192 |

Our regression: R2 0.515, Spearman 0.721. Our TDI: precision 0.433, recall 0.501,
acc 0.813. Incumbent files on the board (do NOT overwrite, new filenames only):
regression `regression_final_cp15_submission.csv` (cp16 near-identical, pool record
nested 0.4569) and TDI `tdi_submission_v5_candidate.csv` (now the 8-col combined
schema: 4 pIC50 + 2 is_TDI). Revert spine: `regression_final_cp10_submission.csv`
+ `tdi_submission_v4_candidate.csv` (both intact).

Two structural facts from the board, both load-bearing for this plan:

**Regression is a ranking cliff.** R2 vs rank: #1-12 sit at R2 0.60-0.66 /
Spearman 0.75-0.78; rank 30 R2 0.564; rank 40 R2 0.548; US rank 50 R2 0.515;
rank 80 R2 0.480; rank 100 R2 0.268 (collapse). The top-20 band is a distinct
plateau. Placement is exhausted (4 scored points, k=1.04-1.06, F_SPREAD
validated). The whole gap is per-isoform Pearson: we need ~+0.06-0.08 macro
Spearman (0.72 -> 0.78) to reach the top-12. Our family-block audits show the
2D-encoder families we own are saturated (+0.002-0.006 per new member at the
blend level). To jump the cliff we need a GENUINELY NEW representation class or
a better reader on the classes we have, not more of the same.

**TDI is a precision game, not a recall game.** Top-12 macro precision:
TeamPozeSCAF 0.851, nova 0.708, Asidsal11 0.712, lex 0.606, NIPER 0.554,
harinu 0.532, TeamPrescience 0.530, N283T 0.515, yourchoice 0.506 - at recall
0.41-0.64. We ship 0.433. The leaders get there by calling FEWER, cleaner
positives. Our standing assumption "the 0.45 precision floor is unreachable on
2D6 by any proba we own" is the #1 thing to attack (see 2).

## 1. CHALLENGE THESE ASSUMPTIONS (explicit, per Jackson's directive)

A1. **"is_TDI is a monolithic classification."** FALSE. Verified Oct 4 on our own
   data (100% of rows, both isoforms): `is_TDI <=> (pi_TDI > 4.301) AND
   (pi_TDI - pi_dir > 0.301)`, where pi_TDI = TDI-condition arm pIC50, pi_dir =
   direct-arm pIC50. This is Lizard Wizard Gizzard's (TDI rank 80, open code,
   CYPChallenge repo) central result, and it matches their published numbers
   exactly: 3A4 "gate" (pi_TDI>4.301) strikes out 39.8% of rows, oracle MCC
   0.567; 2D6 "shift" (pi_TDI-pi_dir>0.301) oracle MCC 0.901; conjunction =
   label. **We have been classifying the conjunction directly with 1.5k/3.5k
   labels. The decomposition into two POTENCY regressions + a boolean rule is
   untried by us and is the highest-leverage TDI idea in this plan.** Potency
   is where we are already strong (UMTM has TDI-condition + direct pIC50 heads),
   so the gate and shift scores come almost free.

A2. **"Precision floor is unreachable on 2D6."** The 2D6 label is *almost pure
   shift* (gate closes only 5.8% of 2D6 rows; shift oracle 0.901). So 2D6 MCC
   is bounded by how well we predict the TDI-arm-minus-direct-arm potency
   DIFFERENCE, a 1D quantity. Our UMTM TDI head (a classifier on embeddings)
   gets 2D6 0.219. A model that predicts the shift directly (regress pi_TDI and
   pi_dir, take the difference, threshold) attacks the actual signal. Untested.

A3. **"Fine-tuning beats freezing."** Our own finding (ft_ext/full-FT > frozen)
   is the OPPOSITE of jeremy's (frozen >> fine-tune; he is rank 51 reg). We
   resolved this by running both and letting the data decide - that was right.
   BUT we never tried jeremy's specific winner: **TabPFN on FROZEN
   chemprop_medium embeddings**, his single best regression model. We are
   401-blocked on TabPFN (original v1 weights now gated; current versions need a
   free PriorLabs token, not an HF one). See lever R2 (unblock TabPFN) and R4.

A4. **"External data helps only via the 29k-row multitask."** We tried
   ChEMBL+AID1851 MIXED pretrain (ft_pre) -> weak (population shift). We never
   tried **qHTS-ONLY domain adaptation** (AID1851, no ChEMBL) then freeze -
   jeremy's best-ever CYP2D6 single. See lever R3.

A5. **"Scaffold folds seed 7 are the right CV."** Every top writeup (jeremy,
   stir_bar) uses **Butina cluster-disjoint folds**, not Murcko scaffold.
   stir_bar's blind-vs-CV plot shows 100% of regression endpoints came out
   WORSE on the blind set than in CV (median MAE +0.208) - i.e. our scaffold CV
   is optimistically biased for the analog-expansion test design. Rebuilding all
   OOF gates on Butina folds is expensive; we do NOT recommend a full rebuild
   this cycle (see 5, trade-off), but new members should be judged on BOTH and
   any candidate that only wins on scaffold CV is suspect.

A6. **"The metric is a ranking metric, optimize Pearson."** Lizard Wizard show
   ST-RAE's per-compound loss `L(p)=|p-clip(p,lo,hi)|` can be trained against
   DIRECTLY: a booster under absolute_error fitted to `clip(oof_pred, lo, hi)`
   (out-of-fold, to avoid the self-target trap) is a majorise-minimise descent
   on the metric's own gradient, worth +0.0197 rank / -0.0199 ST-RAE for them,
   sign 16/16 per-enzyme cells. We optimize Pearson then place; the dead-zone
   refit is untried. Cheap, orthogonal to ranking, likely +0.01-0.02 ST-RAE.
   See lever R5.

## 2. INTEL FROM PUBLIC SUBMISSIONS (mined Oct 4, repos in /tmp/_scout)

### Positive (things that work, that we have NOT done)
- **Lizard Wizard Gizzard** (CYPChallenge, TDI rank 80, reg rank 60): the
  conjunction decomposition (A1/A2); the ST-RAE dead-zone refit (A6); a
  credible-band-width model via isotonic regression from the label alone
  (R2 0.93-0.97 on band width, gives the ST-RAE-optimal spread directly);
  a per-enzyme 31-subset member search. Their shipped TDI classifier =
  (gate_prob from a 4-member dead-zone ensemble on the TDI arm) x (shift_prob
  from a classifier on Delta>0.301), threshold the calibrated product.
- **stir_bar** (openadmet-cyp-challenge-public, TDI Tier-1 rank 6, reg rank 28):
  representation diversity = SMILES-sequence transformers + MPNN + 3D-conformer
  + tabFM-on-frozen-emb + FP + additive-fragment; masked multitask aux
  pretraining; 88,683 blind near-neighbours retrieved (4,999 admitted,
  properties-only); Butina folds; NON-NEGATIVE stacking (cancelling
  coefficients transfer badly); PCA feature budget on tabular legs; per-source
  legs. Their 3A4 TDI blind 0.504 (they win TDI on 3A4, lose on 2D6 - same
  split we have).
- **jeremy** (openadmet_scripts, reg 51 / TDI 17, open code, /tmp/jeremyscripts):
  TabPFN on frozen chemprop_medium (best single reg); TabICL on frozen
  chemprop_medium (TDI); 17-head multitask warm-started from chemprop_medium
  (largest ensemble contributor); **qHTS-ONLY domain-adapt then freeze**
  (best-ever 2D6); Caruana bagged ES (not NNLS, not plain mean);
  blind-moments placement (we already do this, k validates).
- **briford** (reg rank 12, blog): 4 Chemprop D-MPNNs on challenge+ChEMBL+qHTS
  as SEPARATE heads incl. two CYP2D6-SLICE specialists; affine placement to the
  blind population (we already do).

### Negative (confirmed dead ends - do NOT re-run)
- **AutoGluon / plain LightGBM on frozen embeddings** < TabPFN on same features
  (jeremy). Net-neutral-to-negative in the ensemble.
- **3D shape/polarity descriptors** (Jazzy polarity + USR/USRCAT/PMI): weakest
  standalone, net-negative (jeremy). Confirms our Uni-Mol ridge DEAD result.
- **"Vanilla" CheMeleon** (public ckpt, zero CYP adaptation): redundant with
  classical descriptors, NOT decorrelated as hoped (jeremy).
- **Self-supervised SimCLR contrastive pretrain**: converged too easily, WORST
  embedding source (jeremy). Confirms our L2p-MLM skip.
- **Pseudo-labeling missing pIC50 from single-conc screen**: hurt badly
  (assay saturation). Using log2fc as an aux TARGET is fine; as GROUND TRUTH is
  not (jeremy). We already do the right thing.
- **TabICL/TabPFN on classical ECFP4+RDKit2D**: r~0.9 with the LGBM baseline,
  redundant (jeremy).
- **cpmed vs cpchm as "two families"**: one family, twins (our sec 16, matches
  jeremy's Mitra/TabPFN residual-corr 0.94-0.98 finding).
- **Mitra beat TabPFN** in stir_bar's full stack (one family, residual corr
  0.94-0.98) - tabular in-context learners read from the same embeddings are
  pre-correlated; they need the PCA budget + per-source legs to not crowd out
  the encoders.

## 3. THE LEVERS (ranked by expected value / cost, both tracks)

### TDI (priority 1 - the conjunction is a genuine paradigm shift we can verify)

**T1. Conjunction-decomposed TDI (A1/A2) - THE BIG ONE.**
Build, per isoform:
  - `gate_hat` = a POTENCY regression on the TDI-condition arm (predict pi_TDI,
    threshold at 4.301). Reuse the UMTM TDI-condition pIC50 head (it already
    predicts pi_TDI!) or a fresh 4-head/1-head regressor. On 3A4 this is the
    dominant signal (oracle 0.567); on 2D6 it's nearly constant-open (5.8%
    closed) so 2D6 barely needs it.
  - `shift_hat` = a regression/classifier on `Delta = pi_TDI - pi_dir`
    (threshold 0.301). This is a 1D potency DIFFERENCE - predict pi_TDI and
    pi_dir (both already UMTM heads), take the difference. On 2D6 this is the
    dominant signal (oracle 0.901).
  - `p_TDI = p_gate * p_shift` (or the conjunction of the two thresholded
    scores), then run the EXISTING fraction/posterior machinery
    (tdi_fraction_opt_v7) on the new score.
  GATE: nested-honest macro MCC on the decomposed score must beat v6b capped
    0.3448 (and per-isoform non-regression: 2D6 >= 0.219, 3A4 >= 0.470). Judge
    on Butina folds too (A5) as a sanity check. Family-block audit (the gate
    and shift models share the UMTM encoder family - paranoid-merge them and
    confirm the gain survives, like sec 22 did for UMTM).
  WHY IT SHOULD BEAT THE MONOLITH: the monolith spends 1.5k/3.5k labels learning
    a conjunction; the decomposition learns each conjunct from the SAME labels
    as a simpler (potency) problem with a known structure, and the rule is exact.
    Lizard Wizard's shipped model is exactly this and it moved their TDI into
    Tier 1. This is the single most likely TDI jump we have.
  COST: mostly reuses UMTM heads (cheap). New work = build gate/shift regressors
    if UMTM heads are insufficient, the product score, re-run fraction+audit.
    ~1-2 GPU-free days + a little GPU for fresh regressors.

**T2. Precision-constrained fraction re-tune on the new T1 score.**
Once T1's score exists, re-run tdi_fraction_opt_v7 with the board pi prior AND
an explicit precision objective: the leaders sit at macro precision 0.55-0.85.
Our v5 ships 2D6 f=0.15 / 3A4 f=0.24. If T1 improves the ROC, the E[MCC]-argmax
fraction will shift; retune and report E[MCC] and E[precision] both. The goal
is to cross macro precision 0.50 (the top-10 floor). Cost: CPU-only.

**T3. Representation diversity for the TDI blend (stir_bar's real win).**
Our TDI pool is all 2D-graph (lgbm, TabICL-on-CPM, UMTM). The one class we lack
that stir_bar/jeremy both use and we do not is a **SMILES-sequence transformer**
(ChemBERTa / MoleculeBERT / a text transformer) as a frozen-embedding TabICL
member. Genuinely decorrelated from every graph MPNN in the pool. GATE: frozen
transformer embeddings -> TabICL TDI must add family-block-honest nested MCC
(LOFO gain survives paranoid merge) before it enters the candidate. Cost: 1
embedder run (CPU/GPU ~30 min) + TabICL. This is the "new family" lever that
worked for Monroe in regression (L5, +0.002 pool) - try the analog for TDI.

### Regression (priority 2 - the cliff needs a new class or a better reader)

**R1. QHTS-ONLY domain adaptation then freeze (A4) - best 2D6 lever.**
Take jeremy's frozen `chemprop_medium.pt` (in /tmp/jeremyscripts/.../
checkpoints/), domain-adapt it on AID1851 qHTS ALONE (no ChEMBL - ChEMBL is
1-1.65 log too potent and caused our ft_pre failure), then FREEZE and embed.
This is jeremy's best-ever CYP2D6 single and we never ran the qHTS-only variant
(we ran the mixed one). GATE: frozen-DA ridge probe must beat the chmridge
floor 0.468/0.585/0.344/0.723 on >=2 isoforms (same gate as Monroe L5), then
family-block as a new family. Cost: 1 DA fine-tune (~30-60 min GPU) + ridge.

**R2. Unblock TabPFN (A3) - the field's favorite reader.**
jeremy's best single reg model is TabPFN on frozen chemprop_medium. We are
401-blocked, and the block is now understood precisely (verified Oct 4):
  - The ORIGINAL v1 weights (Prior-Labs/TabPFN-v1-reg/-clf, the token-free 2024
    release) are now GATED (HTTP 401) - Prior Labs pulled them. That was the
    source of our Sept 26 "401" note.
  - The CURRENT versions (v2, v2.5, v2.6, v3, v3.5) are NOT gated (files
    publicly listed on HF). The tabpfn package (v9.1.0) requires a FREE
    PriorLabs account + license acceptance on first use, yielding a
    TABPFN_TOKEN (from ux.priorlabs.ai). This is NOT an HF token (our notes
    were imprecise).
  - License: v2 weights = Apache 2.0 + attribution (fully permissive, even
    commercial); v2.5/2.6/3/3.5 = non-commercial licenses. Code = Apache 2.0.
    For this research/open-code/disclosed competition, the non-commercial
    license is acceptable; v2 is even more permissive. No paywall.
Action: ask Jackson to create a PriorLabs account at ux.priorlabs.ai, accept
the license, and provide the TABPFN_TOKEN (one env var). Then run TabPFN v2 on
(a) frozen chemprop_medium, (b) frozen Monroe 720-d, (c) the UMTM intermediate
embeddings. TabPFN is a genuinely different reader than our GBM/ridge/blend,
and decorrelated. If the token is not forthcoming, R2 is blocked and we fall
back to R4 (Mitra, open-source in-context learner, stir_bar found it
comparable). Cost: gated on the free PriorLabs token; then CPU/GPU inference
only (TabPFN is inference-only, no training).

**R3. SMILES-sequence transformer as a regression family (stir_bar's base).**
Same embedder as T3, frozen, -> ridge/TabPFN on the embeddings as a NEW
regression family. The pool's one missing representation class. GATE: same as
R1 (beat chmridge floor on >=2 iso, family-block clean). Cost: shares the
embedder with T3.

**R4. Dead-zone ST-RAE refit (A6) - cheap, orthogonal, metric-native.**
Take our best blend's OOF predictions, clip each to its DRC band [lo,hi], and
refit a LightGBM member under `loss=absolute_error` to that clipped target
(out-of-fold, NOT the model's own train-row predictions - that is the self-target
trap Lizard Wizard documents). Add as a 6th member, family-block. Expected
+0.01-0.02 ST-RAE (their +0.0197 rank). Cost: CPU-only, reuses existing OOF.
This is the highest ROI regression item if it works and is independent of the
representation race.

**R5. Credible-band-width model (Lizard Wizard) for the placement.**
Isotonic-regress the DRC band WIDTH from the label alone (R2 0.93-0.97 for
them) to get the per-compound band, then place spread per-compound instead of
our constant F_SPREAD. This directly optimizes the ST-RAE denominator's shape.
Cost: CPU-only. Bounded gain (~0.01-0.03) but it is the one placement lever
still open (our constant F_SPREAD is a blunt instrument vs a per-compound band).

### Explicitly NOT doing this cycle (with reason, per the negative intel)
- Full Butina-fold CV rebuild of all gates (A5): expensive, net-negative on the
  4-week clock vs making ONE new encoder work. We judge new members on both
  scaffold AND Butina, we do not re-derive every incumbent on Butina.
- More seeds of UMTM L2 (diminishing: 8 seeds already, +0.0018/seed and flat).
- Uni-Mol / 3D descriptors (DEAD, confirmed by 3 sources).
- SimCLR SSL pretrain (DEAD, confirmed by 2 sources).
- Pseudo-labeling from single-conc (DEAD, confirmed).
- Chasing placement shrink knobs (exhausted, 4 scored points).

## 4. EXECUTION ORDER (single GPU, one heavy job at a time, ~4 weeks to Nov 3)

Phase 0 (day 0, CPU): verify the conjunction on the FULL TDI table both isoforms
   (already done: 100%); build the gate/shift score from EXISTING UMTM heads
   (TDI-condition + direct pIC50 OOF preds are cached). This is a CPU-only
   prototype of T1 - if the product score already beats v6b nested, the whole TDI
   jump is nearly free. Log numbers in NOTES.md sec 23.
Phase 1 (day 1-3, TDI): T1 fresh gate/shift regressors if UMTM heads are
   insufficient; T2 fraction re-tune; family-block + paranoid audit; build
   tdi_submission_v6_candidate (8-col schema - reuse
   src/fix_tdi_submission_combined.py to attach the 4 pIC50 cols). GATE before
   any submit ask: nested macro > 0.3448 AND 2D6 > 0.219 AND 3A4 > 0.470 AND
   paranoid-merge clean.
Phase 2 (day 3-6, TDI diversity): T3 SMILES-transformer embedder (shared with
   R3); TabICL TDI member; family-block; fold into the v6 pool if it adds
   honest MCC.
Phase 3 (day 5-10, regression): R1 qHTS-only-DA (GPU) -> ridge gate; R4
   dead-zone refit (CPU); R5 band-width placement (CPU). Build
   regression_final_cp17 candidate (cp16 + surviving new members), family-block.
   GATE: nested macro R2 > 0.4569 AND per-isoform non-regression AND
   paranoid-merge clean.
Phase 4 (day 8-12, regression reader, token-gated): R2 TabPFN on
   chemprop_medium + Monroe + UMTM embeddings IF Jackson provides the free
   PriorLabs TABPFN_TOKEN;
   else R3 transformer-ridge family. Fold survivors into cp17/cp18.
Phase 5 (day 12-15): final blend/ES across everything that survived; both
   tracks family-block paranoid; build the final candidate pair; report honest
   nested numbers + the blind-transfer expectation book (nested->blind ratios:
   reg k~1.04-1.06, TDI v2 1.28 / v3-free 0.94 / v5 TBD).
Phase 6 (Nov 1-2): lock the pair, submit WITH Jackson (NEVER without his
   explicit go-ahead + HF login). Latest valid submission counts, so the final
   pair is what matters. Keep cp15/cp16 + tdi_v5 as the safe fallback.

Cadence rules (from NOTES.md, non-negotiable):
- ONE heavy GPU job at a time; background via terminal(background=true), one
  tracked process per script (foreground caps ~420s and SIGTERM's children).
- NEVER kill llama-server or any running process without Jackson's permission.
- commit AND push at every stage (git main, repo is the open-code model report).
- Every candidate: official validators + src/verify_submissions.py row-for-row
  BEFORE showing Jackson a file. New filenames; never overwrite incumbents.
- Judge on nested/honest numbers; one scored-reveal delta ~0.02 is noise.
- 1 submission per 12h; latest valid counts.

## 5. HONEST EXPECTATIONS (what each lever is worth, not hype)
- T1 conjunction (TDI): the highest-upside single bet. Lizard Wizard's
  decomposition put them in TDI Tier 1 from ordinary models. If the gate/shift
  score transfers like our v2 (nested->blind 1.28) we go 0.3448 -> ~0.40-0.45
  macro MCC, i.e. top-10. If it transfers like v3-free (0.94), ~0.33 (no gain).
  The precision story (macro precision 0.43 -> 0.55+) is where the rank moves.
- R4 dead-zone (reg): +0.01-0.02 ST-RAE, low risk, metric-native. Real but not
  a cliff-jump.
- R1 qHTS-DA / R2 TabPFN / R3 transformer (reg): each is a candidate new
  family worth +0.005-0.015 nested IF it clears the gate and the paranoid
  merge. The cliff (0.72 -> 0.78 Spearman) probably needs TWO of these to land,
  not one. Do not promise top-12 on regression this cycle; promise a real
  climb toward rank 20-30.
- T3/R3 transformer: the representation class we genuinely lack; the most
  likely source of a decorrelated jump on both tracks at once.
- The TDI jump is more certain than the regression jump this cycle. Lead with TDI.

## 6. FILES THIS PLAN TOUCHES (all new, incumbents preserved)
- New: src/tdi_conjunction.py (gate/shift/product score),
  src/tdi_submission_v6.py, src/reg_deadzone.py (R4),
  src/reg_bandwidth_placement.py (R5), src/smiles_transformer_embed.py (T3/R3
  shared), src/qhts_domain_adapt.py (R1), src/tabpfn_probe.py (R2, token-gated).
- Reuse: src/umtm_data.py (master table, TDI-condition + direct heads),
  src/tdi_fraction_opt_v7.py, src/tdi_blend_family_block4.py,
  src/regression_family_block.py, src/blendN_all.py,
  src/fix_tdi_submission_combined.py (8-col TDI schema), src/verify_submissions.py.
- Data: TDI-condition + direct pIC50 arms are in data/cyp-challenge-TRAIN_TDI.csv
  (columns CYP{2D6,3A4}_pIC50_TDI_condition and _direct_inhibition); DRC bands
  are the conf_low/conf_high columns in data/cyp-challenge-TRAIN_inhibition.csv.
- conda envs: cyp (LightGBM/rdkit, NO torch), chemeleon, chemprop-dev, monroe,
  unimol. New deps (transformer embedder, TabPFN) go in a FRESH env; never pip
  into cyp.

## 7. FIRST ACTION FOR THE FRESH SESSION
Run Phase 0 NOW (CPU-only, no GPU, no submits): prototype the T1 conjunction
score from the cached UMTM OOF predictions (TDI-condition arm + direct arm) for
both isoforms, compute nested-honest macro MCC, and report the number vs the
0.3448 bar. That single number decides how much of Phase 1 is free. Do not
touch the GPU, do not submit anything, and log the result in NOTES.md sec 23
before asking Jackson for anything.
