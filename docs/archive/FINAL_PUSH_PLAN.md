# THE BIG SWING: CYP-UMTM (v2 plan, Oct 3)
# Unified Multi-Task, Multi-View Model - one model to reach for the top of both boards

Supersedes the v1 lever-ranked plan (kept in git history). Jackson's directive:
no incrementalism, no self-imposed resource ceilings - consolidate every
disclosed and derived insight into the single best model we can build in the
~4 weeks before Nov 3. All existing incumbents (cp7/cp10 regression, v2/v4 TDI)
stay as fallback + revert targets; UMTM either overtakes them on honest gates or
its members fold back into the legacy pool. No regression in shipped quality,
ever.

## 0. THESIS (why one model, not the 20-member blend pool)

Every diagnostic we have points the same direction:
1. Regression gap to the top is 100% ranking (Spearman 0.709 vs 0.75-0.78);
   placement is near-optimal and confirmed so on both scored reveals.
2. TDI leaders are NOT regression-ranker transfers (TDI rank-1-adjacent entrants
   exist at regression rank 212). They exploit TDI-specific signal: our TDI
   binaries train on ~1.5k (2D6) / ~3.6k (3A4) labels, alone, from encoders that
   never saw the other views of the same molecules.
3. The data is a multi-view superstructure: ALL 4,905 regression molecules also
   carry TDI-condition pIC50 (corr 0.90-0.95 with direct but NOT deterministic
   of is_TDI); 4,376 molecules carry single-conc log2fc (Spearman -0.83..-0.94
   vs direct pIC50); TDI train is a superset of inhibition train; ~50k external
   ChEMBL/qHTS molecules carry the same-kinase-family potency signal.
   The blend pool attacks these views separately and merges OOF predictions;
   shared-representation learning exploits cross-view correlation DURING
   training, which is exactly the regime (small labels, rich aux views) where it
   wins. jeremy's 17-head multitask being "the single largest ensemble
   contributor" is external evidence the direction pays.
4. Our best-ever singles came from the most label-sharing recipes (ft_ext 12-head,
   admecd full-tunes). The fully-shared version has never been trained.

BIG SWING: one encoder recipe, ~23 NaN-masked heads spanning every label that
exists anywhere in this competition, multiple encoder lineages, multiple seeds,
an in-model stacked TDI head, calibrated with the proven placement + precision-
constrained operating points. Target: regression Spearman >= 0.75 (ST-RAE
<= 0.43, top-10 band), TDI macro MCC >= 0.45 (top-5 band).

## 1. ARCHITECTURE

### 1.1 Master table (src/umtm_data.py)
One row per unique SMILES across ALL sources; NaN = not measured.
- challenge_inhib (4,905): direct pIC50 x4
- challenge_TDI (6,145): is_TDI x2 (2D6/3A4) + TDI-condition pIC50 x4
- single_conc (4,376): log2fc x4 (pivot enzyme-wide, median of replicates)
- external ChEMBL CYP (Capricho-style: standard_relation '=', units nM,
  median per mol x isoform): pIC50 x4
- external PubChem AID1851 (Fit_LogAC50): pIC50-like x4 (1A2/2C9/2D6/3A4)
- test (750) rows for prediction only.
Folds: Murcko scaffold-grouped 5-fold, seed 7 (run_regression.scaffold_groups
conventions, incl. __NONE__ bucketing). Same fold vector used for EVERY head.
External rows never gate validation metrics; they train aux heads only.
The TDI-only 1,240 molecules train the TDI + TDI-condition heads with no
primary label - first time they enter any model at all.

### 1.2 Heads (23 total), single FFNModule over shared MP state
| group | heads | loss | weight |
|---|---|---|---|
| primary direct pIC50 | 4 | MSE | inverse-count, mean-normalized to 1.0 |
| TDI-condition pIC50 | 4 | MSE | 0.5 |
| is_TDI binary | 2 | BCE | 0.5 (see 1.4 stack) |
| single-conc log2fc | 4 | MSE | 0.3 |
| ChEMBL pIC50 aux | 4 | MSE | 0.3 |
| qHTS pIC50 aux | 4 | MSE | 0.3 |
Mixed-loss implementation: custom LightningModule around chemprop
MultiTaskFFN (per-cell valid mask, per-task loss fn, per-task weight; ~80 lines
on top of ft_ext.py which already does NaN-masked MSE multi-task). Smoke test:
one fold, one epoch, loss curves monotone per head group before full runs.

### 1.3 Encoder lineages (identical head recipe, run as siblings)
- L1 CheMeLeon MP d_h 2048 (~/chemeleon_nazarov/chemeleon_mp.pt), full fine-tune
- L2 adme_pretrain chemprop_chemeleon (adapt_adme_ckpt.py -> admecd lineage;
  best singles we ever measured)
- L3 adme_pretrain chemprop_medium
- L4 from-scratch D-MPNN d_h 300 (diversity anchor; population-match insurance)
- L5 (stretch, gated): Monroe GRIT (torch-geometric; GO gate = weights download
  + frozen TabICL probe beating the chmridge singles floor 0.468/0.585/0.344/
  0.723 on >= 2 isoforms). If gate fails, log and proceed without.
Optional variant L2p: adme_pretrain + 2-epoch MLM continued pretraining on all
~15k unlabeled challenge+external SMILES before fine-tune. Run AFTER L1-L4 are
training (it is the one bet against jeremy's SSL-negative evidence, so it gets
slot priority last, and we keep its result regardless).

### 1.4 In-model stacked TDI head (the TDI-specific big lever)
is_TDI is NOT a threshold of TDI-condition pIC50 (best acc 0.78) but potency
distributions separate (positives mean 5.19/5.28 vs 5.01/4.55). So:
  p_TDI = sigmoid(W[cat(h_enc, stopgrad(pIC50_direct_hat), stopgrad(pIC50_TDIcond_hat))])
Binary head consumes the encoder state PLUS detached primary-head outputs.
No train-time target leakage (predictions, not labels; stop-gradient so binary
loss cannot poison the potency heads). This is the intra-model version of the
cross-track transfer Jackson's idea (2) was reaching for.

### 1.5 Operating points & calibration (submission layer, knobs unchanged)
- Regression: per-isoform Caruana bagged ensemble selection across seeds x
  lineages on fold-nested OOF (jeremy's ES-over-NNLS finding), then affine
  placement onto BLIND_MOMENTS shrunk by OOF Pearson (the technique that has
  worked at every reveal; knobs F_SPREAD/OOF_TO_BLIND/BLIND_MOMENTS UNCHANGED).
- TDI: probabilities averaged over seeds, then EXPECTED-MCC fraction search
  under a precision floor: maximize E[MCC] subject to E[precision] >= 0.45
  (board 4: MCC-vs-precision corr 0.94 at fixed recall; nothing above 0.4 MCC
  ships precision below 0.43). Positive-rate prior stays v6's.

## 2. EVALUATION GATES (ambition does not mean less honesty)
- Everything on scaffold folds seed 7; nested/honest numbers only.
- Regression gate: fold-nested macro R2 >= cp10's 0.4475 to claim default;
  family-block audit treats UMTM as families by LINEAGE (L1 CheMeLeon lineage
  merges with legacy emb/ftchm family; L2/L3 with admecd; L4 with dmpnn) so we
  learn whether UMTM is strictly better or just re-correlated. If UMTM-as-blend
  loses to legacy, fold surviving UMTM members into the legacy pool under the
  same caps and ship that.
- TDI gate: capped fold-nested macro MCC >= 0.35 (v4 machinery ~0.328-0.343 +
  visible headroom) with per-isoform non-regression, then the precision-floor
  fraction layer. UMTM candidate takes the next submission slot (Jackson's call).
- Every candidate: both official validators + verify_submissions.py row-for-row;
  new filenames; incumbents never overwritten.
- One scored-reveal difference is noise (~0.02); decisions ride nested gates.

## 3. EXECUTION (start now; single GPU but jobs queue, they don't parallelize)

Phase 1 (Oct 3-5, code): umtm_data.py master table + head spec; mixed-loss
LightningModule; smoke test 1 fold 1 epoch; per-head sanity (primary heads match
ft_ext single-head OOF within noise when aux weights are zeroed).
Phase 2 (Oct 5-7, GPU chain A): L2 (admecd lineage) x 4 seeds - the lineage with
the best prior - full OOF + test preds, both tracks evaluated the same night.
First go/no-go: does unified multi-view TDI beat standalone tdi members
(2D6 best single 0.166-0.206, 3A4 0.38-0.44)? If yes, this validates the whole
thesis; if no, still finish (regression side + stacking head may carry it) but
flag it to Jackson immediately.
Phase 3 (Oct 7-9, GPU chains B/C): L1 x 4 seeds, L3 x 4 seeds. Same eval.
Phase 4 (Oct 9-11): L4 x 4 seeds; L5 Monroe go/no-go; L2p MLM variant if time.
Phase 5 (Oct 11-14): blend/ES across everything that survived, family-block both
tracks, build candidates: regression_umtm_v1, tdi_umtm_v1 (+ fraction floor
retune for whichever TDI file ships). Ask Jackson for slot(s).
Phase 6 (Oct 14-20): 4 more seeds for the winning 1-2 lineages ONLY if nested
still climbing; re-verify; final reports (open-code requirement: repo stays the
model report).
Oct 28: lock final pair, submit with Jackson; revert default = cp10 + best
legacy TDI file.

## 4. WHAT THIS BETS AGAINST (stated honestly)
- Heterogeneity over uniformity: our legacy pool is diverse families; a single
  recipe x seeds could underperform the greedy blend even if each member is
  better. Mitigation: the fallback in 2 is real (UMTM members compete in the
  existing pool) and lines 3.x are cheap enough to build either way.
- jeremy's freeze > fine-tune finding conflicts with our full-tune wins. UMTM
  keeps BOTH regimes (L4 from-scratch full-tune vs L5 frozen probe) so the data
  decides, not the lore.
- BCE+MSE mixing can destabilize (loss scale imbalance); smoke gate 1.2 catches
  it; fallback = two-stage (freeze encoder from regression-only UMTM run, train
  stacked TDI heads on top) which is strictly weaker-but-safe.
- The TDI head stack could latch onto the assay-construction correlation
  between pIC50 and is_TDI and miscalibrate on the blind's enriched population.
  Mitigation: fraction layer is fitted under the pi prior, and we can ship the
  unstacked binary head variant if its nested MCC is within 0.01 (prefer the
  unstacked head for robustness, per that tie-break).

## 5. CEILING
Regression: the top band (Spearman 0.75-0.78, ST-RAE 0.38-0.43) is plausibly a
placement-corrected strong multi-view ranker with the population-matched
external aux data we already hold - we have every ingredient, cp10 lineage is
only +0.01 behind on OOF with a weaker recipe. TDI: 0.45+ requires the multi-
view + stacked head to work, which is genuinely uncertain; the floor is the
legacy v4 candidate. This is the maximum-effort play on both fronts with a
safe-revert spine.
