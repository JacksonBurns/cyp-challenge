# FINAL PUSH PLAN (written Oct 3, after board 4) - target: top of both leaderboards

Deadline: Nov 3 2026 final. Model work stops Nov 1 (rule); realistically the last
submission pair should be built by Oct 28-30 so verification does not race the clock.

## 1. WHERE WE STAND (board 4, Oct 3)

- Regression: cp7 on board, rank 58/255, ST-RAE 0.5366, R2 0.496, Spearman 0.709.
  Top 1-2 (preheat-to-450, wbot): ST-RAE 0.377, R2 0.64-0.66, Spearman 0.777-0.782.
  Top-20 band Spearman floor: ~0.75.
- TDI: v2 on board, rank 78/147, MCC 0.283. Top: TeamPozeSCAF 0.535, nova 0.513.
  Our precision is the binding constraint: at our recall band (0.50-0.65) board
  MCC correlates 0.94 with precision; leaders median P=0.506, we are P=0.368.
- Unsubmitted-but-built candidates that dominate incumbents on honest OOF gates:
  - Regression cp10 (nested 0.4475 vs cp8d 0.4403, cp7 0.4385; wins all 4 isoforms;
    family-block clean). Correlation cp10 vs cp7 test preds: 0.976-0.997.
  - TDI v4_candidate (capped v3, fractions 0.33/0.26; E[MCC] posterior ~0.286-0.36).
    NOTE: our last two scored TDI files had P/R = 0.368/0.572 and 0.359/0.672 -
    the board keeps docking us exactly the precision that the fraction machinery
    says to trade away. See section 3, lever D.
- The cp10 slot question is open: cp10's blind delta vs cp7 is ~+0.01 (both blind
  scores of cp7: 0.5192 then 0.5366 on the same file across drifting boards).
  Field drift has FLATTED since Sep 26 (top-10 deltas +/-0.014 median ~0), so the
  old "+0.07 drift" panic is over. Recommendation: cp10 does not need its own
  slot; keep it as the Nov 1 default, spend any mid-course slot on TDI.

## 2. WHAT BOARD 4 TELLS US (read this before arguing about levers)

1. The board is now STABLE. Top-10 regression entries moved <0.02 ST-RAE since
   Sep 26 on unchanged files. Gates are honest again; absolute OOF deltas translate.
2. Regression ranking is the ONLY gap. Our R2 (0.496) ~= implied-by-Spearman;
   placement is near-optimal (confirmed again by the sec 17e probe). Closing
   0.709 -> 0.75 Spearman is worth roughly -0.10 ST-RAE. No member-sized idea
   (each ~+0.005-0.01) gets there. We need a step-change encoder or a step-change
   label source.
3. TDI top-12 cross-board check: several leaders are mediocre at regression
   (NIPERKOLKATA_PI_LAB is rank 212 in reg with MCC 0.475 in TDI). So TDI leaders
   are NOT riding a strong pIC50 ranker; they have TDI-specific training signal
   we do not. Whatever they found is in the TDI data itself or external TDI data.
4. TeamPozeSCAF, NIPERKOLKATA_PI_LAB, TeamPrescience, nova, N283T all made their
   leap in the last week. The winning recipes are circulating, undisclosed.

## 3. THE LEVERS, RANKED

### Lever A (biggest TDI bet, days not hours): multitask head sharing across tracks
This is Jackson's idea (2) and it is right for a sharper reason than correlation
exploitation. Data facts I verified from the raw CSVs on Oct 3:
- TDI train (6145 mols) is a SUPERSET of the regression train (4905): all 4905
  direct-pIC50 molecules also have pIC50-under-TDI-condition columns.
- TDI-condition pIC50 correlates 0.90-0.95 (Pearson) with direct pIC50 on shared
  rows. The is_TDI binary correlates only weakly with TDI-condition potency
  (best single-threshold accuracy 0.78, and TDI-positive mean pIC50 5.19-5.28 vs
  negative 5.01-4.55). So the binary is NOT a deterministic function of the
  TDI-condition pIC50 - there is real orthogonal signal, but the two views of the
  same molecule (direct potency, TDI-condition potency, TDI flag) share most of
  their chemistry.
- Concretely: one encoder, heads = {4 direct pIC50, 4 TDI-condition pIC50,
  2D6-is_TDI, 3A4-is_TDI}, NaN-masked, challenge rows + external aux heads as
  ft_ext already does. The TDI heads train on molecules that ALSO carry
  regression labels (and vice versa) - this is genuine multi-view learning, not
  just more features.
- Why it may be the leaders' trick: our TDI deep members (tdi_dmpnn, CheMeLeon
  binary ft) are the weakest family in the pool and the cp/emb heads are
  ranking-transfers from regression. Nobody in our pool has ever been trained
  with pIC50 and is_TDI as simultaneous tasks.
- Implementation: extend src/ft_ext.py COLS to include the two binary heads
  (mixed MSE+BCE loss - chemprop FFNModule supports per-task loss via
  `MixedDataLoader` task types; if that fights back, write a small custom
  lightning module: it is ~40 lines on top of ft_ext). For TDI OOF, key on the
  TDI-file molecule order exactly like run_tdi.py does (dedupe/reindex gotchas
  in HANDOFF.md still apply).
- Evaluation gate: TDI family-block (tdi_blend_family_block2 pattern) vs the v3
  capped pool. Bar: +0.02 capped nested MCC macro, and it must be a NEW family
  (it shares the corpus lineage with cp/admecd, so run the paranoid map too).
  Ship nothing that regresses per-isoform.

### Lever B (same architecture, both directions): regression heads get TDI-condition aux
Mirror of A for the regression track: add pIC50_TDI_condition columns as aux
heads on the ft_ext/admecd recipe (they are extra potency measurements on ~4500
of the same molecules; corr 0.9+ but not identical - the TDI-condition assay adds
preincubation chemistry). Cheap: it is a head-list change, then 2 seeds x 35 min.
Gate: family-block honest nested, bar +0.005 macro R2. Lower ceiling than A but
attacking the 0.04-0.10 Spearman gap by any real amount moves rank 58 -> 20s.

### Lever C: unlabeled pretraining (Jackson's idea 1) - DEMOTE to opportunistic
Evidence against investing now:
- jeremy's writeup: SimCLR-style contrastive pretraining on ~34k unlabeled
  molecules "converged too easily to be useful" - worst embedding source they
  tried (their challenge unlabeled pool is the same single-concentration screen,
  4376 extra molecules, all of which also have log2fc values - i.e. there is
  almost nothing truly unlabeled beyond the log2fc itself).
- Our own ft_pre result (sec 12): population shift made external pretrain-then-ft
  weak; jeremy's adme_pretrain fine-tune worked (cp10) precisely because the
  PRETRAIN population already contained CYP-labeled data.
- The 1240 TDI-only molecules and 4376 single-conc molecules are the only
  in-challenge unlabeled-ish pools, and both already enter training through
  Lever A/B head masks, which is strictly better than SSL.
- If anything: MLM-style continued pretraining of the D-MPNN on all ~11k unique
  SMILES (challenge + external, no labels) for a few epochs before ft_ext-style
  fine-tune is a 1-day experiment, but priority is BELOW A/B/D and only when the
  GPU is otherwise idle.

### Lever D (TDI operating point - cheap, do immediately): stop buying recall
with precision we do not have. History: shipped v2 at 0.08 fraction scored
P/R = 0.408/0.537 then 0.359/0.672 across reveals; v4_candidate moved fractions
to 0.33/0.26 predicted-positive share, and its expected-MCC posterior assumed a
board positive-rate prior we cannot verify. Board 4 says the MCC-vs-precision
correlation at fixed recall is 0.94 and EVERY entry above MCC 0.4 has P >= 0.43.
Action before any new model lands: re-run tdi_fraction_opt_v6 with a precision-
aware objective - maximize E[MCC] under a constraint P >= ~0.45 (not just argmax
E[MCC], which wanders into high-recall territory whenever the pi prior widens),
using the capped v3 OOF pool. This costs 30 min CPU and produces the TDI file
for any mid-course slot.

### Lever E: field intel - mine the new leaders (0 GPU cost, do in parallel)
- rasayan-labs now has a real repo (was empty in sec 17) and scored both tracks.
- jeremy's open kit (now downloaded at /tmp/jeremyscripts/CYP_Challenge)
  documents levers we have NOT tried: (i) the 17-head multitask with log2fc aux
  heads (partially ours; ours lacks the 4 log2fc heads - fold into Lever A
  since the single-conc wide table pivot is ~15 lines); (ii) Monroe frozen +
  TabPFN as a decorrelated contributor (separate project + torch-geometric;
  GO/NO-GO gate on weights download like Uni-Mol; note Uni-Mol frozen ridge was
  dead for US, but Monroe's GRIT-on-quantum/PubChem pretrain is a different
  lineage - only worth it if a TabPFN/TabICL head on its embeddings beats
  chmridge singles 0.468/0.585/0.344/0.723 on at least two isoforms);
  (iii) "domain-adapting chemprop_medium on qHTS only, then freeze" was their
  best-ever CYP2D6 single - we have never tried qHTS-only domain adaptation
  (our ft_pre mixed ChEMBL+qHTS and got population shift; this recipe is
  qHTS-only and freeze-only - cheap ft_ext variant, ~1 h).
- TeamPozeSCAF TDI report link is a Google Doc that was accessible in Sept -
  re-fetch for their updated (0.535) recipe if reachable.

### Lever F: submission mechanics - the two-point cp7 sample says the truth
cp7 blind scored 0.5192 (Sep 26 board) then 0.5366 (Oct 3 board) - SAME FILE, so
board-side variance is ~0.02 per reveal and our "cp7 -> top-20 needs +0.10" gap
holds. But it also means: each reveal is ONE noisy draw; do not over-fit decisions
to single board moves (sec 14/16 lessons). Ship the pair that dominates on honest
nested gates, and ship it EARLY enough to still have a revert option.

## 4. EXECUTION ORDER (calendar to Nov 1)

Oct 3-4 (no GPU, CPU only):
1. Lever D: precision-constrained fraction retune on capped v3 -> tdi_v5_frac
   candidate file, verify. (Slot decision stays with Jackson; recommend the TDI
   slot for the FIRST genuinely new TDI model, not this retune, unless nothing
   else lands.)
2. Lever E: fetch rasayan-labs + TeamPozeSCAF doc; log any recipe deltas into
   NOTES.md sec 20.
Oct 4-6 (GPU, sequential - one heavy job at a time, check nvidia-smi, never touch
llama-server):
3. Lever A build: ft_multiview.py = ft_ext + {2 TDI binary heads, 4 log2fc heads,
   4 TDI-condition pIC50 aux} - write, smoke (1 fold, 1 epoch), then 2 seeds full.
   ~3 h GPU total background chain (tools/run_multiview.sh pattern).
4. Evaluate TDI heads standalone + blend as new family in family_block2 pattern;
   gate per section 3A. If PASS: build tdi_v5_multiview candidate; this takes the
   next slot (ask Jackson).
Oct 6-8 (GPU):
5. Lever B: same script, regression-side eval with TDI-condition aux heads added
   to ft_ext/admecd recipe; cp13 pool vs cp10; gate +0.005 macro nested; if pass,
   cp13 becomes the Nov 1 regression default.
6. Lever E(iii): qHTS-only domain-adapt freeze -> embeddings -> TabICL/LGBM
   member for both tracks (their 2D6 claim targets our weakest head).
Oct 8-15 (GPU, in order, each with its own gate):
7. Monroe GO/NO-GO (weights download gate FIRST, then frozen embeddings +
   TabPFN/TabICL single probe vs chmridge floor).
8. Only if 3-6 all fail to clear gates: Lever C MLM-continued-pretrain experiment
   (1 day, capped).
Oct 15-25: buffer + seed averaging of whatever passed (2 more seeds of the winning
multiview recipe = +0.002-0.005 honest filler), final re-verification, and the
report/commit hygiene for the open-code requirement.
Oct 28: lock and submit final pair; keep cp10 + best TDI file as revert default.

## 5. GATES AND NON-NEGOTIABLES (unchanged)
- Scaffold folds seed 7; nested/honest numbers only; family-block audit for every
  new pool (merged/split/paranoid maps); no candidate overwrites incumbents; both
  official validators + verify_submissions.py row-for-row before showing files.
- Placement knobs (F_SPREAD/OOF_TO_BLIND/BLIND_MOMENTS) UNCHANGED.
- External data for training/ranking only, never for placement moments.
- One GPU job at a time; background chains with notify_on_complete; never kill
  llama-server.
- Submit only with Jackson's explicit go-ahead (his HF login, 12 h cooldown,
  latest-valid-counts rule).
- NOTES.md section 20+ logs every step with honest numbers; commit per stage.

## 6. REALISTIC CEILING HONESTLY STATED
Lever A is the only idea on this board with a plausible +0.05-0.15 path on TDI
(it is also the cheapest deep idea left). Regression has no identified single
lever to reach 0.75 Spearman; stacking A/B/E members is a +0.02-0.04 program,
which lands us ~rank 20-35. If a top-5 regression finish is mandatory, the
missing ingredient is whatever preheat-to-450/wbot/SVM-class entries are using
that we cannot see - the disclosed evidence (SVM-class and random-forest entries
scoring 0.43-0.45 with presumably heavy external data + placement) says: more and
better-matched external potency data with careful aggregation is the remaining
regression lever (Capricho-style curation of ChEMBL per jeremy's reference),
which is Lever E-adjacent and could be added as extra aux heads into Lever A
rather than a separate project.
