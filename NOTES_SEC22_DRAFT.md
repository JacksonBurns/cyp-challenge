## 22. CYP-UMTM EXECUTION (Oct 3 ->) - Phase 1 code + gates, Phase 2 L2 chain

PICKUP STATE (agent 2, Oct 3 evening): predecessor landed Phase 1 code
untracked (src/umtm_data.py, umtm_module.py, cypfolds.py, umtm_eval_direct/tdi,
22 probes) then stopped before running anything. Now committed+pushed
(f9681e4..). Verified+fixed below; GPU chain running tools/run_umtm_phase2.sh
(logs/umtm_phase2_chain.log): gate_ext -> gate_primary -> L2 plan seeds 0-3.

### Master table (src/umtm_data.py) - VERIFIED, matches ft_ext lineage
- 29,129 rows: challenge 6,145 (22-head superset of ft_data's 12: direct x4 +
  tdic x4 + is_TDI x2 + log2fc x4 on challenge rows; + chembl/pubchem columns),
  ext 22,983 (chembl+qHTS union minus challenge/test collisions; per-head
  1.6k-9.6k labels; ft_ext's own chembl/pubchem rows land inside it minus the
  99 extra rows its dedup semantics kept separately - fold draws identical
  rng stream, weights identical recipe), sc_only 1.
- fold_reg seed 0 assert byte-matches ft_data fold column; fold_tdi seed 7
  stored alongside; both ride every OOF file. GOTCHA respected: built in cyp
  env (numpy 1.26.4) per cypfolds tie-break note.
- Head counts sanity: direct 1412/1285/1493/2335; tdic 1413/1285/1497/3583;
  is_TDI 2D6 1497 / 3A4 3584 (== run_tdi n's); log2fc 4375 x4; TDI-only rows
  (no direct label) 1,240 - enter the model for the first time ever.

### Module (src/umtm_module.py) - bugs found+fixed before any real run
1. OOF rows were predicted from the ES val split (491) not the fold rows
   (1229) - smoke crashed on the shape mismatch; now predicts va_idx rows
   post-training, exactly ft_ext semantics (+ assert predict_frame row count).
2. ES val_loss included bin heads in the MSE term (targets are 0/1, model
   emits logits -> is_TDI=0.19-0.33 junk). Now val_loss == training-loss
   formula restricted to ev rows: weighted z-MSE/active cells + w_bin*masked
   BCE mean; in gate modes (w_bin=0) byte-equivalent to ft_ext's val_loss.
3. Gate modes ext/primary: BCE weight 0 AND no stack module - closest
   possible reproduction of the ft_ext / ft_fullft recipes.
4. --no-test flag (OOF-only gate runs; mirrors ft_ext --skip-folds spirit).
- Unit convention per probes: ffn(Z) raw is z-space both modes; exports
  unscale via scaler buffers; bin cols identity-scaled => raw ARE logits.
- SMOKE OK (fold 0, 2 ep, plan weights, L2): every head's val MSE decreasing
  ep1->ep2; BCE heads 0.27->0.20; 52 s/fold-epoch with 13.4k rows.
- ft_ext scaler probe: chemprop normalize_targets IS nan-aware
  (StandardScaler.fit over non-NaN) - make_scaler matches; z-convention ok.

### Phase 2 L2 RESULTS (Oct 4, 4 seeds plan, ~21 min each)
- **GO on the regression track.** Direct-head singles (seed-avg and per-seed):
  best seed 0.6185 macro Pearson vs best legacy single ft_admchm 0.6059 and
  ft_fullft 0.5791 — every isoform improved (1A2 .570/.562, 2C9 .678/.657,
  2D6 .435/.417, 3A4 .791/.787). cp13 nested blend (ft_umtmL2 family added to
  cp12): **macro R2 0.4534 vs cp12 0.4471** (honest, nested; all four isoforms
  up: .613/.605, .721/.716, .486/.481, .826/.823). That +0.006 is a real
  nested-blend gain — the shipped-bar level.
- **TDI track: no-go for standalone replacement.** Seed-avg is_TDI OOF nested:
  2D6 0.149 / 3A4 0.363 vs legacy lgbm singles 0.147/0.404 and the
  already-shipped TDI blend (0.186/0.35 nested in tdi_blend_nested.json).
  Best single L2 seed hit 2D6 0.204 (s2) — inside the 0.17-0.21 user bar, but
  seed-avg washes to 0.149 (seed noise; per-seed 0.146-0.204). 3A4 ~0.35-0.37
  under the 0.38-0.44 standalone bar. Verdict: UMTM is_TDI is competitive
  juice but does NOT beat the existing lgbm/blend singles on nested eval ->
  keep legacy TDI path; UMTM enters TDI only as *blend member* material if
  later lineages add diversity (test in cp13-style TDI blend later).
- Precision-floored fractions (umtm_fraction_opt.py, seed-avg L2, pi prior
  0.08-0.25): 3A4 argmax f=0.26 E[MCC]=0.330, CONSTRAINED f=0.08 E[MCC]=0.266
  (floor costs 0.064); 2D6 argmax f=0.17 E[MCC]=0.175 but NO f meets the 0.45
  precision floor (max E[P]=0.39 at f=0.05) -> 2D6 proba quality too low for
  the floor; ships must stay on the legacy 2D6 path. (Legacy shipped
  fractions live in tdi_fraction_optima_v6.json.)

### Queued (GPU chain one-at-a-time, tools/run_umtm_rest.sh, Oct 4)
- L1 --weights ext strict gate (must reproduce ft_ext OOF within noise; L2
  gates already passed: ext 0.6034 vs 0.5759, primary 0.6087 vs 0.5791 —
  consistent +0.03 lineage-level improvement, likely data mix, see sec 20 note)
- L1 plan x4 seeds -> eval both tracks; then L3 x4; then L4 x4 (from-scratch
  D-MPNN d_h 300). Logs logs/umtm_<lin>_s*.log; chain log logs/umtm_rest_chain.log.

### Phase 1 GATES (Oct 3 night)
- GATE ext (L2 lineage, new heads zeroed, 5 folds, 24 min): macro OOF Pearson
  **0.6034 vs ft_ext 0.5759** - per-iso ALL improved or equal
  (1A2 .549/.534, 2C9 .660/.630, 2D6 .425/.378, 3A4 .780/.761). Code path
  clean: deltas are the known admecd-vs-CheMeLeon lineage effect (L2 init !=
  ft_ext's nazarov init; row pools same recipe, subsample stream differs by
  construction so 'within noise' = ordering/magnitude check, which passes
  decisively in the right direction). Strict byte-lineage confirmation run
  (L1 + ext weights) queued after the Phase 2 chain.
- GATE primary (direct-only, 9 min): macro **0.6087 vs ft_fullft 0.5791**, same
  per-iso improvement pattern (1A2 .556/.527, 2C9 .667/.640, 2D6 .422/.383,
  3A4 .789/.766). Both gates PASS: consistent, plausible lineage effect, no
  pathology, code path validated against two independent legacy baselines.
- Phase 2 (L2 plan x4 seeds) started automatically after gates, same chain.
- Per-fold epoch trace healthy: direct/tdic/log2fc/aux heads all monotone
  decreasing; BCE ~0.19-0.21 with the STACK (plan mode) vs ~0.27 untrained.

### Go/no-go baseline bar on the identical protocol (seed-7 folds, umtm_eval_tdi)
tools/legacy_tdi_to_eval.py rebuilt legacy singles OOF as CSVs:
- lgbm base:  2D6 nested 0.147 (tuned 0.153 = tdi_cv.json exact) / 3A4 0.404 (0.410)
- ft_chm:     2D6 0.155 / 3A4 0.310 (plan's "2D6 0.17-0.21" was tuned+blend-ish
  range; honest nested singles floor is LOWER - go/no-go bar: macro-nested
  >= 0.276, and 3A4 alone is the binding constraint)
- tabicl:     2D6 0.067 / 3A4 0.364; dmpnn: 2D6 0.049 / 3A4 0.265
- regression nested-honest bar: cp10 pool macro R2 0.4475 (blend_nested_all
  cp10; cp13 = cp12 + ft_umtmL2 family added for the phase-2 read).

### L1 RESULTS (Oct 4, agent 3 pickup; 4 seeds ~22 min each)
- Strict gate PASS: L1+ext weights vs ft_ext OOF = 0.5746 vs 0.5759 macro (per-iso all within +-0.018, 2D6 +0.018 the other way is seed noise) - module reproduces the ft_ext lineage on its OWN init.
- L1 regression singles: best seed 0.5999 macro vs L2's 0.6185 and admchm 0.6059. L1 (nazarov init) weaker than L2 (admecd init) singles, consistent with sec 19 (admecd full-tuned > CheMeLeon ft).
- L1 TDI heads: seed-avg macro 0.2492 (2D6 0.141/3A4 0.358) vs lgbm 0.2753. Same no-go as L2 standalone.
- cp14 nested (cp13 + ft_umtmL1 family): **0.4552** vs cp13 0.4534 vs cp12 0.4471 (+0.0081 total over cp12). L1 enters blend w/ mean weight 0.079 (max 0.333) next to L2's 0.30/0.583.
- family_block cp14 audit: LOFO[umtm_vs_adm] admecd-merge gain +0.0094 (dropping the merged admecd+UMTM family costs 0.0094; UMTM does NOT subsume admecd). LOFO[umtm_paranoid] merging L2/L3 into admecd + L1 into chemeleon + L4 into dmpnn: gain only -0.0006/-0.0006 -> UMTM families survive the paranoid merge; signal is beyond lineage re-draw. cpridge/cpall gain +0.0051 (unchanged story).
- TDI blend-member test (tdi_blend_nested + UMTM seed-avg npz members via tools/umtm_to_tdi_npz.py): BASE nested 2D6 0.1857/3A4 0.4663.
  +umtm_L2: 2D6 **0.2149** (+0.029) / 3A4 0.4648. +umtm_L1: 0.2083/0.4659. +both: 0.2203/0.4613 (macro avg 0.3408 vs 0.3260 base).
  -> UMTM is_TDI is valuable as a BLEND MEMBER (esp. 2D6) even though it loses standalone. L3/L4 members pending; then fraction retune + candidate.
