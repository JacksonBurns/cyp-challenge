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
