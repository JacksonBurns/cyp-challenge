# CYP Challenge - Final Report (Oct 30, 2026)

OpenADMET CYP Blind Challenge: predict pIC50 (CYP1A2/2C9/2D6/3A4; ST-RAE metric,
interval-aware + Spearman) and time-dependent inhibition (TDI) classification
(2D6/3A4; MA-MCC) for 750 blinded compounds.

## Final regression model (cp21) - CV macro ST-RAE 0.5969, board 0.5133
- **Pool (zoo4):** 32-family blend built on a LeRegT-style tabular foundation model
  (CheMeleon) full fine-tuned on the 6,145-molecule training set, plus:
  - RDKit descriptor GBM / RandomForest / HistGB stack (200+ chemistry descriptors)
  - fine-tuned family zoo: hinge-loss FT, unimolecular-transformer (UMTM L1-L4),
    TabPFN/CP-Med distillations, D-MPNN, ChemProp ridge, multiview (mv_cw0.3)
  - external-data FT arms: ChEMBL quantitative pIC50 (extw005, AUX_W=0.05,
    5-seed standalone softRAE 0.3242), ADME-physics band (ft_adme2 w0.02-0.05)
  - PT families removed (dilutive under the constrained weight fit)
- **Blend:** non-negative ridge on scaffold-OOF with ST-RAE-hinge objective,
  L1+L4 gates (bootstrap CIs) per isoform; emit via raw_test_matrix.
- **Score:** rank 51/243 (ST-RAE 0.5133, R2 0.522, Spearman 0.724).

## Final TDI model (v5) - board MA-MCC 0.3422
- Family-block-capped blend (v6b) of UMTM/multiview heads + descriptor GBMs,
  scaffold-CV OOF; submission fraction optimized as expected-MCC under the
  board prior (2D6 rate 0.149, 3A4 0.240); calibrated P ceiling ~0.4 on 2D6.
- **Score:** rank 35/130 (MA-MCC 0.3422).

## Key scientific findings (all with shuffled-label controls)
1. **Assay physics is real - in the feature channel.** Experimental lipophilicity
   (Sangster logP) + TDC ADME endpoints improve isoform GBMs seed-resolved
   (macro +0.0069 MAE; 2C9 +0.0150 4/4 seeds, 3A4 +0.0063 4/4; 2D6 immune).
   Endpoint ablation: lipophilicity universal driver; direct-CYP
   polypharmacology 2C9/1A2-specific; solubility 2C9; permeability null.
2. **Auxiliary pretraining adds ZERO to an already-pretrained pool.** 8/8
   shuffled-label controls: four label families (ADME physics, experimental
   pKa 24k, computed 3D conformational, quantitative external pIC50) show
   headline CV pool gains up to +0.032 (bootstrap CIs excluding 0) that are
   exactly reproduced with shuffled labels (real-minus-shuffled mean +0.0003).
   Pretraining + FT distillation saturates the representation channel.
3. **Metric geometry closes post-processing.** ST-RAE charges zero inside the
   assay [lo, hi]; hard/floor molecules are polar-acidic-low-logP with
   signed residuals -1.2..-2.4, detected by UQ (AUC .61-.83), but clip/LLOD
   fixes are not test-applicable (bounds withheld) and the blend predicts
   almost nothing inside the floor region.
4. **Leakage lesson:** cross-isoform residual co-structure gives +0.25-0.49
   ST-RAE from label leakage; the honest predictions-only version is ~0.
5. **Blind calibration (Q2):** board = CV - 0.099; CV-to-blind gain slope 0.66
   (CI .16-.95) - pool swaps worth <=0.014 CV are board noise. Ship = best
   validated pool, not leaderboard mechanics.
6. Chemistry forensics, docking (holo 2HI4/1R9O/4WNU/6MA8, Vina, heme-Fe
   boxes), pharmacophore geometry, assay-protocol metadata, caption-token
   distillation, PCGrad, MoE, UQ-loss: all adjudicated closed (docs/).

## Files
- `cache/regression_cp21_submission.csv` (md5 f98cd854...)
- `cache/tdi_submission_v5_candidate.csv` (md5 c8ab0626...)
- Ledgers: `docs/LEDGER2.md`, `docs/SCICOMPENDIUM_OCT25.md`,
  `docs/CHEMFORENSICS_OCT26.md`, `docs/ADME2_OCT27.md`, `docs/REPRODUCTION.md`
