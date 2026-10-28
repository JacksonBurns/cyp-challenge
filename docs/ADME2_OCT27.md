# ADME2 Program — Assay-Physics Auxiliary Tasks (Oct 27, full battery)

Directive: "Add more endpoints, consider more strategies, add this to
regression and classification. Work autonomously... learning the assay
physics could be a huge unlock."

## Routes executed (user-directed list, all of them)
1. Predicted endpoints as INPUT FEATURES (A2, A2+alerts, zoo10)
2. Aux TARGETS in the MPNN (ft_adme2, 31 heads; AUX_W sweep; ablations)
3. Pretraining-style transfer (covered by ft_adme2 recipe)
4. Post-processing criteria (R6 center-shrink; gate C/R5 from v1)
5. Rule-based assertions (R5 floor rule, alerts rules)
6. Classification track (TDI classifier + ADME2 features)

## Feature side
- A2 gate (stack + 18-endpoint GBM-predicted block, GBM pIC50 MAE):
  1A2 +.0155, 2C9 +.0163, 3A4 +.0052, 2D6 -.0033. Strongest feature-side
  result of the campaign.
- Endpoint-group ablation (the science):
  - LIPOPHILICITY (logD, Sangster logP) = universal driver: carries
    80-100% of full-block gain on 1A2/2C9/3A4; largest drop-one cost
    (+.0072 1A2, +.0144 3A4). Assay partitioning/nonspecific binding
    modulates apparent pIC50.
  - DIRECT CYP polypharmacology (Veith inh + substrate, 7 cols): 1A2
    +.0095, 2C9 +.0192 (best group on 2C9); 2D6 NEGATIVE (-.0072).
  - Aqueous solubility: 2C9-specific +.0090.
  - PERMEABILITY: clean null on all four (-.0008/-.0012/+.0014/-.0021).
    Transport physics does not inform binding potency.
  - Distribution/metabolism: small (+.001-.010).
  - 2D6 resists all physics (binding-site/kinetics driven).
- Alerts block (29 PAINS/reactive/aggregation SMARTS): +0.0000 on 4/4.
  Continuous endpoints subsume interference chemistry.
- Experimental logP direct feature: 209/6145 train + 2/750 test coverage
  (3.4%/0.3%) - too sparse; predicted lipophilicity is the right proxy.
- B2 floor-hazard AUC: 3A4 .697 -> .742 (+.045, 18-endpoint block);
  detection improves monotonically with endpoints (8: .715 -> 18: .742).
- zoo10 pool pair (dscst_adme2 member): real-minus-shuf ~0. Inflation.

## Model side (ft_adme2, 31 heads)
- AUX_W sweep, seed-resolved: w1.0 .3530 | w0.3 .3299 | w0.1 .3249 (4
  seeds) | w0.02 .3268 (9 seeds) | w0 (fullft, no ADME heads) .3282.
  Best FT arm of the campaign = light physics supervision (w0.02-0.1),
  beats CYP-only FT by ~.004-.005. U-shape: heavy physics starves primary.
- Ablation at w0.3: removing 7 direct-CYP heads costs ~.001 -> ADME
  physics heads carry the arm (user hypothesis confirmed).
- Ablation at w0.02: cost ~.008 -> at light weight, Veith CYP labels
  become complementary (they add signal the physics heads stop providing).
- zoo11 pair (v1 arm): real-minus-shuf ~0 (4th control confirmation).
- zoo12 band8 pair: delta +.002/+.002/-.000/+.001 (mean +.001) - the
  label-specific signal concentrates on 1A2/2C9 = exactly the A2-gain
  isoforms. zoo12 w002 pair: ~0. band14 pair: running.

## Post-processing / rules
- R6 physics-gated center-shrink: +.0000 to +.0002 = metric-invisible.
  Mechanistic closure: floor rows sit INSIDE assay intervals
  (labels ~2.0 vs lo ~1.05) and ST-RAE charges 0 inside [lo,hi]. The
  floor is detectable (.828 AUC 2C9) but free. Post-processing family
  CLOSED with proof.

## Classification track
- TDI gate: ADME2 feats improve in-run base field MCC 2D6 +.0103,
  3A4 +.0062 (first classification signal of campaign), but best-thr
  .4024/.1273 < v4 candidate .409/.163 -> v4 keeps the TDI slot.

## Bottom line
Assay physics is REAL and mechanistically mapped (lipophilicity >>
CYP polypharmacology > solubility(2C9) >> permeability=0; 2D6 immune),
visible in features (+.016 GBM MAE), detection (.742 AUC), and the MPNN
standalone (+.004-.005 from light physics aux). At POOL level on the
ship metric, label-specific gain is ~0-.002 (6 shuffled controls): the
hinge-fit blend absorbs any zoo6-quality decorrelated member at ~+.02
regardless of labels. ADME2's pool contribution is at the edge of gate
noise, concentrated on 1A2/2C9. Not a ship-table mover by itself;
band14 is the final adjudication.
