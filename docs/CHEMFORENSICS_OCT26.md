# CYP Challenge - chemistry forensics: LLOD-floor and hard-to-predict molecules (Oct 26)

## 1. Who falls at the floor?

Definition used: FLOOR = label at/below the 1.5th percentile of that isoform's
direct_inhibition labels (cuts: 1A2 2.07, 2C9 2.49, 2D6 2.04, 3A4 1.98).
The pIC50 labels themselves are left-censored: min ~1.9-2.1, zero rows <=1.5
(assay reporting floor ~10 nM; Emax axis floors at -1.2). Only ~1-1.5% of
rows per isoform (20-36 rows) sit at the floor; pan-floor molecules are rare
(101 ISO_FLOOR, 25 PAN_HARD, 0 PAN_FLOOR at 2+ isoforms).

Chemistry of FLOOR molecules (vs dataset mean):
- LOW logP (2.43 vs 2.81), HIGH TPSA (82 vs 69)
- 20% carboxylic acid (vs 3% background); 45% COOH in 2C9 floor rows (2% mid)
- 45% hydroxamic acid in 2C9 floor rows (2% mid) - classic pan-assay
  aggregator / metal-chelator
- FLOOR 2C9/3A4: acid_centers 0.55/0.44 vs 0.19/0.27 mid
- FLOOR 2D6: aliphatic amine 0.09 vs 0.45 mid (2D6 needs basic N; its
  absence = true inactivity), stereo centers 0.17 vs 0.83 (flat molecules)
- FLOOR 1A2: long alkyl chains 0.41 vs 0.29, pyridine 0.45 vs 0.31
- signed residual (y - P) = -1.2 to -2.4: the blend systematically
  OVER-predicts floor molecules by 1.2-2.4 log units

Interpretation: floor molecules are polar, acidic, flat, amine-poor -
"true junk" (non-specific binders/aggregators, efflux substrates, PAINS-like
hydroxamates) whose measured pIC50 ~1-2 is largely NOT structure-derived.
It is solubility/aggregation/assay-interference chemistry. The model treats
them as normal low-potency points and extrapolates chemistry-based potency
into them, producing 1-2 log unit over-predictions.

## 2. Who is hard to predict? (|blend residual| >= 1.0)

Counts per isoform (train rows): 1A2 234/1412, 2C9 73/1285, 2D6 203/1493,
3A4 234/2335. Signed residual ~0.0: HARD is two-sided (over AND under),
NOT a low-end phenomenon (unlike FLOOR).

Per-isoform HARD vs EASY contrasts:
- 1A2: HARD = high TPSA (72 vs 63), low logP (2.8 vs 3.3), more N, LESS
  aromatic (2.48 vs 2.66 fused rings), more rotatable. 1A2 is a flat,
  planar, aromatic-preferring pocket; HARD 1A2 molecules are flexible,
  polar, 3D - out of the substrate archetype.
- 2C9: HARD = high TPSA (77 vs 71), low logP, fewer rings; aniline/urea
  depleted. 2C9's chemistry is small, well-behaved (fewest HARD, 73);
  its hard cases add polarity.
- 2D6: HARD = HIGH MW (360 vs 347), high logP (3.3 vs 3.1), MORE aromatic
  rings (2.69 vs 2.35), more rings, more halogens. 2D6 HARD = big,
  lipophilic, polycyclic - the classic 2D6 promiscuity surface; potent
  2D6 binders are a tight pharmacophore family (basic amine + aromatics)
  and the model's ceiling cases are the lipophilic mass outliers.
- 3A4: HARD = LOW MW (339 vs 352), LOW logP (2.3 vs 2.9), high TPSA -
  anti-3A4 (3A4 loves large lipophilic molecules); small polar molecules
  with high measured 3A4 potency are the surprises.

Structural-alert gaps (HARD vs EASY, modest): 2C9 urea 0.08v0.19,
3A4 sulfonamide 0.11v0.19, 1A2 amide 0.73v0.59, 3A4 reactive alerts
0.16v0.09 (aldehyde/Michael/alkyl halide - covalent-ish liabilities).

Scaffold analysis: HARD rate 11% base; phenyl (17%), quinoline (20%),
1,4-dihydropyridine-phenyl (50%), anilides 25-67% in small counts. No
dominant toxic scaffold - hardness is diffuse, not scaffold-clustered.

## 3. What makes them hard for the model AS-IS?

Feature stack: ~200 RDKit 2D descriptors + 3072 Morozov autocorrelations
(CM/MOrd/PEOE/VSA/BCUT2D flavors) + Morgan densities. Verified present:
fr_ alerts x85 (coarse), PEOE partial charges, Kappa, stereo COUNTS.
Verified ABSENT: pKa, solubility (no ESOL), reactivity alerts beyond 85
coarse fr_, pharmacophore/geometric distances, 3D conformation, H-bond
acid/base strength, tautomer/enumeration, any assay-protocol features.

Mechanistic mismatches:
1. FLOOR rows: label is assay-physics (aggregation, solubility,
   interference), not target chemistry. No chemistry-trained regressor can
   predict "this molecule gives a flat curve at 10 uM" from a
   potency-trained latent space. MSE fits their chemistry to mid-potency
   region => systematic +1-2 over-prediction.
2. 2D6: needs (basic-N, aromatic) 3D pharmacophore geometry; stack has
   only counts. The known 2D6 resistance survived 10 arms (Q4: label noise
   normal, so feature-side confirmed).
3. 1A2/3A4 hardness = archetype deviation (planar-aromatic vs polar-flex;
   lipophilic-large vs small-polar). Autocorrelation descriptors are
   composition-weighted, not 3D-shape-aware; they underweight conformation
   and pocket-fit.
4. Hard rows are NOT far from training actives (AD: HARD median
   max-Tanimoto 0.45-0.53 >= EASY 0.39-1.0): near-neighbor chemistry,
   far from labels' regime. Not an applicability-domain distance problem
   - it is a label-geometry problem: local chemistry does not determine
   these labels (floor physics, 2D6 pharmacophore geometry).

## 4. Gates: can the proposed descriptor block fix it? (honest CV)

- GATE A: block (pKa proxies, acid/base centers, isoform motifs, reactivity,
  stereo, flatness, 2D6 pharmacophore flags) alone predicts HARD membership:
  AUC 0.48-0.56 = chance.
- GATE B: block added to GBM on the 3456-stack: MAE delta -0.006..+0.002
  (~zero, 4 isoforms).
- Latent OOD arm (frozen CheMeLeon geometry vs UQ): ~0 correlation.
- UQ loss reweighting: gate fail.
- Learned MoE TDI-gating: alpha=0 selected, 0/5 folds.

Conclusion: the hardness is NOT recoverable by descriptor augmentation.
The signal that distinguishes FLOOR/HARD lives in assay-physics
(solubility/aggregation/protocol) and 3D pocket-fit geometry that this
2D+autocorr stack cannot be patched to express, and the labels at the
floor are partially non-chemistry by construction.

## 5. Recommendations (ranked by expected value)

1. FLOOR censoring is the one concrete, model-free lever: floor rows have
  signed residual -1.2..-2.4. A floor-hazard head (P(y <= floor-cut) from
  chemistry: acid_centers, logP-low, TPSA-high, hydroxamic/COOH flags)
  that SHRINKS predictions toward floor when hazard is high would fix the
  worst systematic bias. (Gate test: build hazard head, CV MAE on floor
  rows; if hazard AUC >=0.7 for FLOOR-membership, ship as post-process.)
2. 3D/pharmacophore: a real 3D pipeline (ETKDG conformers + pocket
  pharmacophore-distance features for 2D6 (N-arom distance) and 1A2
  (planarity) is the only route to 2D6 resistance; expensive, unproven
  within the deadline.
3. Assay-protocol covariates (plate/batch fields exist in
  single-concentration data, not on the pIC50 tables) - would explain
  floor physics if available; they are NOT in the challenge pIC50 tables,
  so not exploitable on test.
4. Do NOT ship: descriptor-block augmentation (gate B ~0), HARD-membership
  routing (gate A ~0.5).

## Files
- src/chem_profile.py / cache/chem_groups.csv - groups + RDKit profile
- src/chem_hard2.py / cache/chem_hard2_out.txt - per-iso contrasts,
  scaffolds, applicability domain
- src/chem_mech.py / cache/chem_mech_out.txt - signed residuals, SMARTS,
  2D6 pharmacophore, label ambiguity
- src/chem_block.py / cache/chem_block_out.txt - motif block + gate A/B
- tools/floor_census*.py - floor pileup census per label family
