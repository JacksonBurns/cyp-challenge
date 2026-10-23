# Mining mid/LOW-ranked open-code CYP entries (rank 55–235)

Mined 2026-10-22. Sources cloned to /tmp/mining/ (jeremy, knimest, talindrew, adlvdl,
magicano, sc38, ajogalekar) + HTML fetches (dargason page, Nomad HF report).
Our row: rank 51 — ST-RAE 0.5133, R2 0.5219, rho 0.7245. Target: rank 30 = ST-RAE 0.4512.
Field fact anchoring (b): 'SVM' = rank 50, rho 0.7665, R2 0.494 — pure-ranking-optimized.

---

## Per-entry dossiers

### 1. jeremy (rank 55) — `jeremycheminf/openadmet_scripts/CYP_Challenge`
- **Architecture:** LightGBM on ECFP4+RDKit2D; TabPFN on frozen chemprop_medium (D-MPNN
  pretrained on public CYP/ADME) embeddings (best single); TabPFN on frozen chemprop_chemeleon
  embeddings (decorrelated cluster); 17-head multitask chemprop fine-tune warm-started from
  chemprop_medium (4 primary pIC50 + 4 log2FC + 4 ChEMBL + 5 PubChem qHTS heads) — "consistently
  the best-ranking chemprop-family model and the single largest ensemble contributor".
  Combined by **Caruana bagged ensemble selection** (explicitly chosen over NNLS: NNLS
  "destructively reallocates weight" on blind data).
- **CV:** Butina-fold CV. Kit OOF macro ST-RAE **0.614** (full research ensemble 0.604;
  mean-predictor floor 1.0).
- **Calibration/placement:** THE key move. First submission placed on *their own* OOF
  population: blind **0.7247, CYP2D6 R2 −0.749**. Rescaled final predictions onto
  **BLIND_MOMENTS** (blind-population mean/sd reverse-engineered by SuperCowPowers from their
  scored rows; their own pop mean pIC50 ≈4.8 vs blind ≈3.1 on 2D6): blind **0.5214, CYP2D6
  R2 +0.377**. Pure recalibration, ranking unchanged. Shrinkage by own OOF Pearson rho.
- **External data:** PubChem AID 1851 (NCATS qHTS, ~16.5k, Fit_LogAC50) — chosen because
  ChEMBL CYP data is med-chem biased, "~1–1.65 log units more potent" than the challenge
  diversity-library population; ChEMBL 37 pull; adme_pretrain checkpoints.
- **Stated score:** blind MA-ST-RAE 0.5214 (board). "Freeze, don't fine-tune" — fine-tuning
  on ChEMBL-pull labels consistently hurt (population mismatch).
- **Why they are where they are:** right models + right ensemble + blind-moments placement;
  ceiling is OOF 0.614 on their own pop — they lack the selection/censoring label fix, so
  their ranking signal (and 2D6) is weaker than top-half.

### 2. KNIMEST (rank 71) — `KNIMEST/openadmet-cyp-model-report`
- **Architecture:** two-stage TabICL over frozen CheMeleon embeddings PCA→256. Stage 1
  predicts 4-enzyme screen log2FC; Stage 2 adds the 4 predicted log2FC as features, one
  TabICL per CYP for pIC50.
- **CV:** 20% scaffold holdout × 4 splits; official ST-RAE + a test-set-similarity-weighted
  variant. Local T (out-of-fold stage-1) beat A (in-sample) on all 4 splits.
- **Calibration/placement:** **none** — "I made no submission-time adjustments, such as
  shifting predictions."
- **External data:** none (pretrained CheMeleon + TabICL weights only).
- **Stated scores (blind):** v4 no-log2FC **0.8999** (rho 0.5965, R2 0.0405); T (OOF)
  **0.8111** (rho 0.6545, R2 0.1409); A (in-sample, final) **0.5977** (rho 0.6698, R2 0.4170).
- **Why:** predicted-screen feature works (−0.089 blind, matching −0.092 local estimate);
  placement is the entire gap A→T (0.5977 vs 0.8111 with identical ranking — the in-sample
  stage-1 values inflate spread into the blind regime). They chose A **after seeing blind
  results** (self-declared post-hoc; local validation disagreed with blind on A-vs-T).
  No calibration discipline + rho 0.67 keeps them mid-table.

### 3. talindrew (rank 75) — `Arnab28122000/openadmet-cyp-challenge` (docs/drugrx-training/MODEL-REPORT.md)
The single most informative report in the field. **Rank 172 → 60** via metric mechanics,
not architecture ("LightGBM and a small multitask MLP ensemble, both unremarkable").
- **Stated score:** macro ST-RAE 0.8663 → **0.5414**, R2 0.0748 → 0.5329, rho 0.6212 →
  **0.7093**. Per-isoform ST-RAE 0.549/0.462/0.587/0.567.
- **Three measured drivers:** (1) dose-response labels are **selected** labels (promotion
  per isoform measured: 2D6 DRC iff log2FC < −0.6; 1A2 rises with inhibition; 2C9 ~random;
  3A4 enriched for weak) — treat non-promoted compounds as **interval labels** (no-inhibition
  at 50 µM ⇒ pIC50 ≈ 2.2, window ≈ [1.07, 3.52]): **−0.243 macro ST-RAE**, 2D6 rho 0.378→0.476.
  (2) validation surface must contain non-inhibitors (V2 reproduces the board per isoform
  0.77/0.53/1.35/0.48 vs board 0.88/0.52/1.46/0.57; V1 could not see the defect).
  (3) per-isoform affine placement onto the blinded population.
- **Placement = direct metric exploitation:** ST-RAE optimum spread ≈ **0.896 × squared-error
  optimum** (spread discount confirmed to 4 decimals via board probe); wants "less spread and
  a centre above the mean". **Board-as-instrument**: ST-RAE on fixed compounds is
  deterministic, so one-axis-per-submission secants measure the metric derivative (2C9 secant
  −0.297→−0.045, 3A4 overshot and paid only +0.0007 → "overshooting is cheap"). Also derived
  the closed form to **solve blind population mean/sd from your own reported row**
  (s² = s_y² + s_p² − 2 r s_y s_p; m² + s² = (1−R2) s_y², r = 2 sin(πρ/6)) — recovers test sds
  1.44/0.97/1.66/1.16.
- **Decision rule:** ST-RAE minimiser = **balance point (weighted median over pooled bound
  draws)**, not the mean ("80% likely inactive ⇒ predict 3.52, not 2.87"); ties go to the
  RIGHT end of a flat window.
- **External data:** none in the shipped model; Veith external inactives made it WORSE
  (+0.038, 3/3 seeds).
- **Rho evidence (their negative arms):** ranking objectives **all lose**: lambdarank −0.177
  rho, rank_xendcg −0.038, soft-Spearman −0.0729 (t −17.6), RankNet −0.0182, ordinal −0.0185,
  0/4 isoforms positive; tie-collapse of wide censored intervals costs +0.0543 rho (the
  censoring policy, not the loss, crippled rankers). "Training against the metric beats
  changing the representation": molf_ft − molf_lin = **+0.0385** under the interval hinge;
  CheMeleon D-MPNN beats MoLFormer-XL by +0.0745 (domain D-MPNN carries chemistry; general
  chemical LM carries less than a random rotation of descriptors, −0.0225). Ensemble advantage
  over plain GBM **unmeasurable** (paired: −0.0066).
- **Validation geography:** 2/3 of scaffold-holdout sits in a NN-Tanimoto band holding <5% of
  the test (test median NN 0.587 vs validation 0.439); "scaffold split" here ≈ random.
  Importance-weight to the test histogram flips signs on effects <0.01.

### 4. adlvdl (rank 77) — `adlvdl/cyp_competition`
- **Architecture:** one Chemprop D-MPNN, 4 heads, encoder **pretrained on PubChem qHTS
  (Veith) panel**, fine-tuned unfrozen on all four endpoints ("pubchem" arm). Five later
  routes (06–11) all ended "ship pubchem as-is". Pool B (2026-10-20, unscored): Caruana
  blend pubchem + TabICL on Monroe / Atom-JEPA / CheMeleon frozen embeddings (PCA 128),
  weights refit on 25-fold OOF; CV macro 0.6604 vs 0.716 pubchem.
- **CV:** nested scaffold 5×5, shared folds; paired bootstrap + Holm–Bonferroni before
  ranking arms ("a grid with few stars means the CV cannot support a ranking").
- **Calibration/placement:** the repo's core artifact (`src/cyp/placement.py`):
  **R2 = 2ρk − k² − b²** (k = sd_pred/sd_true, ρ = Pearson); placement moves no rank.
  Board history: 0.7785 (05) → **0.7040** (07, placed at k=ρ, rank 87) → **0.6578**
  (12, ST-RAE-optimal placement on gated endpoints, rank 84/205; R2 0.4587, rho 0.6623).
  **Gating:** placement on an already-placed endpoint is a cost — 07 placed all four, CYP3A4
  (96% of R2 ceiling) came back +0.187 worse; gate at recoverable-R2 < 0.05. Blind
  population **solved by intersection of two differently-placed scored vectors** (mean/sd
  solved to disagreement 0.0005–0.0039; CYP2D6 "no consistent population"). CYP1A2 placement
  probe: sd 0.826→0.6887, 1.075→0.6559, **0.648→0.6256 best** (narrower wins on 1A2).
  "Simulated ST-RAE: direction, never magnitude" — overshot by 0.203 after bracketing at
  0.040/0.128.
- **Rho/ensemble evidence:** "07's four regression arms correlate at r ≈ 0.94 — four variants
  of one pretrained chemprop, where a blend has nothing to average away" vs TDI arms at
  r 0.30–0.62 where blending works. CYP2D6 rho 0.34 caps R2 at 0.126; scored −0.742 → the
  0.868 gap is pure placement.
- **External data:** PubChem qHTS pretraining (the shipped model's encoder); union arms with
  ChEMBL + aux measured, "none beat this baseline at a resolvable margin".
- **Stated score:** macro ST-RAE **0.6578**, rank 84/205 (task board: rank 77).
- **Why:** excellent metric algebra + disciplined board probing; the shipped model's ranking
  is mediocre (rho 0.6623) — placement gains are capped by a weak ranker.

### 5. CompChem Magicano (rank 80) — `mokhalaf22/openadmet-cyp-public` (REPORT.md + FINDINGS.md)
- **Architecture:** from-scratch multitask **D-MPNN, 4 interval heads, 3-seed ensemble +
  predicted-primary-screen features** (predicted OOF log2FC, 4 features). Thesis: "the
  ceiling on this benchmark is representational, not capacity" — ridge ≈ LightGBM ≈ MLP
  (OOF ST-RAE 0.489 vs 0.451). CheMeleon frozen (→TabICL) and fine-tuned did NOT beat it.
- **CV:** scaffold-split OOF, deterministic folds; seed-ensemble spread ~0.004 = noise floor.
- **Metric optimization (direct, the clearest in the field):** **interval hinge loss** =
  "L1 distance to nearest credible-interval bound, zero inside" — "matching the competition's
  soft-threshold metric exactly, so training and scoring agree". Point→interval targets:
  OOF ST-RAE 0.445 → 0.434, **recovers the CYP3A4 gap** (0.323→0.305) — CYP3A4 has 23.6% of
  rows wider than 2 log units, and its gap "had survived every capacity lever".
  **Loss-reweighting sweep (exponent of 1/(1+width)^p) is a V with minimum at uniform p=0
  (0.4131)**: metric per-row gradient is uniform (±1/den), so any reweighting mismatches it.
  Proxy **point** targets (11,505 cells, Pearson 0.76–0.93) cost **ST-RAE +0.0442** vs rho
  −0.0141: "more supervision at the wrong fidelity is worse than less at the right one".
  Sampling targets inside intervals: +0.0065.
- **Calibration:** board trajectory **0.9356 → 0.7114 → 0.6683 → 0.6411** (rank 200→95→83→77):
  CYP2D6 −0.5 location shift (organizers confirmed 2D6 test excludes hit-expansion), then
  **dispersion expansion** 0.85× then full training spread, ranking invariant (rho fixed
  0.6965, R2 0.2495→0.3477). OOF said spread expansion HURTS, the board said it gained 0.043 —
  "and the board was right": OOF sd(true) 0.78–1.09 vs blind **1.297** (36% wider).
  ST-RAE-optimal spread measured **≈0.8ρ in-distribution** (flat at min, gap not exploitable).
  Live-board discipline: act only on moves ≥0.03, OOF-and-board agreement, unscored half =
  permanent holdout.
- **Rho evidence:** ranking loss null; rank transform / ordinal link null on 2D6;
  isoform **sharing** is worth +0.0189 macro rho (2D6 −0.038, 2C9 −0.019, 1A2 −0.014, 3A4
  −0.004) — sharing subsidises data-poor tasks. Ensemble rule: "averaging pays only when
  members are both mutually diverse AND comparably strong"; gain correlates +0.892 with member
  strength, −0.736 with decorrelation; reached rho 0.758 between members and the average
  gained +0.0002.
- **External data (all null):** AID 1851 auxiliary heads degraded; ChEMBL/PubChem neighbors
  = 324 compounds (13.3% anchor density at Tanimoto 0.7); **Enamine REAL 111,361
  unlabelled near-neighbors (median NN 0.818, 90.3% ≥0.7) pretrain = −0.0042 rho** (holding
  the encoder: −0.017/−0.018) → "unlabelled structural density is not the information this
  task lacks; the missing quantity is measured potency". Octant release = 99.3% the
  challenge's own compounds; found **5 blinded test compounds leaked** in it, quarantined.
- **Stated score:** board MA-ST-RAE **0.6411** (MAE 0.8394, R2 0.3477→0.4272, rho 0.6965).
  "Ranking is at the representational ceiling (OOF 0.604; blind 0.6965; **leaders ≈ 0.78**)."
- **Why:** the most rigorous negative-results science in the field; 25 levers enumerated,
  one improved the model. Ships the best loss formulation; capped by rho ≈ 0.70.

### 6. Nomad-link-id / Igor Eduardo, Cortexa Social (rank 107) — HF `Nomad-link-id/openadmet-cyp-methods`
- **Architecture:** LightGBM per isoform on ECFP4+RDKit2D (± Emax/screen aux features);
  Chemprop v2 multitask D-MPNN 4 tasks (+8 aux tasks); Chemprop fine-tuned from CheMeleon
  foundation weights (best per-isoform, 0.4–0.5 blend weight). Per-isoform grid-searched
  blend weights.
- **CV:** 5-fold Murcko scaffold, shared folds, OOF; blend weights picked on the same OOF
  (self-declared mildly optimistic).
- **Calibration/placement:** **none** (no affine/level step at all).
- **External data:** none (CheMeleon public weights only).
- **Stated score:** OOF blend RAE 1A2 0.830 / 2C9 0.714 / 2D6 0.932 / 3A4 0.567; blend rho
  0.563/0.691/0.438/0.800. No board number in report.
- **Why:** a competent standard-Kaggle pipeline (GBM+D-MPNN+FT-foundation, scaffold CV,
  blend) with **no interval/censoring treatment and no placement** — point-label training on
  selected rows, predictions left on training-population scale. This is the modal bottom-half
  recipe; its OOF ~0.76 translates to a mid-0.8s/0.6s blind score.

### 7. SC38 / CYP-Shift (rank 119) — `SimonCrouzet/CYP-Shift` docs/approach.md
- **Architecture:** modular config-swept pipeline: GBT on fingerprints+physchem, MLP,
  Chemprop D-MPNN, and a **dose-response curve model** (predicts the Hill curve; DRC label
  constrains midpoint, single-concentration constrains one point — "two assays become two
  views of one object"). Enzyme described by chemistry-derived descriptors (leave-one-enzyme-out
  pruned). Boltz-2 enzyme+heme+ligand **co-folding** features (pose, predicted affinity)
  cached to disk. Aux sources (ChEMBL, SAIR, Veith, Tox21, Novartis, Octant) all
  on/off switches, off by default.
- **CV:** similarity-matched holdout reproducing the blinded test's NN-Tanimoto profile
  ("a random split is worse than merely optimistic: it is biased in favour of descriptors
  and against fingerprints, which inverts the test regime").
- **Calibration:** "train directly on the soft-threshold objective" (end-to-end official
  metric reproduced in-pipeline); open lever = calibrate **per-isoform predicted spread and
  offset so each point falls inside its band** rather than at the posterior mean.
- **Domain insight:** CYP2D6 is not bulk-physchem (CV R2 0.375 on 3A4, negative R2 on 2D6);
  it tracks a **protonated aliphatic amine** (+0.55 log units on 2D6, negative on the other
  three — the Asp301/Glu216 ion pair). Cross-isoform correlation ≤0.06; "it decides the macro
  average, it is where the remaining gains most likely sit."
- **Stated score:** none in README/approach.md.
- **Why:** ambitious modular sweep with the right metric + validation ideas; the report is a
  pre-deadline "what we expect to matter" note — spread calibration and 2D6 representation
  were still open, i.e. shipped before the levers landed.

### 8. dargason (rank 235) — https://admet-cyp.aetherark.com/regression-submissions.html
- **Page content (HTML fetched):** a submission log with **only two rows**:
  - #1 "Constant 4.0 baseline" (2026-08-20): **Overall ST-RAE 1.3132, rho 0.0000;
    CYP1A2 1.0886, CYP2C9 1.6432, CYP2D6 1.1188, CYP3A4 1.4022** (per-isoform ST-RAE of a
    constant predictor — reveals the metric denominator is the *mean-predictor* loss, not the
    constant-4.0 loss, and per-isoform scale sensitivity: 2C9 is the most punishing isoform
    for a 4.0-centered flat prediction).
  - #2 "002-OADMET-CYP-chemeleon_linear-fit" (2026-08-21): board columns **blank** on the
    page (no stats published; stats are in the linked report).
- **002 report (saltzberg/OpenADMET-CYP, the organizers' own baseline track):** frozen public
  OpenADMET ChEMBL-37 CheMeleon CYP checkpoint → raw pIC50 → per-endpoint affine linear fit
  on challenge labels (5-fold Bemis-Murcko scaffold CV). **OOF macro ST-RAE 0.8981 raw →
  0.5734 linear-fit; rho 0.2881 → 0.2859 (unchanged)** — calibration moves level only; the
  public ChEMBL-trained CheMeleon checkpoint has near-useless ranking on this assay (rho
  ~0.22–0.37 per isoform). Slopes <1 on 2D6 (0.490) = the model over-dispersed vs challenge.
- **Method:** "Bayesian claim-evidence graph" integrating heterogeneous sources —
  "Framework in progress", no scored model.
- **Why:** in-progress framework + two baseline submissions; the page's value to us is the
  per-isoform constant-predictor ST-RAE vector (1.0886/1.6432/1.1188/1.4022) confirming
  metric normalization, and the raw-CheMeleon rho ~0.29 ceiling of the unadapted public
  checkpoint. NOTE: the page lists only 2 rows — no fleet of submission stats to mine; the
  hoped-for cross-entry spread-pattern analysis is not possible from this page alone.

### 9. HauserGroup (rank 205) — `HauserGroup/cyp-challenge`
- **404 — repository does not exist** (GitHub API: repo not found; user exists with 12 other
  repos, none CYP/ADMET-related). Nothing to mine.

### 10. ajogalekar / curiouslywavefn (rank 230) — METHODS.md
- **Architecture:** four independent OLS models on **14 fixed RDKit physchem descriptors**
  (MolLogP, MolWt, MolMR, TPSA, HBD/HBA, RB, rings, heteroatoms, CSP3, heavy atoms, charge),
  StandardScaler + LinearRegression per endpoint on labeled rows only. Explicitly "no CV,
  no hyperparameter optimization, no ensembling, no clipping, no offset correction,
  no interval weighting" in the shipped recipe.
- **CV:** none for the submission (full-data refit); the wider program has grouped repeated
  CV; internal `macro_leaderboard.csv`: best T1_simple_desc_ridge **0.7045** MA-ST-RAE /
  mean rho 0.587; mean-predictor 1.0012; T0_1NN 1.6515.
- **Calibration:** none; review notes "compression of the predicted range, particularly for
  CYP2D6" (a prediction-distribution diagnostic).
- **External data:** none (no pretrained weights, no TDI, no Emax, no screen).
- **Stated score:** no blind number claimed ("format compliance and numerical reproducibility
  do not establish blind-test accuracy").
- **Why:** a deliberately transparent reproducibility-graded baseline; sits at the bottom
  because it is a 14-descriptor linear model with the known 2× prediction-spread compression
  and zero calibration. Its ridge-vs-OLS-vs-RF table (0.7045/0.7068/0.7227, rho 0.587/0.588/0.478)
  is a clean read on descriptor-only linearity: rank collapses (RF rho 0.478, Morgan RF 0.358)
  while linear models hold rho ~0.59.

---

## SYNTHESIS

### (a) Structural separators: top-half (≤~0.55) vs bottom-half (≥~0.60)
1. **Censoring/selection fix.** Dose-response rows exist only for screen-promoted compounds
   (per-isoform promotion rules; 2D6: DRC iff log2FC < −0.6). Top-half: non-promoted
   compounds as interval/censored labels (talindrew −0.243 macro; Magicano hinge recovers
   CYP3A4). Bottom-half (Nomad, ajogalekar, 002-baseline) train point labels on selected
   rows only → systematically over-predict the inactive-heavy blind set (the raw 002 CheMeleon
   checkpoint shows the extreme: rho 0.29, ST-RAE 0.898).
2. **Blind-population placement (affine, rank-preserving).** Measured gains: jeremy
   0.7247→0.5214 (+0.203), talindrew 0.8663→0.5414 (rank 172→60), adlvdl 0.7785→0.7040→0.6578
   (+0.12), Magicano 0.9356→0.6411 (+0.29). Entries with NO placement (KNIMEST 0.8111, Nomad,
   ajogalekar) sit 0.15–0.30 worse with identical rankings. Spread discount ~0.896 (talindrew)
   / k*≈0.8ρ (Magicano): ST-RAE wants less spread than R2's k=ρ, centre above the mean.
3. **The predicted primary screen (log2FC).** The only auxiliary lever that works repeatedly:
   Magicano D-MPNN 0.434→0.415 OOF, KNIMEST blind −0.089, talindrew uses the screen as the
   censoring source. It works because it is strongly structure-learnable (Pearson 0.59–0.74)
   AND a different assay.
4. **Population-matched pretraining, population-matched data.** Jeremys best ensemble
   contributor is the AID1851-augmented chemprop (NCATS qHTS ≈ challenge screening pop);
   adlvdl ships a PubChem-qHTS-pretrained encoder. ChEMBL med-chem pull is 1–1.65 log units
   too potent (jeremy), and distant AID1851 heads *hurt* Magicano (distant chemistry).
   The blind set is Enamine catalogue chemistry: public measured labels on it do not exist
   (13.3% anchor density at Tanimoto 0.7; 111k unlabelled REAL near-neighbors = null).
5. **Validation surface matched to the blind NN profile.** talindrew: 2/3 of scaffold-split
   validation sits in a band covering <5% of test; Magicano: OOF ST-RAE 0.451 vs blind 0.9356
   (2×), OOF sd(true) 0.78–1.09 vs blind 1.297; sc38 + KNIMEST both built similarity-matched
   / test-weighted surfaces. Anyone calibrating off raw scaffold-CV magnitude gets the
   spread direction backwards.

### (b) What raises Spearman rho (we are 0.7245; rank-30 band needs 0.75+)
- **Rank losses do NOT raise rho — repeatedly, with statistics.** talindrew 90-paired-unit
  arms: soft-Spearman −0.0729 (t −17.6), RankNet −0.0182, gap-weighted −0.0187, ordinal
  −0.0185, lambdarank −0.177, xendcg −0.038; oracle blend ceiling of the family +0.009.
  Magicano: pairwise margin loss null. CYP2D6 rho caps ~0.34–0.44 in several reports — it is
  data-limited, not objective-limited.
- **Trained-against-the-metric raises rho:** Magicano point→interval hinge recovers CYP3A4
  (the wide-interval isoform); talindrew fine-tuning the encoder under the interval hinge
  beats every frozen arm (+0.0385 over frozen MoLFormer, 0.5179 vs all frozen arms). The
  hinge keeps the metric's zero-error zone; anything that re-imposes point targets
  (proxy cells, inside-interval sampling) costs ST-RAE 3× more than rho.
- **Multi-task sharing across isoforms:** splitting to single-task costs −0.0189 macro rho
  (2D6 −0.038) — sharing subsidises data-poor isoforms (Magicano, measured).
- **Representation:** domain D-MPNN fingerprints (chemprop/CheMeleon) carry real chemistry;
  general LMs (MoLFormer, ChemBERTa, Uni-Mol) excluded by top entries; Magicano says ranking
  is at the *representational* ceiling with public labels (OOF 0.604, blind 0.6965, leaders
  ≈0.78). The 0.72→0.78 rho gap is a **data/label-coverage gap**, not an architecture gap.
- **Ensemble diversity raises rho only under a precondition:** members both decorrelated AND
  comparably strong (Magicano: gain +0.892 corr. with strength, −0.736 with decorrelation;
  rho 0.758 between members → average gained +0.0002; adlvdl: arms at r≈0.94, blend =
  nothing). Jeremys Caruana bagged selection over a pool of strong+decorrelated models
  (TabPFN×2 encoders + 17-head FT + LGB) is the working recipe.
- **'SVM' (rank 50, rho 0.7665, R2 0.494) is consistent with all of this:** a pure ranker
  maximizes rho by construction (SVM-rank style objective, wide spread), sacrifices level
  calibration; the metric lets you keep rho and buy ST-RAE back with placement — which they
  apparently did not optimize (R2 0.494 below their rho ceiling).

### (c) Who optimized the banded metric DIRECTLY, and how
- **Magicano:** interval-hinge loss (dead-zone L1 to credible bounds) = the metric; proved
  uniform row weighting is optimal (V-shaped sweep, p=0 best, gradient-matching argument);
  proxy point-targets rejected on fidelity grounds (+0.0442).
- **SC38:** pipeline "trains directly on the soft-threshold objective", metric reproduced
  end-to-end in training.
- **talindrew:** analytic decision rule — ST-RAE risk minimiser = weighted median balance
  point over bound draws; right-end tie-breaking; spread discount 0.896 measured to 4
  decimals by board secants; R2/rho inversion algebra to recover blind population moments.
- **adlvdl:** R2 = 2ρk − k² − b² placement algebra; ST-RAE-optimal (not R2-optimal)
  placement; gate placement where recoverable-R2 < 0.05; intersection solve of blind moments
  from two scored vectors.
- **jeremy:** placement target swapped to reverse-engineered BLIND_MOMENTS (k = ρ Pearson-
  optimal slope, SuperCowPowers recipe) — biggest single measured blind gain in this set.
- Nobody in this mid/low set trained a banded/hinge loss *with a rank objective hybrid*;
  the metric-exact hinge is always L1-to-bound, uniform weights.

### (d) dargason page — what the listed stats reveal
- Constant-4.0 per-isoform ST-RAE **1.0886 / 1.6432 / 1.1188 / 1.4022** (overall 1.3132,
  rho 0): denominator is mean-predictor loss; isoform sensitivity to a flat 4.0 center
  varies ~1.5× (2C9 worst) — a free calibration prior for spread/center stress-testing.
- 002 (frozen public CheMeleon CYP + linear fit): **OOF ST-RAE 0.8981→0.5734 with rho
  unchanged (0.2881→0.2859)** — the cleanest public demonstration that affine placement buys
  ~0.32 ST-RAE at zero rho cost, and that the public CYP-CheMeleon checkpoint is a weak
  ranker (rho ≈0.29) on challenge chemistry.
- The page lists ONLY these 2 rows (submission #2's board cells blank) — no fleet of
  extreme-spread entries to trade off ST-RAE vs R2; the cross-entry metric-structure mining
  must come from talindrew/adlvdl (they did it on their own rows via board probes).

---

## WHAT RANK 51 LACKS THAT RANK 20 HAS (ranked, 1-line each)

1. **rho ≈ 0.78 vs 0.7245** — every report that got near it says it comes from label coverage
   (screen-censored rows, multi-task sharing, population-matched pretraining), not from rank
   losses (all rank-loss arms measured NEGATIVE).
2. **A metric-exact interval-hinge training loss on the full censored label set** (not
   point-label training + post-hoc calibration) — Magicano/talindrew measured −0.011 to −0.243
   from this switch, concentrated exactly on CYP3A4's wide-interval rows.
3. **Board-instrumented, per-isoform placement secants** — top-half teams probe the
   deterministic board one axis per submission (spread discount k*≈0.896ρ per isoform,
   per-endpoint centers); we calibrate once, off OOF, which the field shows runs 2× optimistic
   in spread and 0.4–0.5 log units off in center.
4. **A test-matched validation surface** (NN-Tanimoto-reweighted holdout incl. non-inhibitors):
   without it, every spread/weight decision is anchored to a population holding <5% of the
   test's chemistry neighborhood.
5. **The predicted-primary-screen feature done OOF** (predicted log2FC as features): the one
   auxiliary lever every mid-to-top entry confirms positive (−0.089 blind, −0.019 OOF).
6. **A second, independently-strong model in the blend** — ensemble value requires members
   comparably strong AND decorrelated (precondition test: pairwise r; our GBM/TabPFN arms are
   the weak-and-correlated case; "make a second model strong, don't combine weak ones").
7. **CYP2D6-specific engineering**: it decides the macro (widest score band, cross-isoform
   correlation ≤0.06, ion-pair substructure feature, promotion threshold log2FC<−0.6, test
   excludes hit-expansion); top entries attack it individually, mid entries report it as a
   lost cause.
8. **Exploitation of the metric's asymmetries**: center above the mean, ties to the right end
   of the free window, weighted-median (not mean) point placement, overshoot-cheap/undershoot-
   expensive flat surface — free ST-RAE with zero rho cost.
9. **Submission-budget discipline as measurement science**: deterministic metric ⇒ board deltas
   are exact measurements; move one axis at a time, act only above the ~0.03 split SE, keep
   the unscored half as holdout (KNIMEST shows the post-hoc-exploitation failure mode:
   local/blind sign flips).
10. **Nothing more in external data**: the field's negative results (AID1851 heads degrade,
    111k REAL unlabelled neighbors null, Veith inactives worse, Octant = 99.3% the training
    set) bound the external-data play; rank 20's edge is label FIDELITY and calibration
    discipline, not more molecules.
