# Mining report: TOP open-code entries, OpenADMET CYP regression leaderboard
Mined 2026-10-22. Sources fetched in the assigned order. Quotes are verbatim from the
reports/repos; numbers are as stated by the authors. Nothing here was re-verified against
the organizer board.

Our context: rank 51, ~12 days left. Our stack: chemprop D-MPNN (external pretrain
AID1851/TAME/PKSim/CYPKIDI) + CheMeLeon GEM embeddings + TabPFN/TabICL + GBM/mordred +
5-fold scaffold ridge blend (19 members) + reveal-calibrated constant-spread placement
(per-isoform F_SPREAD 0.6–1.0).

---

## 1. briford (SuperCowPowers / ADMET Workbench) — rank 20, macro ST-RAE 0.4378

Source: https://supercowpowers.github.io/workbench/blogs/cyp_challenge/ (fetched 200 OK) +
full pipeline code in the public repo `SuperCowPowers/workbench`,
`ml_pipelines/OpenADMET/cyp/` (cloned; scripts quoted below).

### (a) Model architecture + training recipe
- Headline: "An average of four Chemprop models over the challenge data plus public CYP
  potency, stock hyperparameters, plus a per-isoform calibration. **Macro ST-RAE 0.4378**
  on the live half of the test set."
- "Four Chemprop D-MPNNs, averaged. All are SMILES-only with **stock hyperparameters**,
  trained on the challenge release plus ChEMBL and the NCATS qHTS panel — **31,670 compounds
  and 161,400 labels** at the current FeatureSet. They differ in which targets share an encoder."
- Members table (verbatim): `union-p30` 18 heads, all four isoforms, challenge+ChEMBL+qHTS;
  `mt-aux-100` 8 heads, all four isoforms, challenge arms only; `2d6-isoform` 6 heads,
  "every CYP2D6 readout, nothing else"; `2d6-single` 1 head, CYP2D6 pIC50 alone.
  "Only the four scored pIC50 heads are ever submitted; the rest exist to shape the
  representation."
- Ensemble logic: "On CYP2D6, where all four members predict and the pool can be scored out
  of fold, the average beats every member it contains — **0.503 Spearman against 0.445**
  for the best of them — and the gain flattens by the fourth."
- Auxiliary heads as representation shaping: `mt_aux_100` runs pIC50 targets +
  `cyp*_log2fc` single-concentration aux heads with `AUX_WEIGHT = 0.3`
  (code: `cyp_chemprop_mt_aux_100.py`). Gradient-share math from code docstring:
  "A task's gradient share is its label count times its weight, not its share of the task
  list… The public targets carry 137,375 labels against the four scored targets' 6,525,
  which is why a 0.3 multiplier on them is not a modest thing: `--public-weight 0.05` →
  scored 27% assay 42% public 30%; `--public-weight 0.30` → scored 11% assay 17% **public 72%**."
- Ensemble weighting (code `scripts/cyp_ensemble_submit.py`): "Members are weighted per
  compound by **calibrated confidence**, not averaged flat. Each member's confidence is
  scaled by its own |conf-to-error correlation|… Out of fold that beats the flat mean on
  every isoform by more than the isoform resolves: CYP1A2 0.6074→0.6569 (+0.0495), CYP2C9
  0.7307→0.7832 (+0.0525), CYP2D6 0.4949→0.5583 (+0.0634), CYP3A4 0.8321→0.8574 (+0.0253)."
  Confidence = 5-fold-CV ensemble std, conformal-calibrated (Workbench `docs/confidence/index.md`:
  "5-model ensemble… std across 5 predictions… Conformal scaling").
- Extra representation leg: "TabICL over a **720-d [Monroe] molecular embedding**"
  (`cyp_tabicl_monroe.py`), top-variance 360 dims default; CYP2D6 OOF rho 0.448 vs
  best chemprop 0.445 and "the first change to move" the AUC <4.0 vs >=4.5 discrimination
  (0.799 vs 0.763).
- CheMeleon option in the 2d6 script: "`--foundation` swaps the from-scratch encoder for
  CheMeleon, a D-MPNN pretrained on Mordred descriptors over 1M PubChem molecules…
  CheMeleon's own headline is a 97% win rate on MoleculeACE, an activity-cliff benchmark."
- Seeds: three replicates of one config at different seeds (`cyp_seed_noise.py`,
  `-s43`, `-s44`) to measure the seed noise floor.

### (b) External data
- "Everything the entry trains on is public." Sources: OpenADMET challenge release
  (4,905 compounds + single-concentration log2fc on 4,375, TDI curves, emax);
  **ChEMBL 37** (CC BY-SA 3.0, 24,918 skeletons, 5 isoforms, "185 structures overlap the
  challenge deck and **zero** overlap the blind set", "nothing sits below pIC50 4.0");
  **PubChem AID 1851 Veith qHTS** (public domain, 16,546 compounds; "We keep `max_response`
  rather than its pIC50… the 42,355 rows the screen called inactive still carry signal");
  **Tox21 CYP screen** (7,196 compounds, "Available but **not** in the current entry").
- "ChEMBL and the qHTS panel are what the current entry uses… as separate heads rather
  than as extra rows" — "Pooling any of them into the scored columns needs a cross-assay
  correction; as separate heads none is required, since the encoder learns from all of them
  and each head keeps its own scale."
- "ChEMBL is almost entirely chemistry the challenge deck never saw, which is what a shared
  encoder can use; the qHTS panel is the only source that says anything at all about the
  compounds that did nothing."
- Censored variants (`IC50 > x` as bounds + `_lt` flag) and a `bounded_loss` are implemented
  in code ("A bound then costs the model only when its prediction rises above it") —
  "on this challenge it has not moved the scored metric."

### (c) Cross-validation scheme
- **Scaffold.** "The union's scaffold split runs over 31,670 rows, so the challenge
  compounds sit in folds alongside public chemistry where the control's folds were
  challenge-only."
- "Scaffold-OOF Pearson understates blind Pearson — the split is harder than the blind half,
  and an OOF prediction comes from one fold model where the endpoint averages five. Measured
  on `cyp-reg-chemprop-mt-aux-100`: OOF 0.570 / 0.681 / 0.408 / 0.795 against the solved
  0.753 / 0.838 / 0.678 / 0.851" → `OOF_TO_BLIND = {1A2: 1.32, 2C9: 1.23, 2D6: 1.66, 3A4: 1.07}`.
- Ruler power: "Repeated training runs of one configuration at different seeds disagree by
  more than candidate models do, so out-of-fold cross-validation cannot resolve the
  differences now being chased." Seed-noise floor: "0.05 Pearson on CYP1A2 and CYP2D6,
  0.013 and 0.012 on CYP2C9 and CYP3A4". Resolution thresholds 0.043 / 0.031 / 0.056 / 0.018
  (A1/C9/D6/A4).

### (d) Calibration / placement method — the crown jewel
- R² decomposition (verbatim): "R² decomposes exactly: `R^2 = 2ρk − k² − b²` with ρ the
  Pearson correlation, `k = sd(pred)/sd(true)` the spread ratio, and `b` the mean offset in
  sd(true) units… **The optimum is k = ρ, not k = 1.** Matching the spread of the truth is
  wrong: a model with correlation ρ should be ρ times as wide as reality."
- Blind-set moments are **solved from scored submissions**: "BLIND_MOMENTS… come from
  `solve_blind_moments`, which reads three scored submissions… Each board row is an exact
  equation: `R2*V = 2*cov(y,p) − var(p) − (mean(y) − mean(p))^2`, with V = var(y). Three
  unknowns… three entries spanning two spreads and two centres determine all three."
  Solved values: `BLIND_MOMENTS = {CYP1A2: mean 4.412 sd 1.553; CYP2C9: 4.830/1.101;
  CYP2D6: 3.107/1.599; CYP3A4: 4.880/1.272}`; `SOLVED_PEARSON = {1A2: 0.753, 2C9: 0.838,
  2D6: 0.678, 3A4: 0.851}`. "board noise of +-0.005 in R2 moves [rho] by roughly +-0.07…
  The moments are far more robust."
- **ST-RAE ≠ R² placement**: "The calibration that maximises R² does not minimise the scored
  metric. Soft-threshold RAE is zero anywhere inside a compound's credible interval, and
  low-activity compounds carry wide intervals, so **predicting high is nearly free while
  predicting low is punished by the actives**. The ST-RAE optimum therefore sits **above**
  the population centre, and on CYP2D6, the lowest-centred isoform, the two objectives
  point in opposite directions."
- Measured CYP2D6 probes (same predictions, affine only): "(3.57, 0.90) -> ST-RAE 0.5653
  (3.32, 0.90) -> 0.6271 (3.11, 1.08) -> 0.6944" and in code: "CYP2D6 board probes at mean
  3.11 / 3.32 / 3.57 / 4.35 score 0.6944 / 0.6271 / 0.5502 / 0.8558, so the minimum sits
  between 3.57 and 4.35 and **the penalty above it is far steeper than the gain below**."
  → `STRAE_MOMENTS = {"CYP2D6": {"mean": 3.57, "sd": 0.90}}`.
- Bias warning: "a weak vector minimises ST-RAE by hiding in the wide low-activity
  intervals, so an OOF-derived optimum is biased high. `sd` is the spread to hit outright,
  not `rho * sd`."
- Integrity check: "Spearman and Kendall come back bit-identical under an affine transform,
  which doubles as the integrity check." Clamps create ties and "a binding floor costs
  ranking" (FLOOR = 1.0 "Safety net only").

### (e) Metric-specific optimization
- No ST-RAE-direct *loss*; the metric optimization is at placement (above) plus
  bounded-loss experiments for censored labels. "Where does the ST-RAE optimum sit…
  A credible-interval width model would give it directly." (Open problem in blog.)

### (f) Stated scores
- "Macro ST-RAE 0.4378 on the live half of the test set" (blog header). CYP2D6 placement
  probes quoted above (0.5502–0.6944 for one isoform); R² 0.363 → 0.447 across placements.

---

## 2. rasayan-labs (rank 10, ST-RAE 0.4234) — repo EMPTY (noted, no report obtained)

Source: https://github.com/whis9/rasayan-cyp — `git clone` succeeds with
"warning: You appear to have cloned an empty repository." GitHub API confirms:
0 branches, 0 tags, contents → "This repository is empty.", commits → 409 "Git Repository
is empty.", raw README → 404. No gists. `whis9`'s only other relevant repo is
**`whis9/Rasayan-pxr`** ("Preliminary model report for the OpenADMET PXR Challenge",
pushed 2026-07-06) — the PXR report below is their *method house-style*, NOT a CYP report.
Treat all of it as proxy only.

Proxy method (from `Rasayan-pxr/MODEL_REPORT.md`, PXR pEC50 challenge, "rank 9 of 100,
Tier 1" there):
- (a) "`r26` — a **12-voice stacked ensemble**… **median (L1) stacking cascade** with
  isotonic calibration (per-voice → median stacker → final calibration → clip)".
  Lead block "≈ 61% of the model" is "a **CheMeleon encoder pretrained on the assay's own
  HTS screen and fine-tuned with fused pocket-docking (DDP) and quantum-pose (QVS)
  structural signal**"; voices also include MolE, UniMol, descriptor GBMs;
  stacker "Median (QR‑L1) stacker + isotonic **selected**"; deployed "censored-regression
  voice (Tobit)".
- (b) "4,083 curated PXR pEC50 + assay HTS screen for encoder pretraining"; "Public PXR
  augmentation (ChEMBL/PubChem/etc.) — **rejected — cross-assay label inconsistency
  (up to 1.5 log)**".
- (c) "validated by **Butina 5-fold out-of-fold** throughout".
- (d) Isotonic; "Tail de-shrink (stretch, hinges, disagreement-gating) — **MAE-optimal at
  identity — the tail is irreducible**"; "quantile shifts — overfit the held-out set";
  "Nearest-analog residual correction (ECFP4 k-NN) — overfits train, does not transfer".
- (e) "A strict 'both-metrics' validation gate. Every candidate had to improve *both* the
  training out-of-fold *and* a held-out set to be admitted."
- (f) PXR blind: "MAE 0.417 · RAE 0.578 · R² 0.56 · Spearman ρ 0.81… rank 9 of 100".
- Key takeaways (verbatim): "Assay-specific pretraining beats bigger models. CheMeleon-on-HTS
  (OOF 0.52) outperformed UniMol2-1.1B (0.72) and every general FM."; "The accurate-XOR-
  orthogonal wall is real… every candidate was either accurate-but-redundant (corr > 0.83)
  or orthogonal-but-weak (solo RAE > 0.7) — never both."

CYP leaderboard numbers (rank 10, ST-RAE 0.4234) are from our leaderboard copy, not from any
rasayan publication. **No CYP method report exists publicly as of today (Oct 22).**

---

## 3. stir_bar / lachrymator — rank 26 (live) & 28 (interim), interim macro ST-RAE 0.465; TDI rank 6/114 (Tier 1)

Source: https://github.com/lachrymator/openadmet-cyp-challenge-public (cloned; README.md
208 lines, "Write-up only… not the training code").

### (a) Model architecture + training recipe
- "A representation-diverse ensemble, **stacked per endpoint**." Base learners:
  "SMILES-sequence transformers, message-passing graph networks, 3D-conformer models,
  tabular in-context learners over frozen embeddings, a fingerprint baseline, and an
  additive-fragment model."
- "Auxiliary pretraining: **Masked multi-task pretraining on public bioactivity data**,
  warm-starting each backbone before the high-fidelity fine-tune."
- Stacking: "Per-endpoint comparison of ridge / elastic-net / non-negative / NNLS combiners,
  selected on held-out folds."
- Tabular FMs: "The frozen-embedding learners are fitted with **Mitra**, AutoGluon's tabular
  foundation model… Two comparable models were evaluated on identical embeddings, TabICL and
  TabPFN. **TabPFN was omitted.**… Across seven matched embedding sources **Mitra beat TabPFN
  on macro held-out R² in six**, and the two models' residuals correlated 0.94–0.98 —
  they were making very nearly the same mistakes, so leaving one out cost almost no ensemble
  diversity." (AutoGluon `zeroshot_2025_tabfm` config with TABPFNV2 popped.)
- Ensemble hygiene: "Residual correlations between members run ~0.55–0.88… unconstrained
  ridge and elastic-net combiners returned large cancelling coefficients: one member near
  +1.9 against another at −0.6, and on one endpoint the best standalone model pushed
  **negative**… one revision improved cross-validated error on all four endpoints while real
  held-out error got *worse* on three. The stacker is now restricted to **non-negative
  combiners**."
- "Each embedding source is **PCA-capped** before it reaches a tabular model. With only one
  to two thousand labelled rows per endpoint, an uncapped concatenation of several thousand
  embedding dimensions lets those models memorise the fold."
- "Pruning by correlation, not by count… **on the classification track the direct encoder
  heads individually outranked every tabular embedding leg, and including all of them
  produced the worst result on record for one endpoint.**"
- "Down-weighted calculated features. A property computed from structure alone carries a
  label on every row, so at full weight it supplies the large majority of the pretraining
  gradient… Those heads are down-weighted as a per-task weight — a constant per-row weight
  cancels out of a weighted mean and changes nothing."
- No seeds/hyperparameters disclosed (write-up only).

### (b) External data
- "Tens of thousands of near-neighbours of the held-out compounds, pulled from a large
  public chemical catalogue" — "**88,683 near-neighbours of the held-out structures** were
  retrieved from a large public catalogue — filtered at retrieval time to the held-out set's
  own physicochemical envelope and screened for structural liabilities… The **4,999 closest**
  are admitted to the training pool, entering with only their calculated physical properties
  and no assay endpoints." Similarity floor "chosen so that every admitted compound is
  closer to the scored molecules than the training set's own average nearest neighbour."
- Screening: "screened with rd_filters — ~1,150 named alerts… **any alert that fires on even
  one held-out compound is dropped** (~75 of 1249)".
- No ChEMBL/AID1851 naming (not disclosed); "Auxiliary pretraining helps where labels are
  thin, and the gain scales with how much of the corpus each head can actually see."

### (c) Cross-validation scheme
- "**Butina cluster-disjoint folds** — clusters assigned whole to a single fold, so no
  analogue series spans train and validation." "Cluster-disjoint validation first. **Random
  splits flattered every model. Butina folds made the held-out numbers track the
  leaderboard**, which is what made iteration meaningful at all."
- CV→blind transfer, quantified: "on **regression**, **100% of submission-endpoint pairs
  came out worse on the blind set than in cross-validation (median MAE +0.208)**; on TDI,
  82%… (median MCC −0.042)". "Pooled, MCC reads r = 0.90, while neither endpoint on its own
  exceeds 0.69."

### (d) Calibration / placement
- "Calibration is separable from ranking. Several learners ranked compounds well while being
  badly scaled; letting the stacker recalibrate recovered most of that." No spread-factor /
  conformal / clipping details published.

### (e) Metric-specific optimization
- None published; explicitly declines to pair their CV with ST-RAE: "Regression is shown as
  MAE rather than the board's own ST-RAE: that statistic is the organisers' and has no
  cross-validated counterpart here."

### (f) Stated scores (interim board, full test set, 219 entries)
- Regression interim macro: ST-RAE 0.465, MAE 0.595, R² 0.747, rank **28 (Tier 6)**;
  live board latest rank 28 of 230 ("top 12%"). Per-endpoint blind ST-RAE: 1A2 0.441,
  2C9 0.380, 2D6 0.653, 3A4 0.386.
- CV out-of-fold macro MAE 0.447 / R² 0.688 (1A2 0.429/0.671; 2C9 0.327/0.667;
  2D6 0.712/0.585; 3A4 0.320/0.828).

### TDI track (we care — they are rank 6)
- "**rank 6 of 114** on the TDI track, inside **Tier 1** — the top tier, the 26 entries the
  organisers could not separate from first place." Interim macro MCC 0.380, Accuracy 0.840,
  F1 0.484; live board best rank 14. CV out-of-fold: CYP2D6 MCC 0.270 / AUC 0.719;
  CYP3A4 MCC 0.554 / AUC 0.864.
- TDI lesson (verbatim): "**TDI is a different problem from potency.** Features that predict
  inhibition strength carry little about time-dependent inactivation, which is
  mechanism-driven." And on stacking TDI: "on the classification track the direct encoder
  heads individually outranked every tabular embedding leg, and including all of them
  produced the worst result on record for one endpoint" (diversity-over-membership discipline).

---

## 4. Scigantic — rank 28 live / **rank 11 of 219 interim, macro ST-RAE 0.4436**

Source page https://scigantic.com/sample/chem-cyp-challenge is a login-gated notebook
launcher (no report text server-side; bot launch refused). **The full report is the public
GitHub repo `Scigantic/open-admet-challenge-cyp`** ("How Scigantic's schema cards and
retrieval infrastructure were used for the OpenADMET CYP inhibition blind challenge",
pushed 2026-10-20) — cloned; README + `training/README.md` + `scripts/build_r25.py` +
`docs/submission_history.md` + lineage RESULT.md's are the report and full recipe
(MIT-0, frozen member predictions included).

### (a) Model architecture + training recipe
- "The twelve members… **Two backbones, both directed message-passing networks over the
  molecular graph via Chemprop**, all trained as multi-task regressors over the 22-column
  table…, **five scaffold folds each. 364 SageMaker training jobs** were run in total;
  these are the twelve that earned a place."
- Members (verbatim from table): armA `d_h 700 / depth 5`, random init, MAE,
  inverse-variance sample weights, 22 heads; armB `d_h 2048 / depth 4` random init;
  armC = B "message passing initialised from the CheMeleon checkpoint"; armD = B
  "initialised from the organizers' CYP-finetuned CheMeleon checkpoint"; ivB2048 / ivC2048
  = "interval hinge on scored heads" with **uniform** weights; mu2048 = MAE uniform;
  ii2048 = interval hinge + inverse-variance; mt_aux8 & union18 = "recipe-level reproductions
  of the public SuperCowPowers Workbench CYP entry"; aux_ac50; cp_interval.
  "**CheMeleon supplies eight of the twelve.**" (armC/armD init; foundation checkpoint
  zenodo 15460715, organizer checkpoint `openadmet/cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1`.)
- "MolFormer, ChemBERTa and Uni-Mol were fine-tuned and measured but did not earn a place
  in the stack." "Mordred descriptors with gradient-boosted trees sit alongside as a
  non-neural check."
- Hyperparameters (arm arms): d_h 700/2048, depth 5/4, FFN 300/1024, dropout 0.1/0.2,
  "batch 128, 200 epochs / patience 30, 5 scaffold folds"; fold seed 42; fixed `folds.npy`.
- Per-fold discipline: "**Score folds individually before pooling; a diverged fold hides
  inside a pooled average.**" (a NaN-diverged fold at epoch 4 with healthy-looking
  predictions: mean 5.326 std 0.393 vs healthy 5.02–5.12/0.297–0.340; only the per-fold
  score exposed it: 1.3763 vs 0.947–1.058).

### (b) External data
- Six-source union table: "**~32K compounds and 180K+ labels across 26 endpoints**":
  organizers scored pIC50 (6,525), organizer single-conc log2FC (17,500), organizer
  TDI-condition pIC50 (6,538), **ChEMBL pIC50 5 isoforms incl. CYP2C19** (56,705 / 24,921
  compounds), **Veith qHTS AC50** (15,070), **qHTS maximum response** (80,665 / 16,133).
  "**The scored labels are 3.6% of the table.** The rest is never scored and never predicted
  at test time. Training against it is what stops a 4,900-compound fine-tune from
  memorizing."
- "the qHTS maximum-response block is the largest single source at 80,665 labels: it is not
  a potency value but the raw screening response, which carries whether a compound does
  anything at all, and that turns out to be **worth more rows than any curated endpoint**."
- "The ChEMBL block covers five isoforms including CYP2C19, which the challenge does not
  score at all, because **a correlated fifth task improves the four that are scored**."
- "External data: ChEMBL (CC BY-SA 3.0) and PubChem bioassays (public domain). No
  proprietary data. Both were joined on InChIKey connectivity and purged against the blind
  set; one blind compound remains in the `union18` table with AID 1851 maximum-response
  values only (no potency label), kept for parity with the public recipe."
- TDI-condition heads used in R25: `g26_tdi_armB` = "arm B retrained with four extra heads
  for the organizers' TDI-condition pIC50" → blended w=0.2 into CYP1A2 only.

### (c) Cross-validation scheme
- "All members are **5-fold scaffold ensembles**; the fold assignment is fixed in
  `data/training/chemeleon_22/folds.npy`… (seed 42)."
- "Out-of-fold numbers in the notebook are on the rows the weights were fitted on and read
  optimistic; the per-endpoint transfer from offline gain to board gain is measured in
  `docs/submission_history.md` (CYP3A4 6 of 6 calls right, CYP2D6 2 of 5)."
- Directional transfer record (their headline discipline result): "The full reconstructable
  record is **13 of 20 genuine offline directional calls correct (65%…)**… **radically
  non-uniform across endpoints**": CYP3A4 6/6 (mean signed transfer ratio +0.82), CYP2C9 3/4
  (+0.40), CYP1A2 2/5 (+0.24), **CYP2D6 2/5 (−0.18): "the historical expectation of acting on
  a CYP2D6 offline gain is to make the column worse."**

### (d) Calibration / placement + (e) metric-specific optimization — ST-RAE-native
- "**The metric forgives near misses.** Each compound's measured value comes with a margin
  of uncertainty, and a prediction inside that margin is treated as correct. At CYP1A2 the
  margin is about 0.17 either side, and **23% of our predictions land inside it and cost
  nothing**."
- "**Squared error penalizes for those misses although the metric does not. So the twelve
  member weights are fitted to the competition metric itself, on held-out folds, rather than
  to squared error.**" Combiner: "per endpoint, weights over the twelve members' deltas from
  the incumbent are fitted by **Powell minimisation of the ST-RAE numerator (the hinge
  outside each compound's credible interval) plus an L2 penalty, on out-of-fold rows**;
  negative weights are allowed; the fitted column is then **restored to the incumbent's mean
  and standard deviation so that only the reordering is kept**." (Code:
  `hinge = lambda p, lo, hi: np.clip(lo - p, 0, None) + np.clip(p - hi, 0, None)`;
  `st_rae` = hinge(pred).sum() / hinge(mean).sum() — "port of the organizers'
  rae_soft_threshold_absolute_error"; `restore()` = z-score × ref.std + ref.mean.)
- Training-objective result (2×2 experiment, 5 folds, arm B): "**The interval objective has
  real independent value: +0.0275 at uniform weights**… It helps CYP1A2 (+0.0300), CYP2C9
  (+0.0384) and CYP3A4 (+0.0462), and is neutral on CYP2D6 (−0.0044)". "The interval
  objective helps under uniform weights and **HURTS under inverse-variance** [interaction
  −0.0510]". "interval+invvar is genuinely the worst cell in both experiments."
  Full matrix OOF ST-RAE (macro): mae+invvar 0.7187, mae+uniform 0.7170, interval+invvar
  0.7422, **interval+uniform 0.6895**.
- "And my +0.0913 was mostly fold noise… single-fold batch-64 probe vs 5-fold: **+0.0292**.
  **One fold is not a measurement.**"
- Task grouping: "**G4 is far worse than G22. The 18 auxiliary heads are load-bearing**…
  **Dropping the 18 auxiliary heads costs about 0.23–0.30 macro.** Not a sweet spot; a cliff."
- Weight-attenuation probe (R28, built not submitted): scalar per-endpoint attenuation
  c ∈ {1.1, 1.2, 2.5} selected on OOF by 5-fold CV, "Macro from the correction on the three
  shipped columns: **+0.00202**"; honest caveat "CYP3A4 does not reproduce between frames"
  (rotation c=1.75 vs full-OOF c=1.1); "CYP2D6 and CYP3A4 have sharp interior optima… CYP2C9
  and CYP1A2 are broad and flat from about 2.0 to 3.0".

### (f) Stated scores
- "**Interim leaderboard (2026-09-25), regression track: rank 11 of 219, macro ST-RAE
  0.4436.**"
- "Per endpoint, live ST-RAE of R25: CYP1A2 0.4739, CYP2C9 0.3802, CYP2D6 0.5442,
  CYP3A4 0.3762."
- Progression: R17 0.4620 → R24 0.4453 (joint 12-member hinge fit) → R25 0.4436
  (TDI-arm blend into CYP1A2, w=0.2).
- Live as of 2026-10-19: "the best entry is 0.3537 and our live row is rank 25 of 177,
  scoring 0.4456" (their live row is an older upload, not R25).
- Submission history: R00 tutorial LightGBM 0.9358 → R01 (endpoint placement + multitask
  base) 0.5228 → R25 0.4436. "the board scores four independent columns and re-scores an
  untouched column to the last digit" (per-column independence tripwire: 3/3 testable
  transitions PASS exactly, 4×macro-delta = column-delta).

---

# TOP-5 TRANSFERABLE IDEAS FOR RANK 51 -> TOP 30 WITH 12 DAYS LEFT
Ranked by expected-ST-RAE-gain per unit effort on OUR stack. "Gate" = honest go/no-go rule
before spending a submission. Noise-floor references: briford seed-noise 0.05 Pearson
(1A2/2D6), 0.013/0.012 (2C9/3A4); Scigantic per-fold deviation flag 0.085–0.15 ST-RAE.

1. **ST-RAE-native blend: replace the ridge objective with a direct hinge fit + moment
   restore.** (Scigantic R24, single biggest published gain: 0.4620→0.4453 macro, code in
   `scripts/build_r25.py`.) Fit per-isoform weights over our 19-member pool by Powell
   minimization of `sum(hinge(pred, conf_low, conf_high))` + L2 on scaffold-OOF rows (we
   have the organizer credible intervals in cache), then z-restore to the incumbent
   blend's mean/sd so only reordering ships. ~1–2 days, CPU only.
   **Gate:** 5-fold *rotation* CV (fit on 4/5 of OOF rows, score the held 1/5) — ship only
   if the held-out-fold ST-RAE gain is positive in ≥4/5 folds and mean gain > 0.004;
   require Spearman of shipped column unchanged vs incumbent blend within rounding
   (the affine-restore integrity check).

2. **ST-RAE-aware placement: retire constant-spread F_SPREAD in favor of a probed (mean, sd)
   grid per isoform.** (briford: the R²-style placement they replaced moved CYP2D6
   0.6944→0.5502, i.e. 0.14 on one endpoint; "the penalty above [the optimum] is far
   steeper than the gain below", optimum *above* the population centre and sd=0.90 ≪
   ρ·sd≈1.08.) Implement: on OOF rows evaluate the ST-RAE surrogate (hinge/credible
   interval) over an affine grid (mean ∈ [μ_oof−0.6, μ_oof+0.6], sd ∈ [0.6·ρ·sd, 1.1·sd]);
   ship best grid point, clamps off (ties cost ranking). <1 day.
   **Gate:** OOF surrogate is "biased high" (briford) — accept a grid point only if it wins
   on OOF *and* on the analog-enriched CV slice (blind set is hit-expansion clusters);
   probe with one submission per isoform at most, and never ship a grid mean > OOF-truth
   mean + 0.75 without a supporting probe (0.4456-style over-correction risk, briford's
   4.35 probe = 0.8558 disaster).

3. **Solve the blind-half moments from 3 cheap submissions (affine probes of one vector).**
   (briford's `solve_blind_moments`: "Each board row is an exact equation
   R2·V = 2·cov(y,p) − var(p) − (mean(y) − mean(p))²… three entries spanning two spreads
   and two centres determine all three" → they recovered mean, sd, and Pearson of the live
   half exactly, and their whole calibration program rests on it; ours currently guesses
   these via reveal.) Upload the same submission, a variant at 1.15× spread about the same
   centre, and a variant shifted ~+0.25 in centre at the new spread; per-isoform solve
   var(y), mean(y), cov(y,p) with the published quadratic. ~2–3 days incl. 3 board cycles.
   **Gate:** require the physical root (0.25 < var_y < 9.0) and mean(y) within
   [OOF-truth mean − 1.6, +0.5]; cross-check solved Pearson against measured Spearman per
   isoform (must satisfy ρ ≥ rank-correlation); treat ρ as ±0.07 (briford's noise bound).
   Feeds idea 2's grid with the true target distribution.

4. **Interval-hinge (soft-threshold) training arm at uniform sample weights.** (Scigantic:
   interval objective at uniform = +0.0275 macro arm-level OOF, 0.6895 vs 0.7170/0.7187;
   "the interval objective helps under uniform weights and HURTS under inverse-variance".)
   Train 2 arms of our existing chemprop recipe (union-style, 22-head 26-endpoint table:
   organizer pIC50 + log2FC + TDI-condition + ChEMBL 5-isoform + AID1851 max_response +
   AC50 — all public, joined on InChIKey connectivity, purged vs blind) with the scored-head
   loss swapped to hinge-outside-credible-interval, uniform weights, 5 scaffold folds,
   seed 42. ~2–3 GPU-days. Add g26-style TDI-condition aux heads in one arm (their R25:
   +0.0017 macro from a w=0.2 blend of exactly that arm).
   **Gate:** per-fold ST-RAE means, no fold deviating >0.085 from pooled (NaN-divergence
   tripwire); promote into pool only if OOF ST-RAE gain exceeds the seed-noise floor
   measured on our own replicates (run the one config at 3 seeds first); and per
   Scigantic's transfer record, any CYP2D6-only gain gets the strictest gate (2/5 correct,
   signed −0.18) — demand 4/5 folds.

5. **Confidence-weighted member blending + non-negative combiners.** (briford:
   per-compound weights ∝ |conf-to-error correlation| from 5-fold CV std beat flat mean
   on all four isoforms beyond resolution, +0.025…+0.063 Spearman OOF; stir_bar:
   "unconstrained ridge… returned large cancelling coefficients… restricted to non-negative
   combiners" after a CV-up/worse-blind revision.) We already have per-member 5-fold
   scaffold predictions → free per-compound std. Swap our ridge blend for
   confidence-weighted NNLS (delta-space, non-negative, L2) per isoform. ~1 day, no GPU.
   **Gate:** paired bootstrap over OOF rows on the blended column: 95% CI of ST-RAE delta
   excludes 0 AND direction agrees with idea 1's hinge-fit blend (they must not disagree,
   since they share the same pool); ship only through idea 2's moment-restore so no
   placement change rides along.

**Budget note (12 days):** 1 and 5 are CPU, 2 is analysis, 3 costs 3 submission slots, 4
costs GPU + 1–2 slots. Ideas 1→2→3 compose (hinge blend, then place it, then solve the
placement target exactly); spend submission slots on 3 and on confirming 2, and keep the
idea-4 arms out of the submitted pool until they clear their fold gate. Per Scigantic's
per-endpoint transfer record, invest changes on **CYP3A4/CYP2C9** first (6/6, 3/4 call
accuracy) and treat CYP2D6/CYP1A2 offline wins as unmeasured (2/5, 2/5).
