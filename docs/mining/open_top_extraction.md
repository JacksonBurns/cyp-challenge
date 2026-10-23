# Implementation extraction from `open_top.md` (L1 / L2 / L4)

Source: `/home/jackson/cyp-challenge/docs/mining/open_top.md` (451 lines, mined 2026-10-22).
Header caveat (verbatim): "Quotes are verbatim from the reports/repos; numbers are as stated by
the authors. Nothing here was re-verified against the organizer board."

Our context as stated in the doc: rank 51, ~12 days left; stack = chemprop D-MPNN (external
pretrain AID1851/TAME/PKSim/CYPKIDI) + CheMeLeon GEM + TabPFN/TabICL + GBM/mordred + 5-fold
scaffold ridge blend (19 members) + reveal-calibrated constant-spread placement
(per-isoform F_SPREAD 0.6–1.0).

---

## 1. Scigantic hinge-fit combiner — the L1 recipe

**Who:** Scigantic, rank 28 live / **rank 11 of 219 interim, macro ST-RAE 0.4436**.
Source = public repo `Scigantic/open-admet-challenge-cyp` (MIT-0, **frozen member predictions
included**), README + `training/README.md` + `scripts/build_r25.py` +
`docs/submission_history.md` + lineage RESULT.md's. (The scigantic.com page is a login-gated
notebook launcher, no report text.)

### 1.1 Loss / hinge definition (exact code as quoted)

```python
hinge = lambda p, lo, hi: np.clip(lo - p, 0, None) + np.clip(p - hi, 0, None)
st_rae = hinge(pred).sum() / hinge(mean).sum()   # "port of the organizers'
                                                 # rae_soft_threshold_absolute_error"
def restore(col, ref):                            # moment-restore
    return z-score(col) * ref.std + ref.mean     # "z-score × ref.std + ref.mean"
```

- Hinge is the **two-sided residual outside the credible interval** `[lo, hi]`: zero anywhere
  inside, `lo - p` below, `p - hi` above.
- The objective is the **ST-RAE numerator** = `hinge(pred).sum()` (sum, not mean), normalized
  for reporting by `hinge(mean).sum()` where `mean` = the constant-mean baseline column.
- Rationale (verbatim): "**The metric forgives near misses.** Each compound's measured value
  comes with a margin of uncertainty, and a prediction inside that margin is treated as
  correct. At CYP1A2 the margin is about **0.17 either side**, and **23% of our predictions
  land inside it and cost nothing**."
- "**Squared error penalizes for those misses although the metric does not. So the twelve
  member weights are fitted to the competition metric itself, on held-out folds, rather than
  to squared error.**"

### 1.2 Fitter (verbatim core sentence)

> "per endpoint, weights over the twelve members' **deltas from the incumbent** are fitted by
> **Powell minimisation of the ST-RAE numerator (the hinge outside each compound's credible
> interval) plus an L2 penalty, on out-of-fold rows**; negative weights are allowed; the fitted
> column is then **restored to the incumbent's mean and standard deviation so that only the
> reordering is kept**."

Decomposed, implementation-ready:

| Item | Exact as described |
|---|---|
| Optimizer | **Powell minimisation** (derivative-free; no gradient, no bounds needed) |
| Objective | `sum(hinge(p, lo, hi))` **+ L2 penalty** |
| Rows | **out-of-fold (scaffold-OOF) rows only** |
| Parameters | weights over **member deltas from the incumbent**, i.e. `p = incumbent + Σ_i w_i (m_i − incumbent)` |
| Sign constraint | **negative weights ARE allowed** (contrast with stir_bar §3) |
| Unit of fit | **per endpoint / per isoform** (4 independent fits) |
| Post-step | moment-restore to **incumbent mean and sd** (z-score × ref.std + ref.mean) → "only the reordering is kept" |
| Not disclosed | L2 coefficient value; Powell `xtol`/`ftol`/maxiter; starting point; incumbent definition beyond "the incumbent" |

Consequence of the restore: the shipped column is an **affine** transform of the fitted column,
so Spearman/Kendall are invariant. briford states this is the integrity check: "Spearman and
Kendall come back bit-identical under an affine transform, which doubles as the integrity check."

### 1.3 CV / gate discipline (Scigantic's own + the plan's gate for L1)

- "**Score folds individually before pooling; a diverged fold hides inside a pooled average.**"
  Worked example: a NaN-diverged fold at epoch 4 with healthy-looking predictions
  (mean **5.326** std **0.393** vs healthy **5.02–5.12** / **0.297–0.340**); only the per-fold
  score exposed it: **1.3763** vs **0.947–1.058**.
- Noise-floor reference given to us: "**Scigantic per-fold deviation flag 0.085–0.15 ST-RAE**";
  plan gate = "no fold deviating **>0.085** from pooled".
- "**And my +0.0913 was mostly fold noise… single-fold batch-64 probe vs 5-fold: +0.0292.
  One fold is not a measurement.**"
- "Out-of-fold numbers in the notebook are on the rows the weights were fitted on and
  **read optimistic**; the per-endpoint transfer from offline gain to board gain is measured in
  `docs/submission_history.md`."
- Rotation-CV gate (idea 1, the L1 gate): "**5-fold *rotation* CV (fit on 4/5 of OOF rows,
  score the held 1/5) — ship only if the held-out-fold ST-RAE gain is positive in ≥4/5 folds
  and mean gain > 0.004**; require Spearman of shipped column unchanged vs incumbent blend
  within rounding (the affine-restore integrity check)." Effort: "~1–2 days, CPU only."
- Weight-attenuation probe (R28, **built, not submitted**): scalar per-endpoint attenuation
  `c ∈ {1.1, 1.2, 2.5}` selected on OOF by 5-fold CV → "Macro from the correction on the three
  shipped columns: **+0.00202**". Caveats: "CYP3A4 does not reproduce between frames"
  (rotation **c=1.75** vs full-OOF **c=1.1**); "CYP2D6 and CYP3A4 have **sharp interior
  optima**… CYP2C9 and CYP1A2 are **broad and flat from about 2.0 to 3.0**."
- Column-independence tripwire: "the board scores four independent columns and re-scores an
  untouched column to the last digit" — **3/3 testable transitions PASS exactly**, and
  `4 × macro-delta = column-delta`.

### 1.4 Measured gains (exact)

- Progression: **R17 0.4620 → R24 0.4453 (joint 12-member hinge fit) → R25 0.4436**
  (TDI-arm blend into CYP1A2, w=0.2). So the hinge-fit step = **−0.0167** macro; the TDI-arm
  blend = **−0.0017**.
- Full history: R00 tutorial LightGBM **0.9358** → R01 (endpoint placement + multitask base)
  **0.5228** → R25 **0.4436**.
- Per endpoint, live ST-RAE of R25: **CYP1A2 0.4739, CYP2C9 0.3802, CYP2D6 0.5442, CYP3A4 0.3762**.
- Live as of 2026-10-19: "the best entry is **0.3537** and our live row is rank **25 of 177**,
  scoring **0.4456**" (their live row is an older upload, not R25).
- Plan's expectation for L1: "**Expect −0.010 to −0.017** (Scigantic's step)"; plan §5:
  "L1+L4 (CPU pair) is worth **~−0.012 to −0.020 ST-RAE (rank ~35-42)** if gates pass".

### 1.5 L1 pool context (their 12 members, for delta-space design)

Twelve members: **two backbones, both directed message-passing (chemprop) multi-task regressors
over the 22-column table, five scaffold folds each; 364 SageMaker training jobs run in total;
"these are the twelve that earned a place."**
- armA `d_h 700 / depth 5`, random init, MAE, inverse-variance sample weights, 22 heads
- armB `d_h 2048 / depth 4`, random init
- armC = B "message passing initialised from the CheMeleon checkpoint"
- armD = B "initialised from the organizers' CYP-finetuned CheMeleon checkpoint"
- `ivB2048` / `ivC2048` = "interval hinge on scored heads" with **uniform** weights
- `mu2048` = MAE uniform; `ii2048` = interval hinge + inverse-variance
- `mt_aux8` & `union18` = "recipe-level reproductions of the public SuperCowPowers Workbench CYP entry"
- `aux_ac50`, `cp_interval`
- "**CheMeleon supplies eight of the twelve.**" (foundation checkpoint zenodo **15460715**,
  organizer checkpoint `openadmet/cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1`)
- Hyperparameters: d_h 700/2048, depth 5/4, FFN 300/1024, dropout 0.1/0.2,
  "batch 128, 200 epochs / patience 30, 5 scaffold folds"; fold seed **42**; fixed `folds.npy`.
- Rejected legs: "MolFormer, ChemBERTa and Uni-Mol were fine-tuned and measured but did not
  earn a place in the stack." Mordred+GBT kept as "a non-neural check".

### 1.6 Training-loss 2×2 (L3 input, quoted in full)

Full matrix OOF ST-RAE (macro), 5 folds, arm B: **mae+invvar 0.7187, mae+uniform 0.7170,
interval+invvar 0.7422, interval+uniform 0.6895.**
- "**The interval objective has real independent value: +0.0275 at uniform weights**… It helps
  CYP1A2 (**+0.0300**), CYP2C9 (**+0.0384**) and CYP3A4 (**+0.0462**), and is neutral on CYP2D6
  (**−0.0044**)."
- "The interval objective helps under uniform weights and **HURTS under inverse-variance**
  [interaction **−0.0510**]". "interval+invvar is genuinely the worst cell in both experiments."
- Task grouping: "**G4 is far worse than G22. The 18 auxiliary heads are load-bearing**…
  **Dropping the 18 auxiliary heads costs about 0.23–0.30 macro.** Not a sweet spot; a cliff."

---

## 2. briford placement algebra — the L2 recipe

**Who:** briford (SuperCowPowers / ADMET Workbench) — **rank 20, macro ST-RAE 0.4378** "on the
live half of the test set". Code in `SuperCowPowers/workbench`, `ml_pipelines/OpenADMET/cyp/`.

### 2.1 R² decomposition (verbatim)

> "R² decomposes exactly: `R^2 = 2ρk − k² − b²` with ρ the Pearson correlation,
> `k = sd(pred)/sd(true)` the spread ratio, and `b` the mean offset in sd(true) units…
> **The optimum is k = ρ, not k = 1.** Matching the spread of the truth is wrong: a model with
> correlation ρ should be ρ times as wide as reality."

### 2.2 ST-RAE optimum ≠ R² optimum (verbatim)

> "The calibration that maximises R² does not minimise the scored metric. Soft-threshold RAE is
> zero anywhere inside a compound's credible interval, and low-activity compounds carry wide
> intervals, so **predicting high is nearly free while predicting low is punished by the
> actives**. The ST-RAE optimum therefore sits **above** the population centre, and on CYP2D6,
> the lowest-centred isoform, the two objectives point in opposite directions."

### 2.3 CYP2D6 board probes (same predictions, affine only) — exact numbers

Blog list: "**(3.57, 0.90) -> ST-RAE 0.5653   (3.32, 0.90) -> 0.6271   (3.11, 1.08) -> 0.6944**"
Code docstring: "CYP2D6 board probes at mean **3.11 / 3.32 / 3.57 / 4.35** score
**0.6944 / 0.6271 / 0.5502 / 0.8558**, so the minimum sits **between 3.57 and 4.35** and
**the penalty above it is far steeper than the gain below**."

- Adopted target: `STRAE_MOMENTS = {"CYP2D6": {"mean": 3.57, "sd": 0.90}}`
- R² across those placements: **0.363 → 0.447**.
- Spread fact used by the plan: **sd=0.90 ≪ ρ·sd ≈ 1.08** — i.e. for ST-RAE the spread target is
  **not** k=ρ.
- Bias warning (verbatim): "a weak vector minimises ST-RAE by hiding in the wide low-activity
  intervals, so an OOF-derived optimum is **biased high**. `sd` is the spread to hit **outright**,
  not `rho * sd`."
- Note a discrepancy in the source: mean 3.57 is reported as **0.5653** in the blog list and
  **0.5502** in the code docstring. Both are quoted in `open_top.md`; treat as board noise /
  two different vectors.
- Clamps: clamps create ties and "**a binding floor costs ranking**" (FLOOR = 1.0
  "Safety net only"). Plan: "ship best grid point, **clamps off** (ties cost ranking)".

### 2.4 Blind-moment solve from scored submissions

- `BLIND_MOMENTS` come from `solve_blind_moments`, "which reads **three scored submissions**".
- Exact equation (verbatim): "**Each board row is an exact equation:
  `R2*V = 2*cov(y,p) − var(p) − (mean(y) − mean(p))^2`, with V = var(y). Three unknowns… three
  entries spanning two spreads and two centres determine all three.**"
- Solved values (exact):
  - `BLIND_MOMENTS = {CYP1A2: mean 4.412 sd 1.553; CYP2C9: 4.830/1.101; CYP2D6: 3.107/1.599; CYP3A4: 4.880/1.272}`
  - `SOLVED_PEARSON = {1A2: 0.753, 2C9: 0.838, 2D6: 0.678, 3A4: 0.851}`
- Sensitivity: "**board noise of +-0.005 in R2 moves [rho] by roughly +-0.07**… The moments are
  far more robust." (⇒ mean/sd are trustworthy; ρ from the solve is soft.)

### 2.5 Probe submission design (the L2 3-slot plan, verbatim)

> "Upload the same submission, a variant at **1.15× spread about the same centre**, and a variant
> shifted **~+0.25 in centre at the new spread**; per-isoform solve `var(y)`, `mean(y)`,
> `cov(y,p)` with the published quadratic. ~2–3 days incl. 3 board cycles."

Cross-check rules / gate (verbatim):
> "require the physical root (**0.25 < var_y < 9.0**) and mean(y) within
> **[OOF-truth mean − 1.6, +0.5]**; cross-check solved Pearson against measured Spearman per
> isoform (must satisfy **ρ ≥ rank-correlation**); treat ρ as **±0.07** (briford's noise bound)."
Feeds idea 2's grid with the true target distribution. Plan adds: "also calibrates
`OOF_TO_BLIND` exactly."

### 2.6 Grid definition + centre cap (idea 2, verbatim)

- "on OOF rows evaluate the ST-RAE surrogate (hinge/credible interval) over an affine grid
  (**mean ∈ [μ_oof−0.6, μ_oof+0.6]**, **sd ∈ [0.6·ρ·sd, 1.1·sd]**); ship best grid point,
  clamps off (ties cost ranking). **<1 day.**"
- Gate (verbatim): "OOF surrogate is '**biased high**' (briford) — accept a grid point only if it
  wins on OOF **and** on the **analog-enriched CV slice** (blind set is hit-expansion clusters);
  probe with **one submission per isoform at most**, and **never ship a grid mean > OOF-truth
  mean + 0.75** without a supporting probe (0.4456-style over-correction risk, briford's
  **4.35 probe = 0.8558 disaster**)."
- ⇒ **Centre overshoot cap = OOF-truth mean + 0.75.**
- Their headline impact for this lever: "the R²-style placement they replaced moved CYP2D6
  **0.6944→0.5502**, i.e. **0.14 on one endpoint**".
- Our current shipped placement being replaced: "moments × `OOF_TO_BLIND` × `F_SPREAD` in
  `regression_final.py`", F_SPREAD 0.6–1.0 constant-spread.

### 2.7 briford OOF→blind ratios (exact)

"Scaffold-OOF Pearson **understates** blind Pearson — the split is harder than the blind half,
and an OOF prediction comes from one fold model where the endpoint averages five."
Measured on `cyp-reg-chemprop-mt-aux-100`: **OOF 0.570 / 0.681 / 0.408 / 0.795** against the
solved **0.753 / 0.838 / 0.678 / 0.851**
→ `OOF_TO_BLIND = {1A2: 1.32, 2C9: 1.23, 2D6: 1.66, 3A4: 1.07}`.
CV = **scaffold**; "The union's scaffold split runs over **31,670 rows**, so the challenge
compounds sit in folds alongside public chemistry where the control's folds were challenge-only."

---

## 3. stir_bar — non-negative / NNLS + confidence-weighted blending (L4)

**Who:** stir_bar / lachrymator — **rank 26 (live) & 28 (interim)**, interim macro ST-RAE
**0.465**; **TDI rank 6 of 114, Tier 1** (top tier = "the 26 entries the organisers could not
separate from first place"; live board best rank 14). Repo
`lachrymator/openadmet-cyp-challenge-public`, README **208 lines**, "**Write-up only… not the
training code**" — so no code, and "No seeds/hyperparameters disclosed".

### 3.1 What failed with ridge / elastic-net (verbatim, the L4 core lesson)

> "Residual correlations between members run **~0.55–0.88**… unconstrained ridge and elastic-net
> combiners returned **large cancelling coefficients: one member near +1.9 against another at
> −0.6**, and on one endpoint **the best standalone model pushed negative**… **one revision
> improved cross-validated error on all four endpoints while real held-out error got *worse* on
> three.** The stacker is now restricted to **non-negative combiners**."

- The stacker was built as: "Per-endpoint comparison of **ridge / elastic-net / non-negative /
  NNLS** combiners, **selected on held-out folds**." ⇒ NNLS/non-negative won on evidence, not
  by construction.
- **CV-up / worse-blind caution is the load-bearing fact**: 4 endpoints improved in CV, 3 of 4
  got worse in real held-out, in a single revision.
- **CV→blind transfer, quantified**: "on **regression**, **100% of submission-endpoint pairs came
  out worse on the blind set than in cross-validation (median MAE +0.208)**; on TDI, **82%**…
  (median MCC **−0.042**)". "Pooled, MCC reads **r = 0.90**, while neither endpoint on its own
  exceeds **0.69**."
- CV scheme that made numbers honest: "**Butina cluster-disjoint folds** — clusters assigned whole
  to a single fold, so no analogue series spans train and validation." "**Random splits flattered
  every model. Butina folds made the held-out numbers track the leaderboard**, which is what
  made iteration meaningful at all."
- No ST-RAE CV counterpart exists: "Regression is shown as MAE rather than the board's own
  ST-RAE: **that statistic is the organisers' and has no cross-validated counterpart here.**"
- Other stacker hygiene relevant to a 19-member pool:
  - "Each embedding source is **PCA-capped** before it reaches a tabular model. With only **one
    to two thousand labelled rows per endpoint**, an uncapped concatenation of several thousand
    embedding dimensions lets those models **memorise the fold**."
  - "Pruning by **correlation, not by count**… on the classification track **the direct encoder
    heads individually outranked every tabular embedding leg, and including all of them produced
    the worst result on record for one endpoint.**"
  - "Down-weighted calculated features. A property computed from structure alone carries a label
    on every row, so at full weight it supplies the large majority of the pretraining gradient…
    Those heads are down-weighted as a per-task weight — **a constant per-row weight cancels out
    of a weighted mean and changes nothing**."
  - TabPFN omitted: "Across **seven** matched embedding sources **Mitra beat TabPFN on macro
    held-out R² in six**, and the two models' **residuals correlated 0.94–0.98** — they were
    making very nearly the same mistakes, so leaving one out cost almost no ensemble diversity."
  - "Calibration is separable from ranking. Several learners ranked compounds well while being
    badly scaled; **letting the stacker recalibrate recovered most of that**." No spread-factor /
    conformal / clipping details published.
  - External: "**88,683 near-neighbours of the held-out structures**" retrieved, filtered to the
    held-out physicochemical envelope; "**The 4,999 closest** are admitted to the training pool,
    entering with only their calculated physical properties and no assay endpoints"; similarity
    floor "chosen so that every admitted compound is closer to the scored molecules than the
    training set's own average nearest neighbour"; rd_filters with ~1,150 named alerts,
    "**any alert that fires on even one held-out compound is dropped (~75 of 1249)**".

### 3.2 What the confidence weights ARE (from briford, not stir_bar)

Verbatim (`scripts/cyp_ensemble_submit.py`):
> "Members are weighted per compound by **calibrated confidence**, not averaged flat. Each
> member's confidence is scaled by its own **|conf-to-error correlation|**… Out of fold that
> beats the flat mean on **every isoform by more than the isoform resolves**:
> CYP1A2 **0.6074→0.6569 (+0.0495)**, CYP2C9 **0.7307→0.7832 (+0.0525)**,
> CYP2D6 **0.4949→0.5583 (+0.0634)**, CYP3A4 **0.8321→0.8574 (+0.0253)**."

- Confidence definition: "Confidence = **5-fold-CV ensemble std, conformal-calibrated**"
  (Workbench `docs/confidence/index.md`: "5-model ensemble… **std across 5 predictions**…
  **Conformal scaling**").
- Plan's L4 restatement: "per-compound weights ∝ |conf-to-error correlation| from 5-fold CV std
  beat flat mean on all four isoforms beyond resolution, +0.025…+0.063 Spearman OOF";
  "We already have per-member 5-fold scaffold predictions → **free per-compound std**. Swap our
  ridge blend for **confidence-weighted NNLS (delta-space, non-negative, L2) per isoform**.
  ~1 day, no GPU."
- Note: all four gains exceed briford's own per-isoform resolution thresholds
  (0.043 / 0.031 / 0.056 / 0.018 for A1/C9/D6/A4) — that is what "beyond the isoform resolves" means.
- L4 gate (verbatim): "paired bootstrap over OOF rows on the blended column: **95% CI of ST-RAE
  delta excludes 0 AND direction agrees with idea 1's hinge-fit blend (they must not disagree,
  since they share the same pool)**; ship only through idea 2's moment-restore so no placement
  change rides along."
- **Sign-constraint conflict to resolve explicitly in code**: Scigantic's hinge fit says
  "negative weights are allowed"; stir_bar's failure mode says non-negative. Plan's resolution:
  L1 = hinge fit (deltas, L2, negatives allowed) and L4 = NNLS non-negative — two fits on the
  **same pool** that must **agree in direction** as a gate condition.

### 3.3 briford diversity/ensemble facts relevant to pool composition

- "On CYP2D6, where all four members predict and the pool can be scored out of fold, **the
  average beats every member it contains — 0.503 Spearman against 0.445 for the best of them —
  and the gain flattens by the fourth.**"
- TabICL leg: "TabICL over a **720-d [Monroe] molecular embedding**" (`cyp_tabicl_monroe.py`),
  top-variance **360 dims** default; CYP2D6 OOF rho **0.448** vs best chemprop **0.445**, and
  "the first change to move" the AUC <4.0 vs ≥4.5 discrimination (**0.799 vs 0.763**).
- Auxiliary-head gradient-share math (matters if the pool gains aux heads): "A task's gradient
  share is its **label count times its weight**, not its share of the task list… The public
  targets carry **137,375** labels against the four scored targets' **6,525**, which is why a 0.3
  multiplier on them is not a modest thing: `--public-weight 0.05` → scored 27% assay 42% public
  30%; `--public-weight 0.30` → scored 11% assay 17% **public 72%**." (`AUX_WEIGHT = 0.3`.)

---

## 4. Per-isoform transfer record + noise floors (gate constants)

### 4.1 Scigantic directional transfer record (their "headline discipline result")

> "The full reconstructable record is **13 of 20 genuine offline directional calls correct
> (65%…)**… **radically non-uniform across endpoints**"

| Isoform | Calls right | Mean signed transfer ratio |
|---|---|---|
| CYP3A4 | **6 / 6** | **+0.82** |
| CYP2C9 | **3 / 4** | **+0.40** |
| CYP1A2 | **2 / 5** | **+0.24** |
| CYP2D6 | **2 / 5** | **−0.18** |

CYP2D6 verdict (verbatim): "**the historical expectation of acting on a CYP2D6 offline gain is to
make the column worse.**"
Plan directive: "invest changes on **CYP3A4/CYP2C9 first** (6/6, 3/4 call accuracy) and treat
CYP2D6/CYP1A2 offline wins as unmeasured (2/5, 2/5)"; "any CYP2D6-only gain gets the strictest
gate (2/5 correct, signed −0.18) — **demand 4/5 folds**"; "per-isoform decision (2D6 strictest)".

### 4.2 Noise floors (all of them, exact)

- **Seed noise floor (briford, from `cyp_seed_noise.py`, 3 replicates of one config, seeds `-s43`,
  `-s44`): "0.05 Pearson on CYP1A2 and CYP2D6, 0.013 and 0.012 on CYP2C9 and CYP3A4."**
- **Resolution thresholds: 0.043 / 0.031 / 0.056 / 0.018 (A1 / C9 / D6 / A4).**
- Ruler-power conclusion (verbatim): "Repeated training runs of one configuration at different
  seeds **disagree by more than candidate models do**, so out-of-fold cross-validation **cannot
  resolve the differences now being chased**."
- **Scigantic per-fold deviation flag: 0.085–0.15 ST-RAE** (gate: no fold >0.085 off pooled).
- **Board noise: ±0.005 in R² ⇒ ρ moves ±0.07**; moments far more robust.
- Single-fold vs 5-fold: an apparent **+0.0913** was **+0.0292** at 5 folds. "One fold is not a
  measurement."
- stir_bar CV→blind optimism: **100% of regression submission-endpoint pairs worse blind,
  median MAE +0.208**.

### 4.3 Reference board scores (context for gates)

| Entry | Rank | Macro ST-RAE | Per-isoform ST-RAE |
|---|---|---|---|
| rasayan-labs | 10 | 0.4234 | n/a (repo **empty**; no CYP report exists as of Oct 22) |
| briford | 20 | **0.4378** (live half) | CYP2D6 probes 0.5502–0.6944, 4.35 probe = 0.8558 |
| Scigantic R25 | 11 of 219 (interim 2026-09-25) | **0.4436** | 1A2 **0.4739**, 2C9 **0.3802**, 2D6 **0.5442**, 3A4 **0.3762** |
| stir_bar | 28 interim (Tier 6) / 28 of 230 live ("top 12%") | **0.465** (MAE 0.595, R² 0.747) | 1A2 **0.441**, 2C9 **0.380**, 2D6 **0.653**, 3A4 **0.386** |
| best live entry (2026-10-19) | — | **0.3537** | Scigantic live row rank 25 of 177 at 0.4456 |

stir_bar CV out-of-fold: macro MAE **0.447** / R² **0.688**
(1A2 0.429/0.671; 2C9 0.327/0.667; 2D6 0.712/0.585; 3A4 0.320/0.828).
stir_bar TDI: interim macro MCC **0.380**, Accuracy **0.840**, F1 **0.484**; CV OOF CYP2D6 MCC
**0.270** / AUC **0.719**; CYP3A4 MCC **0.554** / AUC **0.864**.

---

## 5. Not present in `open_top.md` (gaps for `hinge_blend.py`)

1. **L2 penalty coefficient** for Scigantic's hinge fit — the doc quotes "plus an L2 penalty" with
   no λ. Not in Scigantic's quoted text.
2. **Powell settings**: no starting point, no bounds, no tolerances. Only "Powell minimisation".
3. **Definition of "the incumbent"** in the delta parameterization (which blend column Scigantic
   treats as incumbent). Plan says "Replace the ridge objective on the cp17 pool" and "z-restore
   to incumbent blend mean/sd" — ours = the 19-member ridge blend.
4. **Interval-hinge training-loss implementation** (L3) details beyond `hinge = clip(lo-p,0)+clip(p-hi,0)`
   applied to scored heads; `bounded_loss` (censored `IC50 > x` + `_lt` flag) "has not moved the
   scored metric" (briford).
5. rasayan-labs: repo is **empty** (`whis9/rasayan-cyp`: 0 branches, 0 tags, 409 "Git Repository is
   empty"); their rank-10/0.4234 numbers are from our leaderboard copy, not a publication. Their
   PXR report is method house-style only (median QR-L1 stacking cascade, isotonic, Butina 5-fold,
   "both-metrics validation gate", "the tail is irreducible", "accurate-XOR-orthogonal wall…
   every candidate was either accurate-but-redundant (corr > 0.83) or orthogonal-but-weak
   (solo RAE > 0.7) — never both").
6. Stir_bar publishes **no code, seeds, or hyperparameters** ("Write-up only"), and publishes no
   spread-factor/conformal/clipping calibration detail.
