# Leaderboard Board Analysis — Oct 22, 2026 snapshot

Source: `doc_9f7f5f16ecc3_regression_leaderboard_8yeuyd5v.csv` — 243 entries (regression leaderboard).
Our position: **rank 51** (JacksonBurns), ST-RAE 0.5133, R2 0.5219, rho 0.7245.
Field fit (given): ST-RAE ≈ −0.743·R2 − 0.748·rho + 1.459, resid std 0.030.
Fit validated on rank ≤ 100: residual mean -0.0067, std 0.0334 (n=100). Full-board residual std 0.1705 — fit is top-of-field only; OLS refit on all 243: ST-RAE ≈ -2.549·rho -0.597·R2 + ... (deep-tail entries dominate).

## 1. Rank sensitivity near rank 51

Neighbors (rank, username, ST-RAE):

| rank | username | ST-RAE | ΔST-RAE vs rank 51 |
|---|---|---|---|
| 45 | Moyu | 0.4897 | -0.0237 |
| 46 | Cheminfo | 0.4909 | -0.0224 |
| 47 | GiantWizard | 0.4938 | -0.0196 |
| 48 | HemeSight | 0.4963 | -0.0170 |
| 49 | CYPNet | 0.5016 | -0.0117 |
| 50 | SVM | 0.5106 | -0.0028 |
| 51 | JacksonBurns | 0.5133 | +0.0000 |
| 52 | nova | 0.5137 | +0.0004 |
| 53 | deoxys | 0.5168 | +0.0035 |
| 54 | thinkMolecular | 0.5183 | +0.0050 |
| 55 | jeremy | 0.5185 | +0.0052 |
| 56 | KalenJosifovski | 0.5276 | +0.0143 |
| 57 | ÇÝPŔËŠŠ ĤÎŁŁ | 0.5290 | +0.0157 |

- Local slope d(rank)/d(ST-RAE) over ranks [46,56] (LSQ): **272.6 ranks per ST-RAE unit** (≈ 0.37 pp ST-RAE per rank).
- Local slope d(rank)/d(ST-RAE) over ranks [41,61] (LSQ): **290.0 ranks per ST-RAE unit** (≈ 0.34 pp ST-RAE per rank).
- Central-difference slope from ranks 50/52: **636.4 ranks per ST-RAE unit** (0.5106 @rank50, 0.5137 @rank52).

ST-RAE needed to reach target ranks (must beat the entry currently at that rank):

| target rank | username at rank | their ST-RAE | required ΔST-RAE (from 0.5133) | implied rank gain @ local slope |
|---|---|---|---|---|
| 45 | Moyu | 0.4897 | **-0.0237** | 6 |
| 40 | tajfun | 0.4784 | **-0.0350** | 11 |
| 35 | dcv | 0.4618 | **-0.0515** | 16 |
| 30 | admetchallenger | 0.4512 | **-0.0621** | 21 |
| 25 | C_CYPher-cypext-B1cal-restore | 0.4428 | **-0.0705** | 26 |
| 20 | briford | 0.4378 | **-0.0755** | 31 |
| 10 | rasayan-labs | 0.4234 | **-0.0900** | 41 |
| 1 | preheat-to-450 | 0.3554 | **-0.1579** | 50 |

Average rank gain per 0.01 ST-RAE improvement near us (ranks 45–60 LSQ): **2.78 ranks**.

## 2. Iso-ST-RAE frontier (field fit inverted: R2* = (1.459 − 0.748·rho − T) / 0.743 at target ST-RAE = T)

Required R2 to sit ON the fit line at target ST-RAE:

| rho | R2 @ ST-RAE 0.4512 (rank 30) | R2 @ ST-RAE 0.4234 (rank 10) |
|---|---|---|
| 0.7245 | **0.6270** | **0.6644** |
| 0.7300 | **0.6215** | **0.6589** |
| 0.7400 | **0.6114** | **0.6488** |
| 0.7500 | **0.6013** | **0.6388** |
| 0.7600 | **0.5913** | **0.6287** |
| 0.7700 | **0.5812** | **0.6186** |

Sanity: our point (R2 0.5219, rho 0.7245) fits to ST-RAE 0.5293 vs actual 0.5133 (resid -0.0160).

## 3. Placement residual leaders (R2 ∈ [0.45, 0.60], most-negative resid = actual − field-fit)

| rank | username | R2 | rho | ST-RAE actual | field-fit | resid |
|---|---|---|---|---|---|---|
| 63 | reddy_1414 | 0.4529 | 0.6971 | 0.5547 | 0.6011 | **-0.0464** |
| 56 | KalenJosifovski | 0.4828 | 0.7221 | 0.5276 | 0.5601 | **-0.0325** |
| 36 | tguttenb1 | 0.5708 | 0.7222 | 0.4672 | 0.4947 | **-0.0275** |
| 33 | Ligand Pro | 0.5709 | 0.7333 | 0.4594 | 0.4864 | **-0.0270** |
| 37 | molairity | 0.5526 | 0.7411 | 0.4694 | 0.4941 | **-0.0248** |
| 61 | Omu | 0.4890 | 0.7039 | 0.5449 | 0.5692 | **-0.0243** |
| 31 | suneelbvs | 0.5597 | 0.7553 | 0.4540 | 0.4782 | **-0.0242** |
| 44 | CoSy | 0.5481 | 0.7253 | 0.4865 | 0.5092 | **-0.0227** |

Best placement-only gain observed at comparable R2 band: **-0.0464** ST-RAE (1.5σ of field residual). Our resid -0.0160.
(Band size n=51, resid 5th pct -0.0272, median -0.0060.)

## 4. Pure-ranking entries (rho > 0.74 AND R2 < 0.55)

| rank | username | rho | R2 | ST-RAE |
|---|---|---|---|---|
| 40 | tajfun | 0.7471 | 0.5390 | 0.4784 |
| 50 | SVM | 0.7665 | 0.4939 | 0.5106 |
| 68 | tmp1234 | 0.7578 | 0.3772 | 0.5798 |
| 76 | Biomni-TusoAI | 0.7488 | 0.3983 | 0.6147 |
| 94 | ChemSpace | 0.7459 | 0.2166 | 0.7104 |
| 97 | NIPERKOLKATA_PI_LAB | 0.7497 | 0.3901 | 0.7331 |
| 105 | Asidsal11 | 0.7415 | 0.1734 | 0.7482 |

Count: 7.
Best ST-RAE among them: 0.4784; median 0.6147 (compare ours 0.5133 at rho 0.7245).

## 5. Submission timing — top 20

Snapshot time (latest submission in file): **2026-10-23 00:52 UTC**. 72h cutoff: 2026-10-20 00:52 UTC.

| rank | username | submitted | age @ snapshot (h) | in last 72h? |
|---|---|---|---|---|
| 1 | preheat-to-450 | 2026-10-22 16:47 | 8 | YES |
| 2 | as_pen_4 | 2026-10-22 13:09 | 12 | YES |
| 3 | cypadeedoodah | 2026-10-22 16:23 | 8 | YES |
| 4 | wbot | 2026-10-22 19:47 | 5 | YES |
| 5 | grapefruit | 2026-10-22 18:00 | 7 | YES |
| 6 | hamza322 | 2026-10-18 13:27 | 107 | no |
| 7 | cyp-3po | 2026-10-22 07:34 | 17 | YES |
| 8 | EleanorRigby | 2026-09-24 13:31 | 683 | no |
| 9 | hfger | 2026-10-20 14:37 | 58 | YES |
| 10 | rasayan-labs | 2026-10-22 23:06 | 2 | YES |
| 11 | Queimo | 2026-10-22 16:32 | 8 | YES |
| 12 | furtivepeony | 2026-10-20 00:30 | 72 | no |
| 13 | z | 2026-10-19 21:20 | 76 | no |
| 14 | deCYPher | 2026-10-22 16:11 | 9 | YES |
| 15 | curiousone | 2026-09-25 21:30 | 651 | no |
| 16 | cypcyp | 2026-10-19 20:43 | 76 | no |
| 17 | beast | 2026-10-23 00:06 | 1 | YES |
| 18 | ADMETForge | 2026-10-12 03:48 | 261 | no |
| 19 | first-chair | 2026-09-14 17:51 | 919 | no |
| 20 | briford | 2026-08-30 15:14 | 1282 | no |

- Top 20 in last 72h: **11/20**. Top 7: 6/7. Ranks 8–20: 5/13.
- Top-20 submission window: 2026-08-30 15:14 → 2026-10-23 00:06 UTC.
- Whole board submitted in last 72h: 65/243 (27%); on Oct 22 (board): 37, Oct 22 (top 20): 9.
- Early (pre-cutoff) top-20: #6 hamza322, #8 EleanorRigby, #12 furtivepeony, #13 z, #15 curiousone, #16 cypcyp, #18 ADMETForge, #19 first-chair, #20 briford.

