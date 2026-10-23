# PROMPT FOR NEXT AGENT (Oct 22) - FINAL PUSH, closes Nov 3

State: all model levers are CLOSED (see NOTES.md sec 24-28: x_d, pool
simplify, 3D SchNet, LLM pharmacophore, docking features, pKa/stereo probe,
pseudo-labeling, qHTS-DA, SMILES-transformer, dead-zone, TabPFN already in).
2D6 is a representation ceiling (sec 26). Standing candidates (rank 63
regression / 38 TDI on Oct 3+ boards):
  - regression cp17: cache/regression_final_cp17_submission.csv (0.4682)
  - TDI tdi_v5:      cache/tdi_submission_v5_candidate.csv (0.3422)
Both md5-verified reproducible: docs/REPRODUCTION.md.

Task list for the final push:
1. Only ship on a GATE PASS (new family delta > +0.004 pool / TDI > 0.3448
   nested). A no-op is the expected default; do not force submissions.
2. If any new lever passes: rebuild via REPRODUCTION.md, run
   src/verify_submissions.py (official validators), report numbers to Jackson.
3. NEVER submit (needs Jackson's HF account + explicit go-ahead).
4. Honest nested recipe for every gate (see REPRODUCTION.md harness section);
   log results to NOTES.md as new sections; commit at each gate.
5. One GPU job at a time; background long runs; never kill running processes.

Read first: NOTES.md sec 24-28 (closed levers + ceiling diagnosis),
docs/REPRODUCTION.md, leaderboard/ Oct 3 boards. Dead-lever list is in
docs/archive/RANK_UP_PLAN.md + NOTES sec 22/25/28 - do not re-litigate.
