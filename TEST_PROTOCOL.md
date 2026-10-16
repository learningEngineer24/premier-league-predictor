# Test protocol (locked Oct 15, 2026)

## The locked test set: 2025-26

- **Results:** `data/raw/E0_2526.csv` — 380 games, full season.
- **Lineups:** `data/lineups/lineups_2025-26.csv` — 380 games,
  exactly 22 starters per game, verified.
- **Eval window:** weeks 5–38 (340 games). Weeks 1–4 are excluded
  everywhere: early-season models have no signal (no established
  regulars, no xG history, tiny samples).

## The rule

1. **Tune and select on 2024-25** (walk-forward), secondarily on the
   live 2026-27 window (with the selection caveat: it's the live
   track, so it informs but never decides).
2. **2025-26 gets exactly ONE final evaluation per candidate
   feature/model.** Paired bootstrap CIs, same as always.
3. A test set peeked at repeatedly is a tuning set. The two Oct 15
   absence nulls (60% and 75% regular thresholds) predate the lock
   and selected nothing — the exam count starts at zero from here.

## Why 2025-26 and not an untouched season

- Data already in hand (results + lineups); most recent full season,
  closest to current conditions.
- The prior peeks returned nulls and selected nothing, so
  selection-bias risk going forward is minimal under the one-exam rule.
- Alternative considered Oct 15: pulling 2023-24 as a virgin test set.
  Deferred — revisit only if 2025-26 ever gets over-examined.
