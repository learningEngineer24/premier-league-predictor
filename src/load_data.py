"""Load and clean Premier League match data from football-data.co.uk CSVs.

The raw files are named E0_YYZZ.csv (e.g. E0_2122.csv = 2021-22 season) and use
the standard football-data.co.uk column layout, encoded as UTF-8 with BOM.

This module turns them into ONE tidy table with:
    season, matchweek, date, home_team, away_team,
    home_goals, away_goals, result (H/D/A),
    avg_h, avg_d, avg_a          <- bookmaker average odds (BENCHMARK ONLY)

Matchweeks are not in the raw files, so we reconstruct them: within a season,
games sorted by date are grouped into chronological blocks of 10 (20 teams ->
10 games per matchweek). This is exact for complete seasons and for the
partial 2026-27 season (50 games = matchweeks 1-5, all fully played).

The bookmaker columns are kept purely so the backtest can compare the model
against bookmaker-implied probabilities. They must NEVER be used as model
features -- that would turn the model into an odds-cloner.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

_SEASON_RE = re.compile(r"E0_(\d{2})(\d{2})\.csv$")

# Columns we keep. Everything else (shots, corners, cards, xG, ...) is ignored
# for v1 -- the model only needs the final scores.
_KEEP_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
              "AvgH", "AvgD", "AvgA"]


def season_label(filename: str) -> str:
    """'E0_2122.csv' -> '2021-22'."""
    m = _SEASON_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse season from filename: {filename}")
    return f"20{m.group(1)}-{m.group(2)}"


def load_season(path: Path) -> pd.DataFrame:
    """Load one season CSV into a clean, chronologically sorted DataFrame."""
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=_KEEP_COLS)

    # --- parse ---
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.rename(columns={
        "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals",
        "FTR": "result",
        "AvgH": "avg_h", "AvgD": "avg_d", "AvgA": "avg_a",
    })
    df["season"] = season_label(path.name)

    # --- sanity checks (fail loudly; silent bad data is worse than a crash) ---
    key = ["season", "home_team", "away_team"]
    assert df[key].notna().all().all(), "missing team names"
    assert df[["home_goals", "away_goals"]].notna().all().all(), "missing scores"
    assert df["result"].isin(["H", "D", "A"]).all(), "unexpected result codes"
    assert not df.duplicated(subset=key).any(), "duplicate fixtures found"

    # result code must agree with the scoreline (guards against corrupt rows)
    implied = np.where(df["home_goals"] > df["away_goals"], "H",
               np.where(df["home_goals"] < df["away_goals"], "A", "D"))
    assert (df["result"].to_numpy() == implied).all(), \
        "result code disagrees with scoreline"

    # --- chronological order + matchweek reconstruction ---
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    # 10 games per matchweek; integer division of the chronological position.
    df["matchweek"] = (np.arange(len(df)) // 10) + 1

    return df[["season", "matchweek", "date", "home_team", "away_team",
               "home_goals", "away_goals", "result",
               "avg_h", "avg_d", "avg_a"]]


def load_all(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every E0_*.csv in raw_dir into one clean table, oldest first."""
    files = sorted(raw_dir.glob("E0_*.csv"))
    if not files:
        raise FileNotFoundError(f"No E0_*.csv files in {raw_dir}")
    df = pd.concat([load_season(f) for f in files], ignore_index=True)
    df = df.sort_values(["date", "season"]).reset_index(drop=True)

    # Global leakage guard: the table must be fully chronological so that
    # "train on everything before date X" can never see the future.
    assert df["date"].is_monotonic_increasing, "dates are not chronological"
    return df


if __name__ == "__main__":
    df = load_all()
    print(f"{len(df)} games, {df['season'].nunique()} seasons, "
          f"{pd.concat([df['home_team'], df['away_team']]).nunique()} teams")
    print(df.groupby("season").agg(games=("result", "size"),
                                   matchweeks=("matchweek", "max"),
                                   first=("date", "min"),
                                   last=("date", "max")))
