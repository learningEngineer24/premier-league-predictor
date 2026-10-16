"""Pull historical starting lineups from ESPN's free scoreboard/summary API.

Why: v3 adjusts team strength for the observable absence of regular
starters, impact-weighted by minutes/xG contribution. That needs
backtestable lineups -- who started each game -- for the tuning window
(2024-25/2025-26) and the current season to date.

Source: site.api.espn.com/apis/site/v2/sports/soccer/eng.1 -- free, no key.
Per season: discover the 20 teams via opening-weekend scoreboards (everyone
plays), enumerate fixtures via per-team schedules, then one summary call per
fixture for rosters with starter flags. Polite crawl: ~1s between calls,
backoff on 429/5xx.

Output: data/lineups/lineups_<season_label>.csv, one row per player
appearance: season, date, event_id, home_team, away_team, team, player,
starter (1/0), position, formation. Names are ESPN strings; mapping to
football-data.co.uk names is a separate step in the v3 build.

Usage: cd src && python3 pull_lineups.py [--seasons 2024,2025,2026]
Season codes are the year the season starts: 2024 = 2024-25.
"""

import argparse
import csv
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "lineups"

# Opening-weekend dates (Fri-Mon) per ESPN season code; all 20 teams play.
OPENING_WEEKENDS = {
    2024: ["20240816", "20240817", "20240818", "20240819"],
    2025: ["20250815", "20250816", "20250817", "20250818"],
    2026: ["20260821", "20260822", "20260823", "20260824"],
}

SEASON_LABEL = {2024: "2024-25", 2025: "2025-26", 2026: "2026-27"}


def get_json(url, tries=4):
    """GET with polite retry on rate limits / transient errors."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return __import__("json").load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"failed after {tries} tries: {url}")


def discover_teams(season):
    """All team {id: name} for a season, via opening-weekend scoreboards."""
    teams = {}
    for day in OPENING_WEEKENDS[season]:
        d = get_json(f"{BASE}/scoreboard?dates={day}&limit=20")
        time.sleep(1.0)
        for e in d.get("events", []):
            for c in e["competitions"][0]["competitors"]:
                teams[c["team"]["id"]] = c["team"]["displayName"]
    assert len(teams) == 20, \
        f"season {season}: found {len(teams)} teams, expected 20"
    return teams


def team_fixtures(team_id, season):
    """{event_id: (date, home, away)} for one team's season schedule."""
    d = get_json(f"{BASE}/teams/{team_id}/schedule?season={season}")
    time.sleep(1.0)
    out = {}
    for e in d.get("events", []):
        comps = e["competitions"][0]["competitors"]
        home = next(c["team"]["displayName"] for c in comps
                    if c["homeAway"] == "home")
        away = next(c["team"]["displayName"] for c in comps
                    if c["homeAway"] == "away")
        out[e["id"]] = (e["date"][:10], home, away)
    return out


def game_lineups(event_id):
    """Roster rows for one fixture; empty list if not yet played."""
    d = get_json(f"{BASE}/summary?event={event_id}")
    time.sleep(1.0)
    rosters = d.get("rosters") or []
    if not rosters:
        return []
    comps = d["header"]["competitions"][0]["competitors"]
    home = next(c["team"]["displayName"] for c in comps
                if c["homeAway"] == "home")
    away = next(c["team"]["displayName"] for c in comps
                if c["homeAway"] == "away")
    rows = []
    for t in rosters:
        team = t["team"]["displayName"]
        formation = t.get("formation", "")
        for p in t["roster"]:
            rows.append({
                "event_id": event_id, "home_team": home,
                "away_team": away, "team": team,
                "player": p["athlete"]["displayName"],
                "starter": 1 if p.get("starter") else 0,
                "position": (p.get("position") or {}).get("abbreviation", ""),
                "formation": formation,
            })
    return rows


def pull_season(season):
    label = SEASON_LABEL[season]
    print(f"[{label}] discovering teams...", flush=True)
    teams = discover_teams(season)

    print(f"[{label}] enumerating fixtures...", flush=True)
    fixtures = {}
    for tid in teams:
        fixtures.update(team_fixtures(tid, season))
    today = datetime.now(timezone.utc).date().isoformat()
    played = {eid: f for eid, f in fixtures.items() if f[0] < today}
    print(f"[{label}] {len(teams)} teams, {len(fixtures)} fixtures, "
          f"{len(played)} played", flush=True)
    if season in (2024, 2025):
        assert len(played) == 380, f"expected 380 games, got {len(played)}"

    rows = []
    for i, (eid, (date, home, away)) in enumerate(sorted(played.items())):
        for r in game_lineups(eid):
            rows.append({"season": label, "date": date, **r})
        if (i + 1) % 50 == 0:
            print(f"[{label}] {i + 1}/{len(played)} games", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2024,2025,2026")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for season in [int(s) for s in args.seasons.split(",")]:
        rows = pull_season(season)
        path = OUT_DIR / f"lineups_{SEASON_LABEL[season]}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "season", "date", "event_id", "home_team", "away_team",
                "team", "player", "starter", "position", "formation"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} player-appearances)", flush=True)


if __name__ == "__main__":
    main()
