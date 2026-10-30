"""Threads preview graphic for a matchweek — same style as the MW6 graphic.

Reads outputs/matchweek{N}_predictions.csv and renders a dark-navy portrait
card stack: gold eyebrow, big title, per-fixture cards with pick pill,
H/D/A probability bar, and a one-line comment.

Usage: cd src && python3 make_preview_graphic.py --week 7
Writes outputs/mwN_preview.png
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

SRC_DIR = Path(__file__).resolve().parent
OUT_PATH = SRC_DIR.parent / "outputs"
FONT_DIR = "/usr/share/fonts/truetype/dejavu"

# --- palette (matched to the MW6 graphic) ---
BG = (11, 21, 38)
CARD_BG = (17, 29, 51)
CARD_BORDER = (32, 48, 82)
GOLD = (232, 184, 75)
GOLD_DIM = (217, 164, 65)
WHITE = (245, 247, 250)
GRAY = (148, 163, 184)
GRAY_DIM = (110, 124, 148)
GREEN_BAR = (46, 204, 113)
GREEN_TXT = (53, 208, 127)
DRAW_BAR = (150, 115, 58)
DRAW_TXT = (217, 164, 65)
RED_BAR = (232, 105, 105)
RED_TXT = (232, 105, 105)
ORANGE = (249, 115, 22)
TRACK = (26, 39, 66)
DARK_TXT = (11, 21, 38)

W = 1080
PAD = 48


def font(name, size):
    return ImageFont.truetype(f"{FONT_DIR}/{name}", size)


F_TITLE = font("DejaVuSans-Bold.ttf", 96)
F_EYEBROW = font("DejaVuSans-Bold.ttf", 30)
F_SUB = font("DejaVuSans-Bold.ttf", 40)
F_SUB2 = font("DejaVuSans.ttf", 30)
F_CARD_TITLE = font("DejaVuSans-Bold.ttf", 38)
F_CARD_TITLE_SM = font("DejaVuSans-Bold.ttf", 32)
F_TAG = font("DejaVuSans-Bold.ttf", 21)
F_PILL = font("DejaVuSans-Bold.ttf", 27)
F_LABEL = font("DejaVuSans-Bold.ttf", 27)
F_COMMENT = font("DejaVuSans.ttf", 27)
F_FOOTER = font("DejaVuSans.ttf", 28)


def tracked_text(draw, xy, text, fnt, fill, tracking=8):
    """Draw text with manual letter-spacing; returns total width."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        w = draw.textbbox((0, 0), ch, font=fnt)[2]
        x += w + tracking
    return x - xy[0] - tracking


def text_w(draw, text, fnt):
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0]


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if text_w(draw, t, fnt) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def pill(draw, xy, text, fnt, fill, text_fill, outline=None, pad_x=18, pad_y=9):
    x, y = xy
    tw, th = text_w(draw, text, fnt), fnt.size
    w, h = tw + pad_x * 2, th + pad_y * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2,
                           fill=fill, outline=outline, width=3)
    draw.text((x + pad_x, y + pad_y - 2), text, font=fnt, fill=text_fill)
    return w


def accent_for(pick_pct):
    if pick_pct >= 60:
        return GREEN_BAR
    if pick_pct >= 40:
        return GOLD
    return ORANGE


# (week, date_range) -> per-fixture (tag, comment); tag in {BANKER, COIN FLIP, BIG ONE, None}
META = {
    7: {
        "dates": "Oct 17–19, 2026",
        "fixtures": [
            ("Everton", "Chelsea", None,
             "Everton slight favorites at home — but Chelsea rarely go quietly."),
            ("Brentford", "Liverpool", None,
             "Brentford edge it at home — Liverpool's attack keeps it close."),
            ("Fulham", "Hull", "COIN FLIP",
             "A genuine coin flip — 34.0% plays 34.5%."),
            ("Man City", "Ipswich", "BANKER",
             "The banker of the week. City at the Etihad."),
            ("Newcastle", "Aston Villa", None,
             "Newcastle favored at home; Villa's draw threat is real at 27%."),
            ("Bournemouth", "Sunderland", None,
             "Bournemouth's attack should prove too much for Sunderland."),
            ("Brighton", "Crystal Palace", None,
             "Brighton's home form does the talking — 63%."),
            ("Leeds", "Man United", "COIN FLIP",
             "Derby day, and the model can barely split them."),
            ("Nott'm Forest", "Arsenal", "BIG ONE",
             "The big one — Arsenal clear favorites on the road."),
            ("Tottenham", "Coventry", "COIN FLIP",
             "Spurs favored, but Coventry have shocked before."),
        ],
    },
    8: {
        "dates": "Oct 23–25, 2026",
        "fixtures": [
            ("Ipswich", "Nott'm Forest", "COIN FLIP",
             "Tonight's opener is a true coin flip — v2 leans Ipswich, Elo leans Forest."),
            ("Aston Villa", "Man City", "BIG ONE",
             "City favored on the road — but Villa Park is no easy trip."),
            ("Arsenal", "Everton", "BANKER",
             "The banker of the week. Arsenal at the Emirates."),
            ("Coventry", "Fulham", "COIN FLIP",
             "Home, draw and away all within seven points."),
            ("Chelsea", "Tottenham", "BIG ONE",
             "The London derby — the model backs Chelsea clearly at home."),
            ("Crystal Palace", "Newcastle", None,
             "Palace edge it at Selhurst; Newcastle's draw threat is real."),
            ("Hull", "Brentford", "COIN FLIP",
             "Three outcomes within four points — Elo actually leans Brentford."),
            ("Liverpool", "Brighton", None,
             "Liverpool favored at Anfield, but Brighton's form makes 40% feel fragile."),
            ("Man United", "Bournemouth", None,
             "United should have enough at home; Bournemouth's threat is live."),
            ("Sunderland", "Leeds", None,
             "No home comfort for bottom club Sunderland — the model leans Leeds."),
        ],
    },
    9: {
        "dates": "Oct 31–Nov 2, 2026",
        "fixtures": [
            ("Chelsea", "Man United", "BIG ONE",
             "Saturday's opener at the Bridge — Chelsea backed, United live at 32%."),
            ("Bournemouth", "Leeds", None,
             "Bournemouth edge it at home; Leeds' draw threat sits at 28%."),
            ("Brentford", "Nott'm Forest", None,
             "Brentford clear-ish at home — Forest get 22% and little love."),
            ("Coventry", "Sunderland", "COIN FLIP",
             "The week's purest coin flip: 35 home, 31 draw, 34 away."),
            ("Hull", "Ipswich", None,
             "Hull's home lean is the week's second strongest at 55%."),
            ("Man City", "Brighton", "BANKER",
             "The week's strongest lean — a modest banker by recent standards."),
            ("Tottenham", "Crystal Palace", None,
             "Spurs favored in the London derby, Palace far from out of it."),
            ("Aston Villa", "Fulham", None,
             "Villa at home after beating City — the model stays with them."),
            ("Liverpool", "Arsenal", "BIG ONE",
             "The headline: Arsenal are the week's only away pick, at Anfield."),
            ("Newcastle", "Everton", None,
             "Monday night at St James' — Newcastle, without much conviction."),
        ],
    },
}

TAG_COLORS = {"BANKER": GREEN_TXT, "COIN FLIP": ORANGE, "BIG ONE": GOLD_DIM}


def render(week, out_path):
    meta = META[week]
    df = pd.read_csv(OUT_PATH / f"matchweek{week}_predictions.csv")
    assert len(df) == 10, f"expected 10 fixtures, got {len(df)}"

    # --- measure cards first ---
    tmp = Image.new("RGB", (W, 10))
    d0 = ImageDraw.Draw(tmp)
    cards = []
    for (row, (home, away, tag, comment)) in zip(df.itertuples(),
                                                 meta["fixtures"]):
        assert row.home_team == home and row.away_team == away, \
            f"fixture mismatch: {row.home_team} v {row.away_team}"
        probs = [row.ens_h, row.ens_d, row.ens_a]
        top = max(range(3), key=lambda i: probs[i])
        pick_team = [home, None, away][top]
        pick_pct = round(probs[top] * 100)
        pick_text = ("★ " if tag == "BANKER" else "") + \
            f"{pick_team} win · {pick_pct}%"
        lines = wrap(d0, comment, F_COMMENT, W - PAD * 2 - 30 - 24)
        cards.append(dict(home=home, away=away, tag=tag, comment_lines=lines,
                          probs=probs, pick_text=pick_text,
                          pick_pct=pick_pct,
                          accent=accent_for(pick_pct)))

    # --- layout ---
    y = 64
    header_h = 64 + 40 + 130 + 14 + 56 + 12 + 44 + 40  # eyebrow..sub2 + gaps
    card_h = []
    for c in cards:
        h = 26 + 54 + 16 + 22 + 12 + 38 + 10 + len(c["comment_lines"]) * 36 + 26
        card_h.append(h)
    gap = 22
    H = header_h + sum(card_h) + gap * (len(cards) - 1) + 60 + 90
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # --- header ---
    tw = tracked_text(d, (0, y), "HAND-BUILT MODEL", F_EYEBROW, GOLD, tracking=10)
    # center the tracked text: redraw centered
    img.paste(BG, (0, y - 10, W, y + 60))
    d = ImageDraw.Draw(img)
    x0 = (W - tw) // 2
    tracked_text(d, (x0, y), "HAND-BUILT MODEL", F_EYEBROW, GOLD, tracking=10)
    y += 52
    t = f"MATCHWEEK {week}"
    d.text(((W - text_w(d, t, F_TITLE)) // 2, y), t, font=F_TITLE, fill=WHITE)
    y += 128
    d.rectangle([W // 2 - 110, y, W // 2 + 110, y + 6], fill=GOLD)
    y += 22
    t = f"Premier League  ·  {meta['dates']}"
    d.text(((W - text_w(d, t, F_SUB)) // 2, y), t, font=F_SUB, fill=WHITE)
    y += 58
    t = "Every pick with its home / draw / away split"
    d.text(((W - text_w(d, t, F_SUB2)) // 2, y), t, font=F_SUB2, fill=GRAY)
    y += 40 + 28

    # --- cards ---
    for c, ch in zip(cards, card_h):
        x0c, x1c = PAD, W - PAD
        d.rounded_rectangle([x0c, y, x1c, y + ch], radius=26,
                            fill=CARD_BG, outline=CARD_BORDER, width=2)
        # accent bar
        d.rounded_rectangle([x0c + 14, y + 22, x0c + 24, y + ch - 22],
                            radius=5, fill=c["accent"])
        cx = x0c + 44
        cy = y + 26

        # right block: tag + pick pill (measure first)
        tag_w = 0
        if c["tag"]:
            tag_w = text_w(d, c["tag"], F_TAG) + 36 + 14
        pill_w = text_w(d, c["pick_text"], F_PILL) + 36
        right_w = tag_w + pill_w
        title = f"{c['home']}  vs  {c['away']}"
        fnt_size = 38
        fnt = F_CARD_TITLE
        while (fnt_size > 26 and text_w(d, title, fnt)
               > (x1c - 24) - cx - right_w - 20):
            fnt_size -= 2
            fnt = font("DejaVuSans-Bold.ttf", fnt_size)
        d.text((cx, cy), title, font=fnt, fill=WHITE)

        rx = x1c - 24 - pill_w
        if c["tag"]:
            pill(d, (rx - tag_w, cy + 2), c["tag"], F_TAG, None,
                 TAG_COLORS[c["tag"]], outline=TAG_COLORS[c["tag"]],
                 pad_x=16, pad_y=6)
        pill(d, (rx, cy), c["pick_text"], F_PILL, GOLD, DARK_TXT)
        cy += 56

        # probability bar
        bx0, bx1 = cx, x1c - 24
        bw = bx1 - bx0
        d.rounded_rectangle([bx0, cy, bx1, cy + 22], radius=11, fill=TRACK)
        segs = [(c["probs"][0], GREEN_BAR), (c["probs"][1], DRAW_BAR),
                (c["probs"][2], RED_BAR)]
        sx = bx0
        widths = [p * bw for p, _ in segs]
        for (p, col), swd in zip(segs, widths):
            if swd > 0:
                d.rectangle([sx, cy, sx + swd, cy + 22], fill=col)
            sx += swd
        d.ellipse([bx0, cy, bx0 + 22, cy + 22], fill=GREEN_BAR)
        d.ellipse([bx1 - 22, cy, bx1, cy + 22], fill=RED_BAR)
        cy += 22 + 12

        # H/D/A labels
        labels = [(f"HOME {round(c['probs'][0]*100)}%", GREEN_TXT, bx0, "l"),
                  (f"DRAW {round(c['probs'][1]*100)}%", DRAW_TXT,
                   (bx0 + bx1) // 2, "c"),
                  (f"AWAY {round(c['probs'][2]*100)}%", RED_TXT, bx1, "r")]
        for txt, col, lx, align in labels:
            twd = text_w(d, txt, F_LABEL)
            x = lx if align == "l" else (lx - twd // 2 if align == "c"
                                         else lx - twd)
            d.text((x, cy), txt, font=F_LABEL, fill=col)
        cy += 38 + 10

        for line in c["comment_lines"]:
            d.text((cx, cy), line, font=F_COMMENT, fill=GRAY)
            cy += 36
        y += ch + gap

    # --- footer ---
    y += 18
    d.line([W // 2 - 300, y, W // 2 + 300, y], fill=CARD_BORDER, width=2)
    y += 24
    t = "neboiwenofu.com  ·  predictions from a hand-built model"
    d.text(((W - text_w(d, t, F_FOOTER)) // 2, y), t,
           font=F_FOOTER, fill=GRAY_DIM)

    img.save(out_path)
    print(f"wrote {out_path} ({W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int, required=True)
    args = ap.parse_args()
    render(args.week, OUT_PATH / f"mw{args.week}_preview.png")


if __name__ == "__main__":
    main()
