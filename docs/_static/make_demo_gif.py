"""Render docs/_static/demo.gif: typewriter-style code + plots.

Run from the repo root:

    python scripts/make_demo_gif.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    HPacker,
    TextArea,
    VPacker,
)
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token
from scipy.signal import chirp

from wdm_transform import TimeSeries

CODE = """\
import numpy as np
from scipy.signal import chirp
from wdm_transform import TimeSeries

Nt = Nf = 2**7
N = Nt * Nf
dt = 0.01
t = np.arange(0, N) * dt
y = chirp(t, f0=5, t1=N*dt, f1=40, method='quadratic')

ts  = TimeSeries(y, dt=dt)
fs  = ts.to_frequency_series()
wdm = ts.to_wdm(nt=Nt)
"""

OUT = "demo.gif"

TRIGGERS = {
    "ts  = TimeSeries(y, dt=dt)\n": "ts",
    "fs  = ts.to_frequency_series()\n": "fs",
    "wdm = ts.to_wdm(nt=Nt)\n": "wdm",
}

CHARS_PER_FRAME = 2
HOLD_FRAMES = 60  # ~5s at 12fps
FPS = 12
FONT_SIZE = 11

# JetBrains-ish palette
STYLE = {
    Token.Keyword: "#0033B3",
    Token.Keyword.Namespace: "#0033B3",
    Token.Name.Builtin: "#0033B3",
    Token.Name.Builtin.Pseudo: "#0033B3",
    Token.Name.Function: "#00627A",
    Token.Name.Class: "#00627A",
    Token.Name.Decorator: "#9E880D",
    Token.Literal.String: "#067D17",
    Token.Literal.String.Doc: "#8C8C8C",
    Token.Literal.Number: "#1750EB",
    Token.Comment: "#8C8C8C",
    Token.Operator: "#000000",
    Token.Operator.Word: "#0033B3",
    Token.Punctuation: "#000000",
}


def token_color(tok_type) -> str:
    t = tok_type
    while t is not None:
        if t in STYLE:
            return STYLE[t]
        t = t.parent
    return "#000000"


def tokenize_chars(code: str) -> list[tuple[str, str]]:
    """Tokenize into a flat (color, char) list preserving order."""
    out: list[tuple[str, str]] = []
    for tok_type, value in lex(code, PythonLexer()):
        color = token_color(tok_type)
        for ch in value:
            out.append((color, ch))
    return out


def group_into_lines(chars: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Group (color, char) into lines of (color, run_text)."""
    lines: list[list[tuple[str, str]]] = [[]]
    for color, ch in chars:
        if ch == "\n":
            lines.append([])
            continue
        line = lines[-1]
        if line and line[-1][0] == color:
            line[-1] = (color, line[-1][1] + ch)
        else:
            line.append((color, ch))
    return lines


def precompute() -> dict:
    Nt = Nf = 2**7
    N = Nt * Nf
    dt = 0.01
    t = np.arange(0, N) * dt
    y = chirp(t, f0=5, t1=N * dt, f1=40, method="quadratic")
    ts = TimeSeries(y, dt=dt)
    fs = ts.to_frequency_series()
    wdm = ts.to_wdm(nt=Nt)
    return {"ts": ts, "fs": fs, "wdm": wdm}


def make_code_box(lines, cursor: bool, ax):
    rows = []
    for i, line in enumerate(lines):
        is_last = i == len(lines) - 1
        children = []
        for color, text in line:
            children.append(
                TextArea(
                    text,
                    textprops=dict(family="monospace", size=FONT_SIZE, color=color),
                )
            )
        if cursor and is_last:
            children.append(
                TextArea(
                    "▎",
                    textprops=dict(family="monospace", size=FONT_SIZE, color="#000000"),
                )
            )
        if not children:
            children.append(
                TextArea(" ", textprops=dict(family="monospace", size=FONT_SIZE))
            )
        rows.append(HPacker(children=children, align="baseline", pad=0, sep=0))
    box = VPacker(children=rows, align="left", pad=0, sep=2)
    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=box,
        frameon=False,
        bbox_to_anchor=(0.01, 0.99),
        bbox_transform=ax.transAxes,
        pad=0,
        borderpad=0,
    )
    ax.add_artist(anchored)
    return anchored


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = precompute()
    all_chars = tokenize_chars(CODE)
    n_chars = len(all_chars)

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(
        2, 3, figure=fig, height_ratios=[1.2, 1.0], hspace=0.45, wspace=0.35
    )
    ax_code = fig.add_subplot(gs[0, :])
    ax_ts = fig.add_subplot(gs[1, 0])
    ax_fs = fig.add_subplot(gs[1, 1])
    ax_wdm = fig.add_subplot(gs[1, 2])

    ax_code.set_axis_off()
    for ax in (ax_ts, ax_fs, ax_wdm):
        ax.set_axis_off()

    drawn = {"ts": False, "fs": False, "wdm": False}
    code_artist = {"box": None}

    def draw_ts() -> None:
        ax_ts.set_axis_on()
        data["ts"].plot(ax=ax_ts)
        ax_ts.set_title("TimeSeries", fontsize=10)
        ax_ts.tick_params(labelsize=8)

    def draw_fs() -> None:
        ax_fs.set_axis_on()
        data["fs"].plot(ax=ax_fs)
        ax_fs.set_xlim(0, 50)
        ax_fs.set_title("FrequencySeries", fontsize=10)
        ax_fs.tick_params(labelsize=8)

    def draw_wdm() -> None:
        ax_wdm.set_axis_on()
        data["wdm"].plot(ax=ax_wdm, show_gridinfo=False)
        ax_wdm.set_title("WDM", fontsize=10)
        ax_wdm.tick_params(labelsize=8)

    drawers = {"ts": draw_ts, "fs": draw_fs, "wdm": draw_wdm}

    n_type_frames = (n_chars + CHARS_PER_FRAME - 1) // CHARS_PER_FRAME
    total_frames = n_type_frames + HOLD_FRAMES

    # Track which prefix length has triggered each plot
    trigger_offsets = {}
    cum = 0
    for trig, key in TRIGGERS.items():
        idx = CODE.index(trig) + len(trig)
        trigger_offsets[key] = idx

    def update(frame: int):
        typing = frame < n_type_frames
        idx = min(n_chars, (frame + 1) * CHARS_PER_FRAME) if typing else n_chars
        prefix = all_chars[:idx]
        lines = group_into_lines(prefix)

        if code_artist["box"] is not None:
            code_artist["box"].remove()
        code_artist["box"] = make_code_box(lines, cursor=typing, ax=ax_code)

        for key, off in trigger_offsets.items():
            if not drawn[key] and idx >= off:
                drawers[key]()
                drawn[key] = True

        return ()

    anim = FuncAnimation(
        fig, update, frames=total_frames, interval=1000 // FPS, blit=False
    )
    writer = PillowWriter(fps=FPS)
    print(f"Writing {OUT} ({total_frames} frames)...")
    anim.save(OUT, writer=writer, dpi=110)
    print("Done.")


if __name__ == "__main__":
    main()
