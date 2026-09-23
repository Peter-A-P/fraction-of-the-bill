"""The Pareto chart: field accuracy against cost per 1,000 extractions, as one SVG file.

Written by `smallprint report`, from the same rows as the tables, so the picture cannot
disagree with the numbers beside it. Plain SVG with no plotting library, because the
chart is one kind of mark on two axes and a dependency for it would be the largest thing
in the package; and a file rather than an image, so it renders on GitHub in either theme
and every point carries its exact figures as a tooltip.

**What is drawn.** One dot per run, its 95% interval on accuracy as a whisker. Cost runs
along a log axis, because the frontier spans 35 times from cheapest to dearest and the
self-hosted points sit an order of magnitude or two below both. Frontier runs are circles
in the first categorical slot, self-hosted fine-tunes are diamonds in the second: the shape
carries identity as well as the colour, for a reader who cannot tell the two apart or has
printed the page. The runs no other run beats on both axes, the Pareto front, are joined by
a step line and are the only ones labelled; labelling every point would bury the ones that
matter.

**Self-hosted cost is a choice made visible.** It is the rent divided by the throughput
measured at one concurrency, at one utilisation, and the caption says which. A reader who
runs their card harder moves those points left, and the break-even table says by how much.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from html import escape
from typing import Final

from pydantic import BaseModel, ConfigDict

WIDTH: Final = 760
HEIGHT: Final = 460
LEFT, RIGHT, TOP, BOTTOM = 72, 24, 56, 64

#: The reference palette's first two categorical slots and its chrome, light and dark,
#: validated as a pair against both surfaces (worst all-pairs CVD separation 24.7).
STYLE: Final = """
.c { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.surface { fill: #fcfcfb; }
.grid { stroke: #e1e0d9; stroke-width: 1; }
.axis { stroke: #c3c2b7; stroke-width: 1; }
.ink { fill: #0b0b0b; }
.ink2 { fill: #52514e; }
.muted { fill: #898781; font-variant-numeric: tabular-nums; }
.s1 { fill: #2a78d6; } .s1l { stroke: #2a78d6; }
.s2 { fill: #eb6834; } .s2l { stroke: #eb6834; }
.ring { stroke: #fcfcfb; stroke-width: 2; }
.front { fill: none; stroke: #52514e; stroke-width: 1; stroke-dasharray: 3 3; }
@media (prefers-color-scheme: dark) {
  .surface { fill: #1a1a19; }
  .grid { stroke: #2c2c2a; }
  .axis { stroke: #383835; }
  .ink { fill: #ffffff; }
  .ink2 { fill: #c3c2b7; }
  .s1 { fill: #3987e5; } .s1l { stroke: #3987e5; }
  .s2 { fill: #d95926; } .s2l { stroke: #d95926; }
  .ring { stroke: #1a1a19; }
  .front { stroke: #c3c2b7; }
}
"""


class Point(BaseModel):
    """One run on the chart."""

    model_config = ConfigDict(frozen=True)

    label: str
    self_hosted: bool
    usd_per_1000: float
    accuracy: float
    low: float
    high: float


def pareto_front(points: Sequence[Point]) -> list[Point]:
    """The runs no other run beats on both axes: cheaper or equal, and at least as accurate,
    with one of the two strictly better. Cheapest first."""
    front = []
    for p in points:
        dominated = any(
            q.usd_per_1000 <= p.usd_per_1000
            and q.accuracy >= p.accuracy
            and (q.usd_per_1000 < p.usd_per_1000 or q.accuracy > p.accuracy)
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p.usd_per_1000)


def _decades(low: float, high: float) -> tuple[int, int]:
    return math.floor(math.log10(low)), math.ceil(math.log10(high))


def _money(value: float) -> str:
    return f"US${value:,.0f}" if value >= 1 else f"US${value:g}"


def render(points: Sequence[Point], *, caption: str) -> str:
    """The chart as an SVG document."""
    if not points:
        raise ValueError("nothing to chart")
    if any(p.usd_per_1000 <= 0 or math.isnan(p.usd_per_1000) for p in points):
        raise ValueError("every point needs a positive cost to sit on a log axis")
    first, last = _decades(min(p.usd_per_1000 for p in points), max(p.usd_per_1000 for p in points))
    last = max(last, first + 1)
    y_low = math.floor(min(p.low for p in points) * 100) / 100
    y_high = math.ceil(max(p.high for p in points) * 100) / 100
    if y_high - y_low < 0.02:
        y_high = y_low + 0.02
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM

    def x(cost: float) -> float:
        return LEFT + (math.log10(cost) - first) / (last - first) * plot_w

    def y(acc: float) -> float:
        return TOP + (y_high - acc) / (y_high - y_low) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" class="c" role="img" '
        'aria-labelledby="title desc">',
        '<title id="title">Field accuracy against cost per 1,000 extractions</title>',
        f'<desc id="desc">{escape(caption)}</desc>',
        f"<style>{STYLE}</style>",
        f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}" rx="8"/>',
    ]
    # Gridlines and ticks: every decade of cost, every point of accuracy.
    for decade in range(first, last + 1):
        gx = x(10.0**decade)
        out.append(
            f'<line class="grid" x1="{gx:.1f}" y1="{TOP}" x2="{gx:.1f}" y2="{TOP + plot_h}"/>'
        )
        out.append(
            f'<text class="muted" x="{gx:.1f}" y="{TOP + plot_h + 18}" font-size="12" '
            f'text-anchor="middle">{_money(10.0**decade)}</text>'
        )
    steps = round((y_high - y_low) * 100)
    stride = 1 if steps <= 10 else 2
    for i in range(0, steps + 1, stride):
        acc = y_low + i / 100
        gy = y(acc)
        out.append(
            f'<line class="grid" x1="{LEFT}" y1="{gy:.1f}" x2="{LEFT + plot_w}" y2="{gy:.1f}"/>'
        )
        out.append(
            f'<text class="muted" x="{LEFT - 8}" y="{gy + 4:.1f}" font-size="12" '
            f'text-anchor="end">{acc:.0%}</text>'
        )
    out.append(
        f'<line class="axis" x1="{LEFT}" y1="{TOP + plot_h}" x2="{LEFT + plot_w}" '
        f'y2="{TOP + plot_h}"/>'
    )
    out.append(
        f'<text class="ink2" x="{LEFT + plot_w / 2:.1f}" y="{HEIGHT - 20}" font-size="13" '
        'text-anchor="middle">Cost per 1,000 extractions, log scale</text>'
    )
    out.append(
        f'<text class="ink2" font-size="13" text-anchor="middle" '
        f'transform="translate(18 {TOP + plot_h / 2:.1f}) rotate(-90)">'
        "Fields correct, 95% interval</text>"
    )
    # Legend, one row above the plot: shape and colour together.
    out.append(f'<circle class="s1 ring" cx="{LEFT + 6}" cy="24" r="5"/>')
    out.append(f'<text class="ink" x="{LEFT + 18}" y="28" font-size="13">Frontier API</text>')
    lx = LEFT + 130
    out.append(f'<path class="s2 ring" d="{_diamond(lx + 6, 24)}"/>')
    out.append(
        f'<text class="ink" x="{lx + 18}" y="28" font-size="13">Self-hosted fine-tune</text>'
    )

    front = pareto_front(points)
    if len(front) > 1:
        path = [f"M{x(front[0].usd_per_1000):.1f},{y(front[0].accuracy):.1f}"]
        for p in front[1:]:
            path.append(f"H{x(p.usd_per_1000):.1f}V{y(p.accuracy):.1f}")
        out.append(f'<path class="front" d="{"".join(path)}"/>')
    for p in sorted(points, key=lambda p: p.self_hosted):
        px, py = x(p.usd_per_1000), y(p.accuracy)
        series = "s2" if p.self_hosted else "s1"
        tip = (
            f"{p.label}: {p.accuracy:.1%} ({p.low:.1%} to {p.high:.1%}), "
            f"US${p.usd_per_1000:,.3f} per 1,000"
        )
        out.append(f"<g><title>{escape(tip)}</title>")
        out.append(
            f'<line class="{series}l" stroke-width="1.5" x1="{px:.1f}" y1="{y(p.low):.1f}" '
            f'x2="{px:.1f}" y2="{y(p.high):.1f}"/>'
        )
        if p.self_hosted:
            out.append(f'<path class="s2 ring" d="{_diamond(px, py)}"/>')
        else:
            out.append(f'<circle class="s1 ring" cx="{px:.1f}" cy="{py:.1f}" r="5"/>')
        out.append("</g>")
    for p in front:
        px, py = x(p.usd_per_1000), y(p.accuracy)
        anchor = "end" if px > LEFT + plot_w * 0.75 else "start"
        dx = -10 if anchor == "end" else 10
        out.append(
            f'<text class="ink" x="{px + dx:.1f}" y="{py - 8:.1f}" font-size="12" '
            f'text-anchor="{anchor}">{escape(p.label)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _diamond(cx: float, cy: float, r: float = 6.0) -> str:
    return (
        f"M{cx:.1f},{cy - r:.1f}L{cx + r:.1f},{cy:.1f}L{cx:.1f},{cy + r:.1f}L{cx - r:.1f},{cy:.1f}Z"
    )
