"""Build every README figure.

    uv run --extra models --extra viz python scripts/make_figures.py

Inputs: ``runs/detect/results.json`` and ``runs/faults/results.json`` (from
``groundsight bench``) for the charts; the banner, the animation and the report
screenshots are computed here from the built-in demo. Outputs go to ``docs/assets/``
as ``<name>-light.*`` and ``<name>-dark.*``.

Colours come from a fixed palette checked for colour-vision deficiencies. Verdicts use
the status colours, always with an icon and a label: supported ✓, unsupported ?,
contradicted ✗.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

ASSETS = Path("docs/assets")
CSS_PX = 96.0
DPI = 200
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    empty: str
    neutral: str
    series: tuple[str, ...]


THEMES = {
    "light": Theme(
        name="light",
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        empty="#ecebe6",
        neutral="#a8a79f",
        series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    ),
    "dark": Theme(
        name="dark",
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        empty="#262624",
        neutral="#64635e",
        series=("#3987e5", "#d95926", "#199e70", "#c98500"),
    ),
}


def use_theme(mode: str) -> Theme:
    theme = THEMES[mode]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink2,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink2,
            "ytick.labelcolor": theme.ink2,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    return theme


def _title(fig: Figure, title: str, subtitle: str, theme: Theme) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, weight="bold")
    fig.text(0.012, 0.905, subtitle, ha="left", va="top", fontsize=9, color=theme.ink2)


def _px_per_unit(ax: Axes) -> tuple[float, float]:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bbox = ax.get_window_extent()
    return bbox.width / abs(x1 - x0), bbox.height / abs(y1 - y0)


def _rounded_bar(
    ax: Axes, start: float, thickness: float, length: float, color: str, *, horizontal: bool
) -> None:
    """Bar with a 4px rounded data end and a square baseline (axis limits must be final)."""
    if length <= 0:
        return
    ppx, ppy = _px_per_unit(ax)
    radius_px = 4.0 * ax.figure.dpi / CSS_PX
    aspect = ppx / ppy
    if horizontal:
        rounding = min(radius_px / ppx, length / 2, thickness / 2 / aspect)
        xy, width, height = (0.0, start), length, thickness
        cover = Rectangle((0.0, start), max(length - rounding, 0.0), thickness)
    else:
        rounding = min(radius_px / ppx, thickness / 2, length / 2 / aspect)
        xy, width, height = (start, 0.0), thickness, length
        cover = Rectangle((start, 0.0), thickness, max(length - rounding * aspect, 0.0))
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            mutation_aspect=aspect,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
        )
    )
    cover.set_facecolor(color)
    cover.set_linewidth(0)
    ax.add_patch(cover)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)
    return path


# ------------------------------------------------------------------------ banner


GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"
VERDICT = {"supported": (GOOD, "✓"), "unsupported": (WARN, "?"), "contradicted": (BAD, "✗")}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


# ------------------------------------------------------------------------ banner


def banner(mode: str) -> str:
    t = THEMES[mode]
    width, height = 1200, 260
    claims = [("supported", 180), ("contradicted", 140), ("supported", 110), ("unsupported", 140)]
    x0, y_claims, y_sources = 590, 70, 176
    shapes, x = [], x0
    sources = [(x0 + i * 112, y_sources) for i in range(5)]
    for i, (verdict, length) in enumerate(claims):
        color, icon = VERDICT[verdict]
        shapes.append(
            f'<rect x="{x}" y="{y_claims}" width="{length - 12}" height="34" rx="9" '
            f'fill="{color}" fill-opacity="0.16" stroke="{color}" stroke-width="2"/>'
        )
        shapes.append(
            f'<text x="{x + 12}" y="{y_claims + 23}" font-family="{FONT}" font-size="17" '
            f'font-weight="700" fill="{t.ink}">{icon}</text>'
        )
        shapes.append(
            f'<rect x="{x + 34}" y="{y_claims + 12}" width="{max(20, length - 70)}" '
            f'height="9" rx="4" fill="{t.ink2}" fill-opacity="0.35"/>'
        )
        if verdict != "unsupported":
            sx, sy = sources[[0, 2, 3, 4][i]]
            dash = ' stroke-dasharray="6 5"' if verdict == "contradicted" else ""
            shapes.append(
                f'<path d="M {x + (length - 12) / 2} {y_claims + 36} C {x + (length - 12) / 2} '
                f'{y_claims + 80}, {sx + 46} {sy - 46}, {sx + 46} {sy - 4}" fill="none" '
                f'stroke="{color}" stroke-width="2"{dash}/>'
            )
        x += length
    for sx, sy in sources:
        shapes.append(
            f'<rect x="{sx}" y="{sy}" width="92" height="52" rx="8" fill="{t.surface}" '
            f'stroke="{t.axis}" stroke-width="1.5"/>'
        )
        for k in range(3):
            shapes.append(
                f'<rect x="{sx + 10}" y="{sy + 11 + k * 11}" width="{72 - k * 14}" height="5" '
                f'rx="2.5" fill="{t.ink2}" fill-opacity="0.3"/>'
            )
    shapes.append(
        f'<text x="{x0}" y="{y_claims - 14}" font-family="{FONT}" font-size="14" '
        f'fill="{t.ink2}">answer, claim by claim</text>'
    )
    shapes.append(
        f'<text x="{x0}" y="{y_sources + 74}" font-family="{FONT}" font-size="14" '
        f'fill="{t.ink2}">retrieved sources</text>'
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Groundsight: see which claims in a RAG answer your sources don't support, and why">
  <rect width="{width}" height="{height}" rx="18" fill="{t.surface}"/>
  <text x="56" y="118" font-family="{FONT}" font-size="64" font-weight="700" fill="{t.ink}">Groundsight</text>
  <text x="58" y="162" font-family="{FONT}" font-size="22" fill="{t.ink2}">Which claims in a RAG answer do your</text>
  <text x="58" y="192" font-family="{FONT}" font-size="22" fill="{t.ink2}">sources not support, and why?</text>
  {"".join(shapes)}
</svg>
"""


# ----------------------------------------------------------------------- charts

LABELS = {
    "cosine": "Embedding similarity",
    "lexical": "Word overlap",
    "nli": "NLI (DeBERTa-v3 base)",
    "nli-large": "NLI (DeBERTa-v3 large)",
    "minicheck": "MiniCheck (RoBERTa-large)",
    "hhem": "HHEM-2.1-open",
}
BASELINE_CHECKERS = {"cosine", "lexical"}


def _value_column(ax: Axes, y: float, text: str, theme: Theme, bold: bool) -> None:
    ax.text(
        1.02,
        y,
        text,
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=9.5,
        weight="bold" if bold else "normal",
        color=theme.ink if bold else theme.ink2,
        clip_on=False,
    )


def plot_detection(
    results: list[dict[str, Any]], sentences: int, answers: int, out: Path, mode: str
) -> Path:
    theme = use_theme(mode)
    rows = [r for r in results if r["checker"] in LABELS]
    rows.sort(key=lambda r: r["auroc"])
    fig = plt.figure(figsize=(7.6, 0.46 * len(rows) + 1.6))
    _title(
        fig,
        "Does the source support this sentence?",
        f"RAGTruth test set, {sentences:,} answer sentences from {answers:,} LLM answers. Area "
        "under the ROC curve for\nflagging the sentences that annotators marked as "
        "hallucinated (50% = chance).",
        theme,
    )
    top = 1 - 0.95 / fig.get_figheight()
    ax = fig.add_axes((0.31, 0.12, 0.57, top - 0.12))
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)), [LABELS[r["checker"]] for r in rows])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["50%", "60%", "70%", "80%", "90%", "100%"])
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.spines["bottom"].set_visible(False)
    for y, r in enumerate(rows):
        ours = r["checker"] not in BASELINE_CHECKERS
        color = theme.series[0] if ours else theme.neutral
        ax.plot([0.5, r["auroc"]], [y, y], color=color, lw=2, solid_capstyle="butt")
        ax.scatter(
            [r["auroc"]], [y], s=90, color=color, zorder=3, edgecolors=theme.surface, linewidths=2
        )
        _value_column(ax, y, f"{r['auroc']:.1%}", theme, ours)
    return _save(fig, out)


def plot_conformal(rows: list[dict[str, Any]], checker: str, out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    fig = plt.figure(figsize=(7.6, 3.7))
    _title(
        fig,
        "Pick the recall you need; see what it costs",
        f"{LABELS.get(checker, checker)}, threshold calibrated on RAGTruth train, measured on "
        "test: share of hallucinated\nsentences caught, and share of supported sentences "
        "flagged anyway.",
        theme,
    )
    ax = fig.add_axes((0.09, 0.12, 0.88, 0.58))
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(len(rows)), [f"target recall {r['target']:.0%}" for r in rows])
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    fig.canvas.draw()
    width, gap = 0.3, 0.02
    for i, r in enumerate(rows):
        for k, (key, slot) in enumerate((("recall", 0), ("false_alarms", 1))):
            x = i - width - gap / 2 if k == 0 else i + gap / 2
            _rounded_bar(ax, x, width, r[key], theme.series[slot], horizontal=False)
            ax.text(
                x + width / 2,
                r[key] + 0.025,
                f"{r[key]:.0%}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                color=theme.ink,
            )
        ax.plot([i - width - gap, i], [r["target"]] * 2, color=theme.ink, lw=1.2, ls=(0, (3, 2)))
    handles = [Rectangle((0, 0), 1, 1, facecolor=theme.series[s], edgecolor="none") for s in (0, 1)]
    handles.append(plt.Line2D([0], [0], color=theme.ink, lw=1.2, ls=(0, (3, 2))))
    fig.legend(
        handles,
        ["Hallucinations caught", "False alarms", "Target"],
        loc="upper left",
        bbox_to_anchor=(0.008, 0.8),
        ncol=3,
        handlelength=1.2,
        columnspacing=1.6,
    )
    return _save(fig, out)


FAULT_LABELS = {
    "retrieval_miss": "Gold passages removed",
    "misleading_source": "Answer edited in the source",
    "conflicting_sources": "Edited copy added",
    "buried_evidence": "Gold buried mid-context",
}


def plot_triage(by_fault: dict[str, Any], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    faults = [f for f in FAULT_LABELS if f in by_fault]
    fig = plt.figure(figsize=(7.6, 0.75 * len(faults) + 1.7))
    _title(
        fig,
        "Why did the answer go wrong?",
        "HotpotQA answers broken by one injected fault (local 1.5B model). Share where "
        "triage names the injected\ncause, and where replaying with the matching fix beats "
        "resampling (95% interval above zero).",
        theme,
    )
    top = 1 - 1.0 / fig.get_figheight()
    ax = fig.add_axes((0.3, 0.13, 0.58, top - 0.13))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(faults) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(
        range(len(faults)), [f"{FAULT_LABELS[f]}\n(n = {by_fault[f]['n']})" for f in faults]
    )
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.spines["bottom"].set_visible(False)
    fig.canvas.draw()
    h = 0.34
    for y, f in enumerate(faults):
        r = by_fault[f]
        values = [(r["triage_accuracy"], 0, -h / 2 - 0.01), (r["verified"] or 0.0, 1, h / 2 + 0.01)]
        for value, slot, offset in values:
            _rounded_bar(ax, y + offset - h / 2, h, value, theme.series[slot], horizontal=True)
            ax.text(
                min(value + 0.015, 0.97),
                y + offset,
                f"{value:.0%}",
                va="center",
                fontsize=9,
                color=theme.ink,
            )
    handles = [Rectangle((0, 0), 1, 1, facecolor=theme.series[s], edgecolor="none") for s in (0, 1)]
    fig.legend(
        handles,
        ["Triage names the cause", "Replay confirms the cause"],
        loc="upper left",
        bbox_to_anchor=(0.008, top + 0.075),
        ncol=2,
        handlelength=1.0,
        handleheight=1.0,
    )
    return _save(fig, out)


ATTRIBUTION_ORDER = [
    "random",
    "word overlap",
    "embedding cosine",
    "likelihood (0.5B proxy)",
    "likelihood (answering model)",
]
ATTRIBUTION_LABELS = {
    "random": "Random passage",
    "word overlap": "Most word overlap",
    "embedding cosine": "Most similar embedding",
    "likelihood (0.5B proxy)": "Likelihood ablation, 0.5B proxy",
    "likelihood (answering model)": "Likelihood ablation, same model",
}


def plot_attribution(summary: dict[str, Any], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    names = [n for n in ATTRIBUTION_ORDER if n in summary]
    n = summary[names[-1]]["n"] if names else 0
    fig = plt.figure(figsize=(7.6, 0.44 * len(names) + 1.6))
    _title(
        fig,
        "Which passage made the model say it?",
        f"{n} wrong answers caused by an edited passage. Share where each method ranks the "
        "edited passage first;\nwhiskers are 95% Wilson intervals.",
        theme,
    )
    top = 1 - 0.95 / fig.get_figheight()
    ax = fig.add_axes((0.36, 0.12, 0.52, top - 0.12))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_yticks(range(len(names)), [ATTRIBUTION_LABELS[k] for k in names])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.spines["bottom"].set_visible(False)
    fig.canvas.draw()
    for y, name in enumerate(names):
        r = summary[name]
        ours = name.startswith("likelihood")
        _rounded_bar(
            ax, y - 0.3, 0.6, r["top1"], theme.series[0] if ours else theme.neutral, horizontal=True
        )
        low, high = r["ci"]
        ax.plot([low, high], [y, y], color=theme.ink2, lw=1.2)
        for edge in (low, high):
            ax.plot([edge, edge], [y - 0.12, y + 0.12], color=theme.ink2, lw=1.2)
        _value_column(ax, y, f"{r['top1']:.0%}", theme, ours)
    return _save(fig, out)


# ------------------------------------------------------------------- animation


def animate_audit(result: Any, out: Path, mode: str) -> Path:
    """Claims are checked one by one; each verdict points at its evidence passage."""
    theme = use_theme(mode)
    claims = result.claims
    chunks = result.chunks
    frames: list[tuple[int, bool]] = []
    for i in range(len(claims)):
        frames += [(i, False), (i, True), (i, True), (i, True)]
    frames += [(len(claims), True)] * 8
    fig = plt.figure(figsize=(9.6, 5.0))

    def has_evidence(claim: Any) -> bool:
        return claim.evidence is not None and claim.verdict.value != "unsupported"

    def draw(frame: int) -> None:
        fig.clear()
        current, shown = frames[frame]
        done = current >= len(claims)
        fig.text(
            0.012,
            0.975,
            "Checking a RAG answer, claim by claim",
            ha="left",
            va="top",
            fontsize=13,
            weight="bold",
        )
        fig.text(
            0.012,
            0.915,
            f"Question: {result.question}",
            ha="left",
            va="top",
            fontsize=9.3,
            color=theme.ink2,
        )
        ax = fig.add_axes((0.0, 0.08, 1.0, 0.78))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")
        ax.text(2, 99, "Answer", fontsize=10.5, weight="bold", va="top")
        ax.text(54, 99, "Retrieved passages", fontsize=10.5, weight="bold", va="top")
        visible = [
            i
            for i, c in enumerate(claims)
            if has_evidence(c) and (done or (i == current and shown))
        ]
        lit = {claims[i].evidence.chunk_id for i in visible}
        slot = 90 / max(len(chunks), 1)
        chunk_y = {}
        for j, chunk in enumerate(chunks):
            y = 92 - j * slot
            chunk_y[chunk.id] = y - slot / 2 + 1
            hot = chunk.id in lit
            ax.add_patch(
                FancyBboxPatch(
                    (54, y - slot + 2.2),
                    44,
                    slot - 3.2,
                    boxstyle="round,pad=0,rounding_size=1.2",
                    facecolor=theme.surface,
                    linewidth=2 if hot else 1,
                    edgecolor=theme.series[0] if hot else theme.axis,
                )
            )
            ax.text(
                55.5,
                y - 1.6,
                "\n".join(textwrap.wrap(chunk.text, 58)[:2]),
                fontsize=7.6,
                va="top",
                color=theme.ink if hot else theme.ink2,
                linespacing=1.25,
            )
        row = min(90 / max(len(claims), 1), 22)
        for i, claim in enumerate(claims):
            y = 92 - i * row
            revealed = i < current or (i == current and shown) or done
            color, icon = VERDICT[claim.verdict.value]
            active = i == current and not done
            box = (2, y - row + 2.5)
            ax.add_patch(
                FancyBboxPatch(
                    box,
                    46,
                    row - 3.5,
                    boxstyle="round,pad=0,rounding_size=1.2",
                    facecolor=color if revealed else theme.surface,
                    alpha=0.16 if revealed else 1.0,
                    linewidth=0,
                )
            )
            ax.add_patch(
                FancyBboxPatch(
                    box,
                    46,
                    row - 3.5,
                    boxstyle="round,pad=0,rounding_size=1.2",
                    facecolor="none",
                    linewidth=2 if active or revealed else 1,
                    edgecolor=color if revealed else (theme.ink if active else theme.axis),
                )
            )
            ax.text(
                4,
                y - 2,
                "\n".join(textwrap.wrap(claim.text, 52)[:3]),
                fontsize=8.6,
                va="top",
                color=theme.ink,
                linespacing=1.3,
            )
            if revealed:
                ax.text(
                    4,
                    y - row + 4.6,
                    f"{icon} {claim.verdict.value}",
                    fontsize=8.6,
                    weight="bold",
                    color=theme.ink,
                )
                if claim.diagnosis is not None:
                    cause = claim.diagnosis.cause.value.replace("_", " ")
                    ax.text(17, y - row + 4.6, f"cause: {cause}", fontsize=8.6, color=theme.ink2)
            if i in visible:
                style = "-" if claim.verdict.value == "supported" else (0, (4, 3))
                ax.annotate(
                    "",
                    xy=(54, chunk_y[claim.evidence.chunk_id]),
                    xytext=(48, y - row / 2 + 0.5),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "lw": 1.8,
                        "linestyle": style,
                        "connectionstyle": "arc3,rad=-0.12",
                    },
                )
        if done:
            counts = result.counts()
            fig.text(
                0.012,
                0.035,
                f"{result.groundedness:.0%} supported, {counts['contradicted']} "
                f"contradicted, {counts['unsupported']} unsupported. Every flag comes with its "
                "evidence and a probable cause.",
                fontsize=9.6,
                color=theme.ink,
            )

    anim = FuncAnimation(fig, draw, frames=len(frames))
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=3), dpi=130, savefig_kwargs={"facecolor": theme.surface})
    plt.close(fig)
    print(out)
    return out


# ------------------------------------------------------------------ screenshots


def screenshot(page: str, name: str) -> None:
    if not CHROME.exists():
        print(f"skipping screenshots: {CHROME} not found")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("light", "dark"):
            themed = Path(tmp) / f"{name}-{mode}.html"
            themed.write_text(page.replace("<html", f'<html data-theme="{mode}"', 1))
            target = ASSETS / f"{name}-{mode}.png"
            subprocess.run(
                [
                    str(CHROME),
                    "--headless=new",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=2",
                    "--window-size=1240,900",
                    f"--screenshot={target.resolve()}",
                    themed.as_uri(),
                ],
                check=True,
                capture_output=True,
            )
            print(target)


def main() -> None:
    import groundsight as gs
    from groundsight.cli import DEMO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detect", default="runs/detect/results.json")
    parser.add_argument("--faults", default="runs/faults/results.json")
    parser.add_argument("--checker", default=gs.DEFAULT_CHECKER)
    parser.add_argument("--generator", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--skip-demo", action="store_true")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        (ASSETS / f"banner-{mode}.svg").write_text(banner(mode))
        print(ASSETS / f"banner-{mode}.svg")

    detect = Path(args.detect)
    if detect.exists():
        data = json.loads(detect.read_text())
        results = data["detection"]
        sentences = max(r.get("test_sentences", 0) for r in results)
        rows_file = detect.parent / "rows-minicheck-test.jsonl"
        answers = (
            len(
                {json.loads(line)["example"] for line in rows_file.read_text().splitlines() if line}
            )
            if rows_file.exists()
            else 0
        )
        default = next((r for r in results if r["checker"] == "minicheck"), results[0])
        for mode in THEMES:
            plot_detection(results, sentences, answers, ASSETS / f"detection-{mode}.png", mode)
            plot_conformal(
                default["conformal"], default["checker"], ASSETS / f"recall-{mode}.png", mode
            )
    faults = Path(args.faults)
    if faults.exists():
        summary = json.loads(faults.read_text())["summary"]
        for mode in THEMES:
            plot_triage(summary["by_fault"], ASSETS / f"triage-{mode}.png", mode)
            plot_attribution(summary["attribution"], ASSETS / f"attribution-{mode}.png", mode)
    if args.skip_demo:
        return
    result = gs.audit(
        DEMO["answer"], DEMO["chunks"], question=DEMO["question"], checker=args.checker
    )
    for mode in THEMES:
        animate_audit(result, ASSETS / f"audit-{mode}.gif", mode)
    chat = gs.LocalChat(args.generator, max_new_tokens=60)
    attribution = gs.attribute(result, gs.LocalScorer(chat))
    checks = gs.verify(result, gs.PromptResponder(chat, temperature=0.7), samples=4)
    screenshot(gs.render(result, attribution=attribution, verifications=checks), "report")


if __name__ == "__main__":
    main()
