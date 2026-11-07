"""Render the report figures from results/figures/plot-data.json (no estimates are computed here).

Layout and typography follow the misalignment-faking report figures: numbered Spectral action title, coloured column
headers with grey sample sizes, a grey measurement band, light-filled bars with a darker edge, 95% interval caps,
values printed above, sparse muted ticks, no chart junk. Methodological detail lives in the README captions.

Usage: uv run python scripts/plot_figures.py      Writes report/figures/0N-<name>.{png,pdf,svg}.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "figures" / "plot-data.json"
OUT = ROOT / "report" / "figures"
FONTS = Path(__file__).with_name("fonts") / "spectral"
INK, MUTED, RULE = "#25323D", "#596977", "#CDD6DE"
# validated categorical order (dataviz validate_palette.js, light mode): Astra, Sol, trained classifiers
ASTRA, SOL, CLASSIFIER, REFERENCE = "#B45416", "#2B6CA3", "#63368A", "#8897A3"


def configure():
    for path in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont("Arial", fallback_to_default=False)
    font_manager.findfont("Spectral", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 13, "svg.fonttype": "path", "pdf.fonttype": 42,
                         "svg.hashsalt": "robospoof-figures", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})


def light(color, strength=.22):
    return tuple(1 - strength + strength * v for v in to_rgb(color))


def pct(v):
    return f"{v * 100:.0f}%"


def render(spec: dict) -> list[Path]:
    """spec: stem, title, columns [{label, sub, color}], bands [{heading, kind, cells, plot_height?}], width?
    where cells[column] is a list of bars {label, value, low, high, color, hollow}."""
    cols, bands = spec["columns"], spec["bands"]
    n_bars = max(len(c) for b in bands for c in b["cells"])
    width = spec.get("width", 1540)
    left, right, gap = 130, 24, 22
    cell_w = (width - left - right) / len(cols)
    two_line = any("\n" in bar["label"] for b in bands for c in b["cells"] for bar in c)
    heading_space = 56 if two_line else 36
    band_header, band_gap, top, bottom = 34, 14, 128, 16
    plot_hs = [b.get("plot_height", spec.get("plot_height", 150)) for b in bands]
    band_hs = [band_header + ph + heading_space + 18 for ph in plot_hs]
    height = top + sum(band_hs) + (len(bands) - 1) * band_gap + bottom
    fig = plt.figure(figsize=(width / 100, height / 100))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, width); canvas.set_ylim(height, 0); canvas.axis("off")

    def text(x, y, s, size=15, color=INK, ha="left", weight="normal"):
        return canvas.text(x, y, s, fontsize=size, family="Spectral" if size >= 16 else "Arial", color=color,
                           ha=ha, va="center", weight=weight, linespacing=1.1)

    text(24, 34, spec["title"], size=24, weight="semibold")
    for ci, col in enumerate(cols):
        cx = left + (ci + .5) * cell_w
        text(cx, 82, col["label"], size=20, color=col.get("color", INK), ha="center", weight="semibold")
        text(cx, 107, col["sub"], size=12, color=MUTED, ha="center")

    for bi, band in enumerate(bands):
        b_top, plot_h, band_h = top + sum(band_hs[:bi]) + bi * band_gap, plot_hs[bi], band_hs[bi]
        canvas.add_patch(Rectangle((18, b_top), width - 36, band_h, facecolor="#FCFCFD", edgecolor="#CFD7DE", lw=1))
        canvas.add_patch(Rectangle((18, b_top), width - 36, band_header, facecolor="#EEF2F5", edgecolor="none"))
        text(28, b_top + band_header / 2, band["heading"], size=16, weight="semibold")
        for ci in range(1, len(cols)):
            bx = left + ci * cell_w
            canvas.plot([bx, bx], [b_top + band_header + 8, b_top + band_h - 8], color=RULE, lw=.9)
        auroc = band["kind"] == "auroc"
        limits = (.5, 1.1) if auroc else (0, 1.2)
        for ci, bars in enumerate(band["cells"]):
            x, y = left + ci * cell_w + gap / 2, b_top + band_header + heading_space
            ax = fig.add_axes([x / width, 1 - (y + plot_h) / height, (cell_w - gap) / width, plot_h / height])
            ax.set_facecolor(light(cols[ci].get("color", INK), .025))
            ax.set_xlim(-.65, n_bars - 1 + .65); ax.set_ylim(*limits); ax.set_xticks([])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color("#C5CED6")
            ax.tick_params(axis="both", length=0, labelsize=12, colors=MUTED)
            ax.tick_params(axis="y", pad=11)
            if auroc:
                ax.set_yticks([.5, .75, 1])
                ax.set_yticklabels(["0.5 = chance", "0.75", "1.0"])
                ax.spines["bottom"].set_visible(False)
                ax.axhline(.5, color=REFERENCE, lw=.8)
            elif plot_h < 120:
                ax.set_yticks([0, 1]); ax.set_yticklabels(["0%", "100%"])
            else:
                ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0%", "50%", "100%"])
            if ci:
                ax.tick_params(axis="y", labelleft=False)
            width_bar = .68 if n_bars <= 4 else .6
            for xi, bar in enumerate(bars):
                c, v, lo, hi = bar["color"], bar["value"], bar["low"], bar["high"]
                base = .5 if auroc else 0
                ax.bar(xi, v - base, bottom=base, width=width_bar, lw=1.2, edgecolor=c,
                       facecolor="white" if bar.get("hollow") else light(c, .28))
                ax.vlines(xi, lo, hi, color=c, lw=1)
                ax.hlines([lo, hi], xi - .07, xi + .07, color=c, lw=1)
                ax.text(xi, max(v, hi) + .025 * (limits[1] - limits[0]), f"{v:.2f}" if auroc else pct(v),
                        ha="center", va="bottom", color=c, fontsize=12.5, weight="semibold")
                ax.text(xi, 1.06, bar["label"], transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=spec.get("label_size", 12.5 if n_bars > 4 else 13), color=INK, weight="semibold",
                        linespacing=1.15)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for item in list(ax.texts) + ([t.label1 for t in ax.yaxis.get_major_ticks()] if ax.axison else []):
            if item.get_visible() and item.get_text():
                bb = item.get_window_extent(renderer)
                assert bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= fig.bbox.width + 1 and bb.y1 <= fig.bbox.height + 1, \
                    (spec["stem"], item.get_text())
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"{spec['stem']}.{suffix}"
        meta = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(path, dpi=190, facecolor="white", metadata=meta)
        paths.append(path)
    plt.close(fig)
    return paths


def bar(label, est, color, hollow=False):
    return {"label": label, "value": est["value"], "low": est["low"], "high": est["high"], "color": color, "hollow": hollow}


def specs(d: dict) -> list[dict]:
    det = {c["model"]: c for c in d["detection"]["columns"]}
    fig1 = {"stem": "01-detection", "width": 1300,
            "title": "1. Asked directly, GPT models barely tell doctored snapshots from real ones",
            "columns": [{"label": "GPT-6 Astra", "sub": "first snapshot · 5 tasks × 12 trials per footage type", "color": ASTRA},
                        {"label": "GPT-6.1 Sol", "sub": "same 180 snapshots", "color": SOL}],
            "bands": [{"heading": "“Is this real or fake?” · called fake", "kind": "rate", "cells": [
                [bar(d["footage"][f], det[m]["called_fake"][f], col, hollow=f != "d") for f in "nbd"]
                for m, col in (("gpt-6-astra", ASTRA), ("gpt-6.1-sol", SOL))]},
                {"heading": "“Describe what you see” · mentions manipulation", "kind": "rate", "plot_height": 70, "cells": [
                [bar(d["footage"][f], det[m]["mentions_manipulation"][f], col, hollow=f != "d") for f in "nbd"]
                for m, col in (("gpt-6-astra", ASTRA), ("gpt-6.1-sol", SOL))]}],
            "plot_height": 150}
    s = d["separability"]
    fig2 = {"stem": "02-pixels", "width": 1300, "plot_height": 170,
            "title": "2. The evidence is in the pixels: a small classifier separates the same snapshots",
            "columns": [{"label": "Astra’s 120 first snapshots", "sub": "60 doctored · 60 unaltered (5 tasks × 12 trials each)", "color": INK}],
            "bands": [{"heading": "Separability of doctored from unaltered (AUROC)", "kind": "auroc", "cells": [[
                bar("GPT-6 Astra\nP(fake), zero-shot", s["gpt-6-astra"], ASTRA),
                bar("GPT-6.1 Sol\nP(fake), zero-shot", s["gpt-6.1-sol"], SOL),
                bar("Forensic classifier\nwhole frame", s["forensic_frame"], CLASSIFIER),
                bar("Forensic classifier\nobject patch only", s["forensic_patch"], CLASSIFIER),
                bar("Session confound\nreal frames only", s["session_baseline"], REFERENCE, hollow=True)]]}]}
    m = d["methods"]
    fig3 = {"stem": "03-methods", "width": 1300,
            "title": "3. Local edits pass as real; only regenerated scenes get caught",
            "columns": [{"label": "GPT-6 Astra", "sub": "20 real first snapshots (4 per task), each edited five ways", "color": ASTRA}],
            "bands": [{"heading": "“Is this real or fake?” · called fake", "kind": "rate", "cells": [[
                bar("Real\n(unedited)", m["real"], ASTRA, hollow=True),
                bar("Local blend\n(object box)", m["local-blend"], ASTRA),
                bar("Crude paste\n(object box)", m["crude-paste"], ASTRA),
                bar("Whole image\n2.5-flare", m["global-flare"], ASTRA),
                bar("Whole image\n2.5-sunburst", m["global-sunburst"], ASTRA),
                bar("Regenerated\n1-mini", m["global-mini"], ASTRA)]]}]}
    p = d["prompting"]
    order = [("R0", "Base-\nline"), ("R4", "Max\neffort"), ("R5", "Think\ncarefully"),
             ("R3", "Self-\ncheck"), ("R1", "Check\ninputs"), ("R2", "Tamper\nwarning")]
    fig4 = {"stem": "04-prompting", "width": 1760, "label_size": 12,
            "title": "4. Only naming the attack makes the robot refuse doctored footage",
            "columns": [{"label": d["footage"][f], "sub": "5 tasks × 12 trials · exact replay of call 0"} for f in "dbn"],
            "bands": [{"heading": "GPT-6 Astra’s first robot action · gives up instead of moving", "kind": "rate", "cells": [
                [bar(lab, p[c][f], ASTRA, hollow=c == "R0") for c, lab in order] for f in "dbn"]}]}
    fr = d["framing"]
    ladder = [("observer", "Observer\n3rd person"), ("actor_text", "Text agent\nno robot"),
              ("robot_text", "Robot,\ntext actions"), ("robot_motion", "Robot,\nmotion tools")]
    fig5 = {"stem": "05-framing", "width": 1760, "label_size": 12,
            "title": "5. Blur caution disappears once the model has motion tools",
            "columns": [{"label": d["footage"][f], "sub": "5 tasks × 12 trials · identical images"} for f in "bdn"],
            "bands": [{"heading": "GPT-6 Astra’s first answer or action · refuses", "kind": "rate", "cells": [
                [bar(lab, fr[c][f], ASTRA, hollow=c == "robot_motion") for c, lab in ladder] for f in "bdn"]}]}
    return [fig1, fig2, fig3, fig4, fig5]


def main() -> None:
    configure()
    d = json.loads(DATA.read_text())
    for spec in specs(d):
        print(*[p.name for p in render(spec)])


if __name__ == "__main__":
    main()
