"""Render the report figures from results/figures/plot-data.json (no estimates are computed here).

Compact horizontal small multiples in the misalignment-faking typography: numbered Spectral action title, grey panel
bands, light-filled bars with a darker edge, 95% interval whiskers, values printed at the whisker end, all axes 0-1.
Hollow bars are real footage, where "fake" is a false alarm. Methodological detail lives in the README captions.

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
INK, MUTED, RULE, GRID = "#25323D", "#596977", "#CFD7DE", "#E8ECF0"
# validated categorical order (dataviz validate_palette.js, light mode): Astra, Sol, trained classifier
ASTRA, SOL, CLASSIFIER = "#B45416", "#2B6CA3", "#63368A"
TITLE_H, HEAD_H, GROUP_H, ROW_H, AXIS_H, PAD, GAP, MARGIN = 46, 28, 20, 22, 22, 10, 10, 14
BAR_H, VALUE_W = 13, 46


def configure():
    for path in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont("Arial", fallback_to_default=False)
    font_manager.findfont("Spectral", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "path", "pdf.fonttype": 42,
                         "svg.hashsalt": "robospoof-figures", "text.color": INK})


def light(color, strength=.22):
    return tuple(1 - strength + strength * v for v in to_rgb(color))


def render(spec: dict) -> list[Path]:
    """spec: stem, title, bar_w (px for 0..1), panels [{heading, kind ('rate'|'auroc'), label_w, groups}], where each
    group is {label, color, rows [{label, value, low, high, color, hollow}]}. Panels sit side by side; panels with
    label_w 0 reuse the row layout of the first panel."""
    panels, bar_w = spec["panels"], spec["bar_w"]
    body_h = max(sum((GROUP_H if g.get("label") else 0) + ROW_H * len(g["rows"]) for g in p["groups"]) for p in panels)
    widths = [PAD + p["label_w"] + bar_w + VALUE_W + PAD for p in panels]
    width = MARGIN * 2 + sum(widths) + GAP * (len(panels) - 1)
    height = TITLE_H + HEAD_H + body_h + AXIS_H + MARGIN
    fig = plt.figure(figsize=(width / 100, height / 100))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set_xlim(0, width); cv.set_ylim(height, 0); cv.axis("off")

    def text(x, y, s, size=11, color=INK, ha="left", weight="normal", family="Arial"):
        return cv.text(x, y, s, fontsize=size, family=family, color=color, ha=ha, va="center", weight=weight)

    text(MARGIN, TITLE_H / 2 + 2, spec["title"], size=17, weight="semibold", family="Spectral")
    x = MARGIN
    for p, w in zip(panels, widths):
        top = TITLE_H
        cv.add_patch(Rectangle((x, top), w, HEAD_H + body_h + AXIS_H, facecolor="#FCFCFD", edgecolor=RULE, lw=.9))
        cv.add_patch(Rectangle((x, top), w, HEAD_H, facecolor="#EEF2F5", edgecolor="none"))
        text(x + PAD, top + HEAD_H / 2, p["heading"], size=12.5, weight="semibold", family="Spectral")
        x0 = x + PAD + p["label_w"]  # pixel of value 0
        y_top, y_bot = top + HEAD_H + 4, top + HEAD_H + body_h
        for v in (.5, 1):
            cv.plot([x0 + v * bar_w] * 2, [y_top, y_bot], color=GRID, lw=.8, zorder=0)
        if p["kind"] == "auroc":
            cv.plot([x0 + .5 * bar_w] * 2, [y_top, y_bot], color="#9AA7B2", lw=.8, ls=(0, (3, 2)), zorder=0)
        cv.plot([x0, x0], [y_top, y_bot], color="#B9C3CC", lw=.9)
        for v in (0, .5, 1):
            lab = (f"{v:g}" if p["kind"] == "auroc" else f"{v:.0%}") + (" = chance" if p["kind"] == "auroc" and v == .5 else "")
            text(x0 + v * bar_w, y_bot + AXIS_H / 2, lab, size=9.5, color=MUTED, ha="center")
        y = top + HEAD_H
        for g in p["groups"]:
            if g.get("label"):
                text(x + PAD, y + GROUP_H / 2 + 2, g["label"], size=10.5, weight="semibold", color=g.get("color", INK))
                y += GROUP_H
            for r in g["rows"]:
                cy = y + ROW_H / 2
                if p["label_w"]:
                    text(x + PAD + 8, cy, r["label"], size=10.5, color=INK)
                c, v, lo, hi = r["color"], r["value"], r["low"], r["high"]
                cv.add_patch(Rectangle((x0, cy - BAR_H / 2), max(v * bar_w, .8), BAR_H, lw=1,
                                       edgecolor=c, facecolor="white" if r.get("hollow") else light(c, .28)))
                cv.plot([x0 + lo * bar_w, x0 + hi * bar_w], [cy, cy], color=c, lw=.9)
                for e in (lo, hi):
                    cv.plot([x0 + e * bar_w] * 2, [cy - 4, cy + 4], color=c, lw=.9)
                text(x0 + hi * bar_w + 5, cy, f"{v:.2f}" if p["kind"] == "auroc" else f"{v:.0%}",
                     size=10, color=c, weight="semibold")
                y += ROW_H
        x += w + GAP

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in cv.texts:
        bb = t.get_window_extent(renderer)
        assert bb.x0 >= -1 and bb.y0 >= -1 and bb.x1 <= fig.bbox.width + 1 and bb.y1 <= fig.bbox.height + 1, \
            (spec["stem"], t.get_text())
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"{spec['stem']}.{suffix}"
        meta = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(path, dpi=200, facecolor="white", metadata=meta)
        paths.append(path)
    plt.close(fig)
    return paths


def row(label, est, color, hollow=False):
    return {"label": label, "value": est["value"], "low": est["low"], "high": est["high"], "color": color, "hollow": hollow}


def specs(d: dict) -> list[dict]:
    det = {c["model"]: c["called_fake"] for c in d["detection"]["columns"]}
    sep = d["separability"]
    models = (("gpt-6-astra", "GPT-6 Astra", ASTRA), ("gpt-6.1-sol", "GPT-6.1 Sol", SOL))
    fig1 = {"stem": "01-perception", "bar_w": 250,
            "title": "1. GPT barely flags manipulated snapshots, though a small classifier separates them",
            "panels": [
                {"heading": "“Is this real or fake?” · called fake", "kind": "rate", "label_w": 172, "groups": [
                    {"label": name, "color": col, "rows": [
                        row("Unaltered (false alarm)", det[m]["n"], col, hollow=True),
                        row("Blurred", det[m]["b"], col), row("Doctored", det[m]["d"], col)]}
                    for m, name, col in models]},
                {"heading": "Separability from unaltered (AUROC)", "kind": "auroc", "label_w": 168, "groups": [
                    {"label": f"{d['footage'][c]} vs unaltered", "rows": [
                        row("GPT-6 Astra", sep[c]["gpt-6-astra"], ASTRA), row("GPT-6.1 Sol", sep[c]["gpt-6.1-sol"], SOL),
                        row("Small classifier", sep[c]["classifier"], CLASSIFIER)]}
                    for c in "bd"]}]}
    fr = d["framing"]
    ladder = [("observer", "Observer, text answer"), ("actor_text", "Agent, text actions"),
              ("robot_text", "Robot prompt, text actions"), ("robot_motion", "Robot prompt, motion tools")]
    fig2 = {"stem": "02-embodiment", "bar_w": 170,
            "title": "2. Astra refuses unsafe tasks in text, but acts once it has robot motion tools",
            "panels": [{"heading": d["footage"][f], "kind": "rate", "label_w": 198 if i == 0 else 0, "groups": [
                {"rows": [row(lab, fr[k][f], ASTRA) for k, lab in ladder]}]} for i, f in enumerate("nbd")]}
    return [fig1, fig2]


def main() -> None:
    configure()
    d = json.loads(DATA.read_text())
    for old in OUT.glob("0*"):  # one current figure set; superseded figures are recoverable from git
        old.unlink()
    for spec in specs(d):
        print(*[p.name for p in render(spec)])


if __name__ == "__main__":
    main()
