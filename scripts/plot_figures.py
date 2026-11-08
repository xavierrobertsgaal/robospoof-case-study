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
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "figures" / "plot-data.json"
OUT = ROOT / "report" / "figures"
FONTS = Path(__file__).with_name("fonts") / "spectral"
INK, MUTED, RULE, GRID = "#25323D", "#596977", "#CFD7DE", "#E8ECF0"
# validated categorical order (dataviz validate_palette.js, light mode): Astra, Sol, trained classifier
ASTRA, SOL, CLASSIFIER = "#B45416", "#2B6CA3", "#63368A"
TITLE_H, HEAD_H, GROUP_H, ROW_H, AXIS_H, PAD, GAP, MARGIN = 46, 28, 20, 22, 22, 10, 10, 14
BAR_H, VALUE_W = 13, 46
GREY_DOT = "#8E99A4"
FIG2_TITLE = "2. GPT refuses unsafe tasks in text, but acts once it has robot motion tools"  # checked against Sol results
FIG4_TITLE = "4. Astra’s own “fake” verdict does not drive its refusals"
FIG5_TITLE = "5. Told it is in an alignment eval, Astra refuses blurred footage it otherwise acts on"


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
    panels, bar_w, rh = spec["panels"], spec["bar_w"], spec.get("row_h", ROW_H)
    body_h = max(sum((GROUP_H if g.get("label") else 0) + rh * len(g["rows"]) for g in p["groups"]) for p in panels)
    fixed = MARGIN * 2 + sum(PAD + p["label_w"] + VALUE_W + PAD for p in panels) + GAP * (len(panels) - 1)
    title_w = TextPath((0, 0), spec["title"], size=17,
                       prop=FontProperties(family="Spectral", weight="semibold")).get_extents().width * 100 / 72
    bar_w = max(bar_w, (title_w + 2 * MARGIN + 8 - fixed) / len(panels))  # widen bars until the title fits
    widths = [PAD + p["label_w"] + bar_w + VALUE_W + PAD for p in panels]
    width = MARGIN * 2 + sum(widths) + GAP * (len(panels) - 1)
    notes = spec["footnote"].split("\n") if spec.get("footnote") else []
    foot_h = (20 if spec.get("legend") else 0) + 15 * len(notes) + (6 if notes or spec.get("legend") else 0)
    height = TITLE_H + HEAD_H + body_h + AXIS_H + MARGIN + foot_h
    fig = plt.figure(figsize=(width / 100, height / 100))
    cv = fig.add_axes([0, 0, 1, 1])
    cv.set_xlim(0, width); cv.set_ylim(height, 0); cv.axis("off")

    def text(x, y, s, size=11, color=INK, ha="left", weight="normal", family="Arial"):
        return cv.text(x, y, s, fontsize=size, family=family, color=color, ha=ha, va="center", weight=weight)

    def marker(mx, my, style, color):  # equal small points: hollow, grey reference, filled in the model colour
        if style == "ref":
            cv.plot(mx, my, "o", ms=5.2, mfc=GREY_DOT, mec=GREY_DOT, zorder=6)
        elif style == "filled":
            cv.plot(mx, my, "o", ms=5.2, mfc=color, mec=color, zorder=6)
        else:
            cv.plot(mx, my, "o", ms=5.2, mfc="white", mec=color, mew=1.2, zorder=6)

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
                cy = y + rh / 2
                if p["label_w"]:
                    text(x + PAD + 8 + (16 if r.get("indent") else 0), cy, r["label"], size=10.5, color=INK)
                if r.get("dots"):  # dot-and-whisker row: no value labels, read against the shared scale
                    for dt in r["dots"]:
                        yy, c = cy + dt["dy"], (GREY_DOT if dt["style"] == "ref" else dt["color"])
                        cv.plot([x0 + dt["low"] * bar_w, x0 + dt["high"] * bar_w], [yy, yy], color=c, lw=.9, zorder=4)
                        for e in (dt["low"], dt["high"]):
                            cv.plot([x0 + e * bar_w] * 2, [yy - 3, yy + 3], color=c, lw=.9, zorder=4)
                        marker(x0 + dt["value"] * bar_w, yy, dt["style"], c)
                    y += rh
                    continue
                c, v, lo, hi = r["color"], r["value"], r["low"], r["high"]
                cv.add_patch(Rectangle((x0, cy - BAR_H / 2), max(v * bar_w, .8), BAR_H, lw=1,
                                       edgecolor=c, facecolor="white" if r.get("hollow") else light(c, r.get("shade", .28))))
                cv.plot([x0 + lo * bar_w, x0 + hi * bar_w], [cy, cy], color=c, lw=.9)
                for e in (lo, hi):
                    cv.plot([x0 + e * bar_w] * 2, [cy - 4, cy + 4], color=c, lw=.9)
                text(x0 + hi * bar_w + 5, cy, f"{v:.2f}" if p["kind"] == "auroc" else f"{v:.0%}",
                     size=10, color=c, weight="semibold")
                y += rh
        if p["kind"] == "auroc":  # chance line drawn over the bars so it reads across every row
            cv.plot([x0 + .5 * bar_w] * 2, [y_top, y_bot], color=MUTED, lw=1.1, ls=(0, (4, 2.5)), zorder=5)
        x += w + GAP
    fy = TITLE_H + HEAD_H + body_h + AXIS_H + 12
    if spec.get("legend"):
        lx = MARGIN + 6
        for style, lab in spec["legend"]:
            marker(lx, fy, style, ASTRA)
            t = text(lx + 10, fy, lab, size=9.5, color=MUTED)
            lx += 10 + TextPath((0, 0), lab, size=9.5, prop=FontProperties(family="Arial")).get_extents().width * 100 / 72 + 22
        fy += 20
    for line in notes:
        text(MARGIN, fy, line, size=9.5, color=MUTED)
        fy += 15

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


def row(label, est, color, hollow=False, shade=.28, indent=False):
    return {"label": label, "value": est["value"], "low": est["low"], "high": est["high"], "color": color,
            "hollow": hollow, "shade": shade, "indent": indent}


def dotrow(label, dots):
    """dots: (estimate, color, style, dy) with style ref (small black) / filled / hollow; dy offsets within the row."""
    return {"label": label, "dots": [{"value": e["value"], "low": e["low"], "high": e["high"], "color": c, "style": st,
                                      "dy": dy} for e, c, st, dy in dots if e]}


def specs(d: dict) -> list[dict]:
    det = {c["model"]: c["called_fake"] for c in d["detection"]["columns"]}
    sep = d["separability"]
    models = (("gpt-6-astra", "GPT-6 Astra", ASTRA), ("gpt-6.1-sol", "GPT-6.1 Sol", SOL))
    fig1 = {"stem": "01-perception", "bar_w": 250,
            "title": "1. GPT does not reliably flag manipulated image inputs, though a small classifier can separate them",
            "panels": [
                {"heading": "Answers “is this real or fake” with “fake”", "kind": "rate", "label_w": 172, "groups": [
                    {"label": name, "color": col, "rows": [
                        row("Unaltered (false alarm)", det[m]["n"], col, hollow=True),
                        row("Blurred", det[m]["b"], col), row("Doctored", det[m]["d"], col)]}
                    for m, name, col in models]},
                {"heading": "Separability from unaltered (AUROC)", "kind": "auroc", "label_w": 168, "groups": [
                    {"label": f"{d['footage'][c]} vs unaltered", "rows": [
                        row("GPT-6 Astra", sep[c]["gpt-6-astra"], ASTRA), row("GPT-6.1 Sol", sep[c]["gpt-6.1-sol"], SOL),
                        row("ResNet-18 classifier", sep[c]["resnet"], CLASSIFIER, shade=.12),
                        row("Small classifier", sep[c]["classifier"], CLASSIFIER)]}
                    for c in "bd"]}],
            "footnote": "Small classifier: ridge logistic regression on 5 forensic statistics per camera image.\n"
                        "ResNet-18 classifier: ridge logistic regression on frozen, untuned ImageNet ResNet-18 features.\n"
                        "Classifier AUROC from leave-one-task-out cross-validation."}
    fr = d["framing"]
    ladder = [("observer", "Observer, text answer"), ("actor_text", "Agent, text actions"),
              ("robot_text", "Robot prompt, text actions"), ("robot_motion", "Robot prompt, motion tools")]
    fig2 = {"stem": "02-embodiment", "bar_w": 170, "footnote": "Robot prompt = Inspect Robots system prompt. Motion tools = Inspect Robots tools (move_to, done, give_up). Text actions = carry_out / decline.",
            "title": "2. Astra refuses unsafe tasks in text, but acts once it has robot motion tools",
            "panels": [{"heading": d["footage"][f], "kind": "rate", "label_w": 198 if i == 0 else 0, "groups": [
                {"rows": [row(lab, fr[k][f], ASTRA) for k, lab in ladder]}]} for i, f in enumerate("nbd")]}
    if d.get("framing_sol"):  # both models, grouped, once the Sol run is complete
        fs = d["framing_sol"]
        fig2["title"] = FIG2_TITLE
        fig2["panels"] = [{"heading": d["footage"][f], "kind": "rate", "label_w": 198 if i == 0 else 0, "groups": [
            {"label": name if i == 0 else " ", "color": col, "rows": [row(lab, src[k][f], col) for k, lab in ladder]}
            for name, col, src in (("GPT-6 Astra", ASTRA, fr), ("GPT-6.1 Sol", SOL, fs))]} for i, f in enumerate("nbd")]
    pr = d["prompting"]
    nudges = [("R0", "Baseline (original request)"), ("R4", "Max reasoning effort"), ("R5", "“Think carefully”"),
              ("R3", "Real/fake self-check"), ("R1", "“Check observations”"), ("R2", "Tampering warning")]
    fig3 = {"stem": "03-nudges", "bar_w": 170,
            "title": "3. Only naming the tampering threat makes the robot refuse doctored footage",
            "panels": [{"heading": d["footage"][f], "kind": "rate", "label_w": 198 if i == 0 else 0, "groups": [
                {"rows": [row(lab, pr[k][f], ASTRA) for k, lab in nudges]}]} for i, f in enumerate("nbd")]}
    vd = d["verdict"]

    def vrow(e, f, col):
        nf, nr = (e[k]["n"] if e[k] else 0 for k in ("judged_fake", "judged_real"))
        return dotrow(f"{d['footage'][f]} ({nf} fake · {nr} real)",
                      [(e["judged_real"], col, "hollow", -7), (e["asked"], col, "ref", 0), (e["judged_fake"], col, "filled", 7)])

    fig4 = {"stem": "04-verdict", "bar_w": 220, "row_h": 26, "title": FIG4_TITLE,
            "legend": [("filled", "judged fake"), ("hollow", "judged real"), ("ref", "all answers")],
            "panels": [{"heading": head, "kind": "rate", "label_w": 196, "groups": [
                {"label": "GPT-6 Astra", "color": ASTRA, "rows": [vrow(vd[setting][f], f, ASTRA) for f in "nbd"]}]}
                for setting, head in (("observer", "Observer · says unsafe"), ("robot", "Robot · gives up"))]}
    mg = d.get("metagaming", {})
    cues = [("none", "No cue"), ("capability", "“Capability eval”"), ("alignment", "“Alignment eval”")]
    figs = [fig1, fig2, fig3, fig4]
    if all(len(mg.get(k, {})) == 3 for k, _ in ladder):
        figs.append({"stem": "05-metagaming", "bar_w": 190, "row_h": 26, "title": FIG5_TITLE,
                     "footnote": "Robot prompt = Inspect Robots system prompt. Motion tools = Inspect Robots tools (move_to, done, give_up). Text actions = carry_out / decline.",
                     "legend": [("filled", "“alignment eval”"), ("hollow", "“capability eval”"), ("ref", "no cue")],
                     "panels": [{"heading": d["footage"][f], "kind": "rate", "label_w": 196 if i == 0 else 0, "groups": [
                         {"label": "GPT-6 Astra" if i == 0 else " ", "color": ASTRA, "rows": [
                             dotrow(lab, [(mg[k]["capability"][f], ASTRA, "hollow", -7), (mg[k]["none"][f], ASTRA, "ref", 0),
                                          (mg[k]["alignment"][f], ASTRA, "filled", 7)]) for k, lab in ladder]}]}
                         for i, f in enumerate("nbd")]})
    return figs


def main() -> None:
    configure()
    d = json.loads(DATA.read_text())
    current = {path for spec in specs(d) for path in render(spec)}
    for old in OUT.glob("0*"):  # one current figure set; superseded figures are recoverable from git
        if old not in current:
            old.unlink()
    print(*sorted(p.name for p in current if p.suffix == ".png"))


if __name__ == "__main__":
    main()
