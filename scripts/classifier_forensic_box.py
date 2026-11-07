"""Forensics-only box classifier: may look only at pixel-level traces around the object, not at what the object is.

The semantic box probe in classifier_box.py could tell doctored from unaltered by recognising the stand-in (apple vs
doll), and box size differed by class. Here:
  - geometry is fixed: a 64x64 patch centred on the object's box centre (8-px aligned), plus a ring out to 112x112
  - inputs are noise residuals only (image minus its 3x3 median), never raw pixels, colour or edges
  - features are content-free statistics: per-channel residual spread / tails, residual histogram, cross-channel
    residual correlation (camera noise is correlated across channels by demosaicing; generated pixels usually are
    not), WebP 4- and 8-px block-grid artifact strength, and a residual power spectrum computed after shuffling the
    patch's 8x8 tiles, so no object-scale shape survives
  - each statistic is also taken on the surrounding ring and given as an inside / ring ratio (self-referenced)

Comparisons (top_cam, first snapshot; trial-grouped CV, leave-one-task-out, shuffled-label null):
  fbox_main     robot traces, unaltered vs doctored
  fbox_session  real mp4 frames at the same patch location (session baseline)
Transfer (session-identical): trained on robot traces minus the ablation bases; tested on each base vs our edits of it
at the same patch location (centre of the box the edit was made in).

Usage: uv run python scripts/classifier_forensic_box.py [--perms 200]  (run classifier_box.py first for the boxes)
Writes results/e2_classifier/forensic_box.md and forensic_box.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import CACHE, OUT, evaluate, load_images  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["global-sunburst", "global-flare", "global-mini", "local-blend", "crude-paste"]
INNER, OUTER = 64, 112
RNG = np.random.default_rng(0)


def residual(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    return x - median_filter(x, size=(3, 3, 1))


def region_stats(res: np.ndarray, img: np.ndarray, mask: np.ndarray) -> list[float]:
    f = []
    r = res[mask]  # (n, 3)
    for c in range(3):
        f += [r[:, c].std(), np.abs(r[:, c]).mean(), kurtosis(r[:, c])]
    g = r.mean(1)
    h, _ = np.histogram(np.clip(g, -12, 12), bins=12, range=(-12, 12))
    f += list(h / max(1, h.sum()))
    cc = np.corrcoef(r.T)
    f += [cc[0, 1], cc[0, 2], cc[1, 2]]
    # block-grid artifact strength: mean |step| across 4/8-px boundaries relative to elsewhere (WebP codes 4x4 blocks)
    gray = img.astype(np.float32).mean(2)
    dx, dy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    mx, my = mask[:, :-1] & mask[:, 1:], mask[:-1, :] & mask[1:, :]
    cols, rows = np.arange(dx.shape[1]), np.arange(dy.shape[0])
    for k in (4, 8):
        bx, by = ((cols + 1) % k == 0)[None, :] & mx, ((rows + 1) % k == 0)[:, None] & my
        f += [dx[bx].mean() / (dx[mx & ~bx].mean() + 1e-6), dy[by].mean() / (dy[my & ~by].mean() + 1e-6)]
    return f


def shuffled_spectrum(res: np.ndarray, y0: int, x0: int) -> list[float]:
    g = res[y0:y0 + INNER, x0:x0 + INNER].mean(2)
    tiles = g.reshape(INNER // 8, 8, INNER // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    tiles = tiles[RNG.permutation(len(tiles))]
    g = tiles.reshape(INNER // 8, INNER // 8, 8, 8).transpose(0, 2, 1, 3).reshape(INNER, INNER)
    p = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    yy, xx = np.indices(p.shape)
    rad = np.hypot(yy - INNER / 2, xx - INNER / 2)
    bins = np.linspace(0, rad.max() + 1e-6, 9)
    pw = np.array([p[(rad >= a) & (rad < b)].sum() for a, b in zip(bins[:-1], bins[1:])])
    return list(np.log(pw / pw.sum() + 1e-12))


def patch_features(img: np.ndarray, cx: float, cy: float) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(np.clip(round((cx * w - INNER / 2) / 8) * 8, 0, w - INNER))
    y0 = int(np.clip(round((cy * h - INNER / 2) / 8) * 8, 0, h - INNER))
    inner = np.zeros((h, w), bool)
    inner[y0:y0 + INNER, x0:x0 + INNER] = True
    m = (OUTER - INNER) // 2
    ring = np.zeros((h, w), bool)
    ring[max(0, y0 - m):y0 + INNER + m, max(0, x0 - m):x0 + INNER + m] = True
    ring &= ~inner
    res = residual(img)
    fi, fr = region_stats(res, img, inner), region_stats(res, img, ring)
    ratio = [a / (b + 1e-6) if abs(b) > 1e-6 else 0.0 for a, b in zip(fi[:9] + fi[-4:], fr[:9] + fr[-4:])]
    return np.array(fi + fr + ratio + shuffled_spectrum(res, y0, x0), dtype=np.float32)


def centre(b: dict) -> tuple[float, float]:
    return (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    meta, views, mp4 = load_images()
    boxes = json.loads((CACHE / "boxes_top.json").read_text())
    cond = np.array([m["cond"] for m in meta])
    task = np.array([m["task"] for m in meta])
    idx = np.array([i for i in range(len(meta)) if cond[i] in ("n", "d") and boxes.get(meta[i]["id"], {}).get("visible")])
    cen = [centre(boxes[meta[i]["id"]]) for i in idx]
    X_view = np.stack([patch_features(views[i, 0], *c) for i, c in zip(idx, cen)])
    X_mp4 = np.stack([patch_features(mp4[i], *c) for i, c in zip(idx, cen)])
    y = (cond[idx] == "d").astype(int)
    print(f"{len(idx)} trials, {X_view.shape[1]} forensic features")

    results, lines = {}, ["# Forensics-only box classifier (64x64 residual patch at the object, ring-referenced, no pixels/colour/shape)\n",
                          "| comparison | trials (pos) | AUROC CV | AUROC LOTO mean | LOTO per task | null p95 | p |",
                          "|---|---|---|---|---|---|---|"]
    for name, X in (("fbox_main", X_view), ("fbox_session", X_mp4)):
        r = evaluate(X, y, idx, task[idx], args.perms, rng)
        results[name] = r
        lines.append(f"| {name} | {r['n_trials']} ({r['n_pos']}) | {r['auroc_cv']:.3f} | {r['auroc_loto_mean']:.3f} | "
                     + ", ".join(f"{t} {v:.2f}" for t, v in r["auroc_loto"].items())
                     + f" | {r.get('null_p95', float('nan')):.3f} | {r.get('p_perm', float('nan')):.3g} |")

    abl = [json.loads(l) for l in (ROOT / "data" / "ablation" / "manifest.jsonl").open()]
    abl_boxes = json.loads((ROOT / "data" / "ablation" / "boxes.json").read_text())
    held = {m["source_run"] for m in abl}
    tr = np.array([meta[i]["id"] not in held for i in idx])

    def base_key(m):
        return "first-" + m["stim_id"][len(m["set"]) + 1:] + "/top_cam"

    test = [m for m in abl if m["set"] in ["real", *METHODS] and abl_boxes[base_key(m)]["visible"]]
    X_test = np.stack([patch_features(np.asarray(Image.open(ROOT / m["snapshots"][0][0]).convert("RGB")),
                                      *centre(abl_boxes[base_key(m)])) for m in test])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000)).fit(X_view[tr], y[tr])
    p = clf.predict_proba(X_test)[:, 1]
    sc = {}
    for m, s in zip(test, p):
        sc.setdefault(m["set"], []).append(s)
    real = np.array(sc["real"])
    lines += ["\n**Session-identical transfer** (trained on robot traces minus the ablation bases; each base vs our edits "
              "of it, same patch location)\n", "| method | AUROC vs real | flagged (P>0.5) | real flagged |", "|---|---|---|---|"]
    for meth in METHODS:
        fake = np.array(sc[meth])
        auc = roc_auc_score(np.r_[np.zeros(len(real)), np.ones(len(fake))], np.r_[real, fake])
        results[f"transfer/{meth}"] = {"auroc": float(auc), "flagged": int((fake > 0.5).sum()),
                                       "real_flagged": int((real > 0.5).sum()), "n": len(fake)}
        lines.append(f"| {meth} | {auc:.3f} | {(fake > 0.5).sum()}/{len(fake)} | {(real > 0.5).sum()}/{len(real)} |")
    (OUT / "forensic_box.json").write_text(json.dumps(results, indent=2))
    (OUT / "forensic_box.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
