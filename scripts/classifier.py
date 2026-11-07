"""Can a small trained classifier separate unaltered from doctored first snapshots, once shortcuts are controlled?

Data: the first snapshot of every trial (all 5 source models, 900 trials), split into its three camera images, plus the
real overhead mp4 frame recorded at the same instant. The mp4 frames are the camera's ground truth: on unaltered
trials they match the model-view top_cam at sigma=0 (PSNR ~29-35 dB) and on blurred trials a sigma=4 Gaussian blur
of them reproduces the model view (PSNR ~34-39 dB), as the paper states.

Comparisons (each with trial-grouped 5-fold CV, leave-one-task-out, and a shuffled-label null):
  main          model-view unaltered vs doctored, all three cameras (trial score = mean over cameras)
  main_top      model-view unaltered vs doctored, top_cam only
  session_top   real mp4 frames of unaltered-condition vs doctored-condition trials. Both are real footage, so any
                separability is session / day / rig / lighting, which the conditions are confounded with (unaltered
                was mostly recorded on 2026-10-24, doctored on 10-25..27). main_top must beat this to count.
  pos_blur      model-view unaltered vs blurred (positive control; trivially separable if the pipeline works)
Features: 'forensic' (noise-residual and power-spectrum shape statistics; no colour or brightness means) and
'resnet' (frozen ImageNet ResNet-18 penultimate layer, linear probe).

Usage: uv run python scripts/classifier.py [--perms 200]
Writes results/e2_classifier/summary.md and summary.json (features cached in data/classifier/).
"""

import argparse
import base64
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "classifier"
OUT = ROOT / "results" / "e2_classifier"
CAMS = ["top_cam", "left_cam", "right_cam"]
TASKS = ["doll", "cord", "toaster", "mix", "drink"]


def load_images() -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Return trial metadata, model-view first snapshots (N, 3, 224, 224, 3) and mp4 frames (N, 224, 224, 3)."""
    runs = json.loads((RAW / "explore" / "runs.json").read_text())
    meta, views, mp4 = [], [], []
    cells: dict[str, dict] = {}
    for r in runs:
        cells.setdefault(r["cell"], json.loads((RAW / "explore" / f"{r['cell']}.json").read_text()))
        calls = cells[r["cell"]].get(r["id"], [])
        if not calls or not calls[0].get("img"):
            continue
        comp = np.asarray(Image.open(io.BytesIO(base64.b64decode(calls[0]["img"]))).convert("RGB"))
        views.append(np.stack([comp[:, i * 224:(i + 1) * 224] for i in range(3)]))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{r['t0'] + 0.01:.3f}", "-i",
                              str(RAW / "explore" / f"{r['cell']}.mp4"), "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        mp4.append(np.frombuffer(raw, np.uint8).reshape(224, 224, 3))
        meta.append({"id": r["id"], "task": r["task"], "cond": r["cond"], "model": r["model"]})
    return meta, np.stack(views), np.stack(mp4)


def forensic(img: np.ndarray) -> np.ndarray:
    """Content-light statistics: noise residual distribution and power-spectrum shape, normalised per image."""
    x = img.astype(np.float32)
    res = x - median_filter(x, size=(3, 3, 1))
    feats = []
    for c in range(3):
        rc = res[..., c].ravel()
        feats += [rc.std(), np.abs(rc).mean(), kurtosis(rc)]
    g = x.mean(2)
    rg = res.mean(2).ravel()
    hist, _ = np.histogram(np.clip(rg, -16, 16), bins=16, range=(-16, 16))
    feats += list(hist / hist.sum())
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    yy, xx = np.indices(f.shape)
    rad = np.hypot(yy - f.shape[0] / 2, xx - f.shape[1] / 2)
    bins = np.linspace(0, rad.max() + 1e-6, 13)
    power = np.array([f[(rad >= a) & (rad < b)].sum() for a, b in zip(bins[:-1], bins[1:])])
    feats += list(np.log(power / power.sum() + 1e-12))
    gy, gx = np.gradient(g)
    feats += [np.hypot(gx, gy).mean() / (g.std() + 1e-6), res.var() / (x.var() + 1e-6)]
    return np.array(feats, dtype=np.float32)


def resnet_features(imgs: np.ndarray, batch: int = 64) -> np.ndarray:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval().to(dev)
    model.fc = torch.nn.Identity()
    tf = weights.transforms()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            x = torch.stack([tf(Image.fromarray(im)) for im in imgs[i:i + batch]]).to(dev)
            out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def cv_scores(X: np.ndarray, y: np.ndarray, trial: np.ndarray, splits) -> np.ndarray:
    """Out-of-fold P(class 1) per row, for the given (train_idx, test_idx) splits over rows."""
    p = np.full(len(y), np.nan)
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def per_trial(p: np.ndarray, y: np.ndarray, trial: np.ndarray, task: np.ndarray):
    ids = np.unique(trial)
    pt = np.array([np.nanmean(p[trial == i]) for i in ids])
    yt = np.array([y[trial == i][0] for i in ids])
    tt = np.array([task[trial == i][0] for i in ids])
    return pt, yt, tt


def evaluate(X, y, trial, task, perms: int, rng: np.random.Generator) -> dict:
    ids = np.unique(trial)
    y_trial = np.array([y[trial == i][0] for i in ids])
    t_trial = np.array([task[trial == i][0] for i in ids])
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    def grouped(yy):
        return list(skf.split(X, yy, groups=trial))

    pt, yt, tt = per_trial(cv_scores(X, y, trial, grouped(y)), y, trial, task)
    res = {"n_trials": int(len(ids)), "n_pos": int(yt.sum()), "auroc_cv": float(roc_auc_score(yt, pt)),
           "bal_acc_cv": float(((pt > 0.5) == yt)[yt == 1].mean() / 2 + ((pt > 0.5) == yt)[yt == 0].mean() / 2)}
    loto = {}
    for t in TASKS:
        te = task == t
        if len(np.unique(y[te])) < 2:
            continue
        p = cv_scores(X, y, trial, [(np.where(~te)[0], np.where(te)[0])])
        a, b, _ = per_trial(p[te], y[te], trial[te], task[te])
        loto[t] = float(roc_auc_score(b, a))
    res["auroc_loto"] = loto
    res["auroc_loto_mean"] = float(np.mean(list(loto.values()))) if loto else None
    null = []
    for _ in range(perms):
        perm_trial = y_trial.copy()
        for t in np.unique(t_trial):  # shuffle trial labels within task, keeping task balance
            m = t_trial == t
            perm_trial[m] = rng.permutation(perm_trial[m])
        lab = dict(zip(ids, perm_trial))
        yp = np.array([lab[i] for i in trial])
        a, b, _ = per_trial(cv_scores(X, yp, trial, grouped(yp)), yp, trial, task)
        null.append(roc_auc_score(b, a))
    if null:
        null = np.array(null)
        res.update({"null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
                    "p_perm": float((1 + (null >= res["auroc_cv"]).sum()) / (1 + len(null)))})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    fpath = CACHE / "first_snapshot_features.npz"
    if fpath.exists():
        z = np.load(fpath, allow_pickle=True)
        meta, F = list(z["meta"]), {k: z[k] for k in z.files if k != "meta"}
    else:
        meta, views, mp4 = load_images()
        print(f"loaded {len(meta)} trials")
        F = {"view_forensic": np.stack([[forensic(v) for v in tri] for tri in views]),
             "mp4_forensic": np.stack([forensic(m) for m in mp4]),
             "view_resnet": resnet_features(views.reshape(-1, 224, 224, 3)).reshape(len(meta), 3, -1),
             "mp4_resnet": resnet_features(mp4)}
        np.savez(fpath, meta=np.array(meta, dtype=object), **F)

    cond = np.array([m["cond"] for m in meta])
    task = np.array([m["task"] for m in meta])
    results = {}
    for feat in ("forensic", "resnet"):
        V, M = F[f"view_{feat}"], F[f"mp4_{feat}"]
        for name, (a, b), cams, src in (("main", ("n", "d"), 3, V), ("main_top", ("n", "d"), 1, V),
                                         ("session_top", ("n", "d"), 1, M), ("pos_blur", ("n", "b"), 3, V)):
            keep = np.isin(cond, [a, b])
            idx = np.where(keep)[0]
            if src is V:
                X = src[idx, :cams].reshape(-1, src.shape[-1])
                rows = np.repeat(idx, cams)
            else:
                X, rows = src[idx], idx
            y = (cond[rows] == b).astype(int)
            r = evaluate(X, y, rows, task[rows], args.perms if name != "pos_blur" else 0, rng)
            results[f"{name}/{feat}"] = r
            print(f"{name:12} {feat:9} AUROC cv {r['auroc_cv']:.3f}  LOTO {r['auroc_loto_mean']:.3f}  "
                  f"null {r.get('null_mean', float('nan')):.3f} (p95 {r.get('null_p95', float('nan')):.3f})")

    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    lines = ["# Small-classifier separability of first snapshots\n",
             "Trial-level AUROC (5-fold CV grouped by trial; leave-one-task-out mean; shuffled-label null mean / 95th pct).\n",
             "| comparison | features | trials (pos) | AUROC CV | balanced acc | AUROC LOTO mean | null mean | null p95 | p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        name, feat = k.split("/")
        lines.append(f"| {name} | {feat} | {r['n_trials']} ({r['n_pos']}) | {r['auroc_cv']:.3f} | {r['bal_acc_cv']:.3f} | "
                     f"{r['auroc_loto_mean']:.3f} | {r.get('null_mean', float('nan')):.3f} | "
                     f"{r.get('null_p95', float('nan')):.3f} | {r.get('p_perm', float('nan')):.3g} |")
    lines.append("\nLOTO per task: " + "; ".join(f"{k}: " + ", ".join(f"{t} {v:.2f}" for t, v in r["auroc_loto"].items())
                                                for k, r in results.items()))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
