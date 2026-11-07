"""Session-identical test of the robot-trace classifier.

Train only on the robot's traces: unaltered vs paper-doctored first snapshots (all source models), holding out the
trials used as bases in data/ablation. Test on session-identical pairs: each held-out base snapshot (real) against our
edits of that same snapshot (global-sunburst / flare / mini, local-blend, crude-paste; 224 px, one WebP q75 pass).
Real and edited images share session, lighting, layout and robot pose, so only the edit can separate them.

Reports, per method: AUROC of edited vs real (stimulus score = mean P(doctored) over the three cameras) and the share
flagged at P > 0.5, next to the real bases' false-positive rate.
Usage: uv run python scripts/classifier_transfer.py   (run scripts/classifier.py and scripts/edit_ablation.py first)
Writes results/e2_classifier/transfer.md and transfer.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import CACHE, OUT, forensic, resnet_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["global-sunburst", "global-flare", "global-mini", "local-blend", "crude-paste"]


def main() -> None:
    z = np.load(CACHE / "first_snapshot_features.npz", allow_pickle=True)
    meta = list(z["meta"])
    abl = [json.loads(l) for l in (ROOT / "data" / "ablation" / "manifest.jsonl").open()]
    held_out = {m["source_run"] for m in abl}
    cond = np.array([m["cond"] for m in meta])
    train = np.array([m["id"] not in held_out for m in meta]) & np.isin(cond, ["n", "d"])
    print(f"training on {train.sum()} robot trials ({(cond[train] == 'd').sum()} doctored); {len(held_out)} bases held out")

    test = [m for m in abl if m["set"] in ["real", *METHODS]]
    imgs = np.stack([np.asarray(Image.open(ROOT / p).convert("RGB")) for m in test for p in m["snapshots"][0]])
    test_feats = {"forensic": np.stack([forensic(im) for im in imgs]), "resnet": resnet_features(imgs)}

    results, lines = {}, ["# Session-identical test: classifier trained on robot traces, tested on edits of held-out robot snapshots\n",
                          "| features | method | AUROC vs real | flagged (P>0.5) | real bases flagged |", "|---|---|---|---|---|"]
    for feat in ("forensic", "resnet"):
        V = z[f"view_{feat}"][train]
        X = V.reshape(-1, V.shape[-1])
        y = np.repeat((cond[train] == "d").astype(int), V.shape[1])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, y)
        p = clf.predict_proba(test_feats[feat])[:, 1].reshape(len(test), 3).mean(1)
        score = {}
        for m, s in zip(test, p):
            score.setdefault(m["set"], []).append(s)
        real = np.array(score["real"])
        for meth in METHODS:
            fake = np.array(score[meth])
            auc = roc_auc_score(np.r_[np.zeros(len(real)), np.ones(len(fake))], np.r_[real, fake])
            results[f"{feat}/{meth}"] = {"auroc": float(auc), "flagged": float((fake > 0.5).mean()),
                                         "real_flagged": float((real > 0.5).mean()), "n": len(fake)}
            lines.append(f"| {feat} | {meth} | {auc:.3f} | {(fake > 0.5).sum()}/{len(fake)} | {(real > 0.5).sum()}/{len(real)} |")
    (OUT / "transfer.json").write_text(json.dumps(results, indent=2))
    (OUT / "transfer.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
