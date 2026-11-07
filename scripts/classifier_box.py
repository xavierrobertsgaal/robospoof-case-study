"""Box-only classifier: train on the robot's traces using only the pixels inside the target object's box.

The whole-frame classifier keys on the 2.5 editors' whole-frame re-render, so local edits (real pixels outside the
object box) mostly evade it. Here features come only from the top_cam crop around the object the instruction refers
to. Boxes come from --box-model (default gpt-6.1-sol, so box-finding does not compete with the subject model's image
rate limit) with a neutral per-task description that fits both the real object and its doctored
stand-in (e.g. "the object beside the knife"), so the box prompt does not presuppose either.

Comparisons (top_cam, first snapshot; trial-grouped CV, leave-one-task-out, shuffled-label null):
  box_main      robot traces, unaltered vs doctored, inside-box crop. Inside the box the content differs (doll vs apple,
                toaster vs pot), so leave-one-task-out is the test that matters: does it transfer to an unseen swap?
  box_session   real mp4 frames cropped at the same box, unaltered- vs doctored-condition trials (session baseline)
Transfer (session-identical): train box_main on robot traces minus the ablation bases, test on each base vs our edits of
it, cropped at the box the edit was made in (data/ablation/boxes.json).

Usage: uv run python scripts/classifier_box.py [--perms 200]
Writes results/e2_classifier/box.md and box.json (boxes cached in data/classifier/boxes_top.json).
"""

import argparse
import asyncio
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import CACHE, OUT, evaluate, forensic, load_images, resnet_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT.parent / ".env")
NEUTRAL = {
    "doll": "the object lying on the table beside the knife",
    "cord": "the small object on the table next to the white pot (not the pot itself, and not the black power strip "
            "at the edge of the workspace)",
    "toaster": "the larger object on the table next to the cup",
    "mix": "the bottles on the table",
    "drink": "the bottle on the table",
}
BOX_SCHEMA = {"type": "object", "properties": {
    "visible": {"type": "boolean"}, "x0": {"type": "number"}, "y0": {"type": "number"},
    "x1": {"type": "number"}, "y1": {"type": "number"}},
    "required": ["visible", "x0", "y0", "x1", "y1"], "additionalProperties": False}
METHODS = ["global-sunburst", "global-flare", "global-mini", "local-blend", "crude-paste"]
PAD = 8


async def get_box(client: AsyncOpenAI, img: np.ndarray, obj: str, model: str) -> dict:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    for attempt in range(6):
        try:
            return await _box_call(client, buf.getvalue(), obj, model)
        except Exception as e:
            if attempt == 5:
                raise
            print(f"box retry {attempt + 1}: {e!r}"[:160])
            await asyncio.sleep(min(60, 5 * 2 ** attempt))


async def _box_call(client: AsyncOpenAI, png: bytes, obj: str, model: str) -> dict:
    r = await client.responses.create(
        model=model, store=False, reasoning={"effort": "medium"},
        input=[{"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png).decode(),
             "detail": "auto"},
            {"type": "input_text", "text": f"Give the bounding box of {obj} in this image as fractions of image width "
                                           f"and height (x0, y0 top-left; x1, y1 bottom-right), padded slightly to "
                                           f"cover it fully. If it is not visible, set visible=false and all "
                                           f"coordinates to 0."}]}],
        text={"format": {"type": "json_schema", "name": "box", "schema": BOX_SCHEMA, "strict": True}})
    return json.loads(r.output_text)


def crop(img: np.ndarray, b: dict) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = max(0, int(b["x0"] * w) - PAD), max(0, int(b["y0"] * h) - PAD)
    x1, y1 = min(w, int(np.ceil(b["x1"] * w)) + PAD), min(h, int(np.ceil(b["y1"] * h)) + PAD)
    return img[y0:max(y1, y0 + 16), x0:max(x1, x0 + 16)]


def featurize(crops: list[np.ndarray]) -> dict[str, np.ndarray]:
    sq = np.stack([np.asarray(Image.fromarray(c).resize((224, 224), Image.BICUBIC)) for c in crops])
    return {"forensic": np.stack([forensic(c) for c in crops]), "resnet": resnet_features(sq)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--box-model", default="gpt-6.1-sol")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    meta, views, mp4 = load_images()
    cond = np.array([m["cond"] for m in meta])
    task = np.array([m["task"] for m in meta])
    keep = np.where(np.isin(cond, ["n", "d"]))[0]
    top = views[:, 0]

    box_path = CACHE / "boxes_top.json"
    boxes = json.loads(box_path.read_text()) if box_path.exists() else {}
    client, sem = AsyncOpenAI(), asyncio.Semaphore(16)

    async def job(i):
        if meta[i]["id"] not in boxes:
            async with sem:
                boxes[meta[i]["id"]] = await get_box(client, top[i], NEUTRAL[meta[i]["task"]], args.box_model)
                box_path.write_text(json.dumps(boxes, indent=2))  # incremental, so a crash keeps finished boxes

    await asyncio.gather(*(job(i) for i in keep))
    vis = np.array([boxes[meta[i]["id"]]["visible"] for i in keep])
    print(f"boxes: {vis.sum()}/{len(keep)} visible")
    idx = keep[vis]

    F_view = featurize([crop(top[i], boxes[meta[i]["id"]]) for i in idx])
    F_mp4 = featurize([crop(mp4[i], boxes[meta[i]["id"]]) for i in idx])
    y = (cond[idx] == "d").astype(int)
    results = {}
    for feat in ("forensic", "resnet"):
        for name, X in (("box_main", F_view[feat]), ("box_session", F_mp4[feat])):
            r = evaluate(X, y, idx, task[idx], args.perms, rng)
            results[f"{name}/{feat}"] = r
            print(f"{name:12} {feat:9} AUROC cv {r['auroc_cv']:.3f}  LOTO {r['auroc_loto_mean']:.3f} {r['auroc_loto']}  "
                  f"null p95 {r.get('null_p95', float('nan')):.3f}")

    # session-identical transfer: train on robot traces minus ablation bases, test on base vs edits at the edit box
    abl = [json.loads(l) for l in (ROOT / "data" / "ablation" / "manifest.jsonl").open()]
    abl_boxes = json.loads((ROOT / "data" / "ablation" / "boxes.json").read_text())
    held = {m["source_run"] for m in abl}
    tr = np.array([meta[i]["id"] not in held for i in idx])
    def base_key(m):  # ablation stim ids are "<set>-<task>-<cond>-<trial>"; boxes are keyed "first-<task>-n-<trial>/<cam>"
        return "first-" + m["stim_id"][len(m["set"]) + 1:] + "/top_cam"

    test = [m for m in abl if m["set"] in ["real", *METHODS] and abl_boxes[base_key(m)]["visible"]]
    F_test = featurize([crop(np.asarray(Image.open(ROOT / m["snapshots"][0][0]).convert("RGB")), abl_boxes[base_key(m)])
                        for m in test])
    lines = ["# Box-only classifier (top_cam, pixels inside the target object's box)\n",
             "| comparison | features | trials (pos) | AUROC CV | AUROC LOTO mean | LOTO per task | null p95 |",
             "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        name, feat = k.split("/")
        lines.append(f"| {name} | {feat} | {r['n_trials']} ({r['n_pos']}) | {r['auroc_cv']:.3f} | {r['auroc_loto_mean']:.3f} | "
                     + ", ".join(f"{t} {v:.2f}" for t, v in r["auroc_loto"].items()) + f" | {r.get('null_p95', float('nan')):.3f} |")
    lines += ["\n**Session-identical transfer** (trained on robot traces minus the 20 ablation bases; tested on each base "
              "vs our edits of it, cropped at the edit box; bases whose top_cam box was not visible are dropped)\n",
              "| features | method | AUROC vs real | flagged (P>0.5) | real flagged |", "|---|---|---|---|---|"]
    for feat in ("forensic", "resnet"):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(F_view[feat][tr], y[tr])
        p = clf.predict_proba(F_test[feat])[:, 1]
        sc = {}
        for m, s in zip(test, p):
            sc.setdefault(m["set"], []).append(s)
        real = np.array(sc["real"])
        for meth in METHODS:
            fake = np.array(sc[meth])
            auc = roc_auc_score(np.r_[np.zeros(len(real)), np.ones(len(fake))], np.r_[real, fake])
            results[f"transfer/{feat}/{meth}"] = {"auroc": float(auc), "flagged": int((fake > 0.5).sum()),
                                                  "real_flagged": int((real > 0.5).sum()), "n": len(fake)}
            lines.append(f"| {feat} | {meth} | {auc:.3f} | {(fake > 0.5).sum()}/{len(fake)} | {(real > 0.5).sum()}/{len(real)} |")
    (OUT / "box.json").write_text(json.dumps(results, indent=2))
    (OUT / "box.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
