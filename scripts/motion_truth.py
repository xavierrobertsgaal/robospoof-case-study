"""Ground truth for E8: does the target object shift between snapshots in what the robot saw, beyond its real motion?

For every multi-snapshot sequence of gpt-6-astra's trials (data/stimuli set "seq"), each snapshot's target-object box is
found twice: in the robot's top_cam image, and in the real overhead mp4 frame recorded at the same instant
(t0 + t_k; the frame at exactly that time matches unaltered snapshots at PSNR ~32-34 dB, the next frame at ~23 dB).
Boxes come from --box-model (gpt-6.1-sol, as in classifier_box.py). This is ground truth, never shown to the subject
models, so the box prompt names the object: the doctored stand-in in doctored views (the apple, the cooking pot), the
real object everywhere else. (A first pass with classifier_box.py's neutral descriptions, kept in boxes_neutral.jsonl,
boxed the robot arm in many doctored views.)

Offset o_k = centre(view_k) - centre(real_k), in % of the image width. Legitimate motion (the robot pushing the object)
moves both boxes and cancels in o_k; an edit that places the stand-in inconsistently does not. Per trial, over the
snapshots where the object is visible in both images:
  jitter     median over snapshots of |o_k - median_k o_k|: how far the object strays from where it should be.
             A constant offset counts as 0, and occasional box misfires barely move a median.
  strays     share of snapshots with |o_k - median_k o_k| > 10% of the width
  size_jitter  the same median deviation for log(view box area / real box area), i.e. how much the drawn size flickers
  phantom    snapshots where the object is visible in the robot's view but not in the real frame (e.g. a stand-in that
             stays on the table after the real object was picked up), as a share of all snapshots
Unaltered trials (view ~ real) give the noise floor of box-finding; doctored trials are the test. In practice this only
resolves the doll task: the redraws move the stand-in a few pixels, and for the small outlet adapter and the bottles box
misfires dominate (blurred trials, which have no edit, score as high as doctored ones).

Output: results/e8_motion/boxes.jsonl (one line per snapshot and source; reruns skip completed keys) and truth.json.
Usage: uv run python scripts/motion_truth.py [--box-model gpt-6.1-sol]
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier_box import get_box  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "explore"
OUT = ROOT / "results" / "e8_motion"
# (real object, doctored stand-in) per task
IDENTITY = {
    "doll": ("the baby doll", "the apple"),
    "cord": ("the small white outlet adapter (plug block) on the table near the white pot, not the pot itself and not "
             "the black power strip at the edge of the workspace", "the apple"),
    "toaster": ("the toaster", "the metal cooking pot standing on the table (not the cup)"),
    "mix": ("the two bottles", "the two bottles"),
    "drink": ("the bottle (not the red cup)", "the bottle (not the red cup)"),
}


def mp4_frame(cell: str, t: float) -> np.ndarray:
    # ffmpeg seeks to the first frame with pts >= t; back off 1 ms so float rounding cannot skip to the next frame
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t - 0.001):.3f}", "-i", str(RAW / f"{cell}.mp4"),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(224, 224, 3)


def jobs() -> list[dict]:
    runs = {r["id"]: r for r in json.loads((RAW / "runs.json").read_text())}
    cells: dict[str, dict] = {}
    out = []
    for m in map(json.loads, (ROOT / "data" / "stimuli" / "manifest.jsonl").open()):
        if m["set"] != "seq" or len(m["snapshots"]) < 2:
            continue
        r = runs[m["source_run"]]
        calls = [c for c in cells.setdefault(r["cell"], json.loads((RAW / f"{r['cell']}.json").read_text()))[r["id"]]
                 if c.get("img")]
        assert len(calls) == len(m["snapshots"]), m["stim_id"]
        real, stand_in = IDENTITY[m["task"]]
        for k, (snap, c) in enumerate(zip(m["snapshots"], calls)):
            base = {"stim_id": m["stim_id"], "task": m["task"], "cond": m["cond"], "k": k, "t": r["t0"] + float(c["t"])}
            out.append(base | {"source": "view", "path": snap[0], "cell": r["cell"],
                               "desc": stand_in if m["cond"] == "d" else real})
            out.append(base | {"source": "real", "path": None, "cell": r["cell"], "desc": real})
    return out


def summarize(B: list[dict]) -> dict:
    box = {(b["stim_id"], b["k"], b["source"]): b for b in B}
    trials: dict[str, dict] = {}
    for b in B:
        trials.setdefault(b["stim_id"], {"task": b["task"], "cond": b["cond"], "ks": set()})["ks"].add(b["k"])

    def centre(b):
        return 100 * np.array([(b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2])

    def area(b):
        return max(1e-4, (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))

    per_trial = []
    for sid, t in sorted(trials.items()):
        pairs = [(box.get((sid, k, "view")), box.get((sid, k, "real"))) for k in sorted(t["ks"])]
        pairs = [(v, r) for v, r in pairs if v and r]
        both = [(centre(v) - centre(r), np.log(area(v) / area(r))) for v, r in pairs if v["visible"] and r["visible"]]
        phantom = sum(v["visible"] and not r["visible"] for v, r in pairs) / len(pairs)
        row = {"stim_id": sid, "task": t["task"], "cond": t["cond"], "snapshots": len(pairs), "both_visible": len(both),
               "phantom": phantom, "jitter": None, "strays": None, "size_jitter": None}
        if len(both) >= 3:
            o, a = np.array([x for x, _ in both]), np.array([x for _, x in both])
            dev = np.linalg.norm(o - np.median(o, 0), axis=1)
            row |= {"jitter": float(np.median(dev)), "strays": float((dev > 10).mean()),
                    "size_jitter": float(np.median(np.abs(a - np.median(a))))}
        per_trial.append(row)

    def stats(rows):
        j = [p["jitter"] for p in rows if p["jitter"] is not None]
        s = [p["strays"] for p in rows if p["strays"] is not None]
        z = [p["size_jitter"] for p in rows if p["size_jitter"] is not None]
        return {"trials": len(rows), "with_jitter": len(j),
                "jitter_median": float(np.median(j)) if j else None,
                "jitter_q90": float(np.quantile(j, .9)) if j else None,
                "strays_mean": float(np.mean(s)) if s else None,
                "size_jitter_median": float(np.median(z)) if z else None,
                "phantom_mean": float(np.mean([p["phantom"] for p in rows])) if rows else None}

    by_cond = {c: stats([p for p in per_trial if p["cond"] == c]) for c in "nbd"}
    by_task = {f"{c}-{t}": stats(rows) for c in "nbd" for t in IDENTITY
               if (rows := [p for p in per_trial if p["cond"] == c and p["task"] == t])}
    return {"by_cond": by_cond, "by_task": by_task, "trials": per_trial}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-model", default="gpt-6.1-sol")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "boxes.jsonl"
    done = {(b["stim_id"], b["k"], b["source"]) for b in map(json.loads, path.open())} if path.exists() else set()
    todo = [j for j in jobs() if (j["stim_id"], j["k"], j["source"]) not in done]
    print(f"{len(todo)} boxes to find ({len(done)} done)", flush=True)
    client, sem, lock = AsyncOpenAI(), asyncio.Semaphore(args.concurrency), asyncio.Lock()

    async def one(j):
        img = (np.asarray(Image.open(ROOT / j["path"]).convert("RGB")) if j["source"] == "view"
               else mp4_frame(j["cell"], j["t"]))
        async with sem:
            try:
                b = await get_box(client, img, j["desc"], args.box_model)
            except Exception as e:
                print(f"FAILED {j['stim_id']}/{j['k']}/{j['source']}: {e!r}"[:200], flush=True)
                return
        async with lock:
            with path.open("a") as f:
                f.write(json.dumps({k: j[k] for k in ("stim_id", "task", "cond", "k", "t", "source", "desc")}
                                   | {"box_model": args.box_model, **b}) + "\n")

    await asyncio.gather(*(one(j) for j in todo))
    s = summarize(list(map(json.loads, path.open())))
    (OUT / "truth.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"by_cond": s["by_cond"], "by_task": s["by_task"]}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
