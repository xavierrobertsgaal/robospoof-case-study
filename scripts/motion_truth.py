"""Ground truth for E8: does the target object shift between snapshots in what the robot saw, beyond its real motion?

For every multi-snapshot sequence of gpt-6-astra's trials (data/stimuli set "seq"), each snapshot's target-object box is
found twice: in the robot's top_cam image, and in the real overhead mp4 frame recorded at the same instant
(t0 + t_k; the frame at exactly that time matches unaltered snapshots at PSNR ~32-34 dB, the next frame at ~23 dB).
Boxes come from --box-model (gpt-6.1-sol, as in classifier_box.py) with the same neutral per-task descriptions, which
fit both the real object and its doctored stand-in.

Offset o_k = centre(view_k) - centre(real_k), in fractions of the image. Legitimate motion (the robot pushing the object)
moves both boxes and cancels in o_k; an edit that places the stand-in inconsistently does not. Per consecutive pair of
snapshots where the object is visible in all four images:
  shift_k   |o_{k+1} - o_k| (Euclidean, % of image width)    unexplained change of position
  resize_k  |log(area ratio view/real)_{k+1} - (..)_k|          unexplained change of size
Per trial: the largest shift and resize over its pairs. Unaltered and blurred trials (real footage) give the noise floor
of box-finding; doctored trials are the test.

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
from classifier_box import NEUTRAL, get_box  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "explore"
OUT = ROOT / "results" / "e8_motion"


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
        for k, (snap, c) in enumerate(zip(m["snapshots"], calls)):
            base = {"stim_id": m["stim_id"], "task": m["task"], "cond": m["cond"], "k": k, "t": r["t0"] + float(c["t"])}
            out.append(base | {"source": "view", "path": snap[0], "cell": r["cell"]})
            out.append(base | {"source": "real", "path": None, "cell": r["cell"]})
    return out


def summarize(B: list[dict]) -> dict:
    box = {(b["stim_id"], b["k"], b["source"]): b for b in B}
    trials: dict[str, dict] = {}
    for b in B:
        trials.setdefault(b["stim_id"], {"task": b["task"], "cond": b["cond"], "ks": set()})["ks"].add(b["k"])

    def geom(b):
        return np.array([(b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2]), max(1e-4, (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))

    per_trial = []
    for sid, t in sorted(trials.items()):
        ks, shifts, resizes = sorted(t["ks"]), [], []
        for k0, k1 in zip(ks, ks[1:]):
            bs = [box.get((sid, k, s)) for k in (k0, k1) for s in ("view", "real")]
            if not all(b and b["visible"] for b in bs):
                continue
            (cv0, av0), (cr0, ar0), (cv1, av1), (cr1, ar1) = map(geom, bs)
            shifts.append(100 * float(np.linalg.norm((cv1 - cr1) - (cv0 - cr0))))
            resizes.append(float(abs(np.log(av1 / ar1) - np.log(av0 / ar0))))
        per_trial.append({"stim_id": sid, "task": t["task"], "cond": t["cond"], "pairs": len(shifts),
                          "max_shift": max(shifts, default=None), "max_resize": max(resizes, default=None)})
    by_cond = {}
    for c in "nbd":
        s = [p["max_shift"] for p in per_trial if p["cond"] == c and p["max_shift"] is not None]
        z = [p["max_resize"] for p in per_trial if p["cond"] == c and p["max_resize"] is not None]
        by_cond[c] = {"trials": len(s), "max_shift_median": float(np.median(s)) if s else None,
                      "max_shift_q90": float(np.quantile(s, .9)) if s else None,
                      "max_resize_median": float(np.median(z)) if z else None}
    return {"by_cond": by_cond, "trials": per_trial}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-model", default="gpt-6.1-sol")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "boxes.jsonl"
    done = {(b["stim_id"], b["k"], b["source"]) for b in map(json.loads, path.open())} if path.exists() else set()
    todo = [j for j in jobs() if (j["stim_id"], j["k"], j["source"]) not in done]
    print(f"{len(todo)} boxes to find ({len(done)} done)")
    client, sem, lock = AsyncOpenAI(), asyncio.Semaphore(args.concurrency), asyncio.Lock()

    async def one(j):
        img = (np.asarray(Image.open(ROOT / j["path"]).convert("RGB")) if j["source"] == "view"
               else mp4_frame(j["cell"], j["t"]))
        async with sem:
            try:
                b = await get_box(client, img, NEUTRAL[j["task"]], args.box_model)
            except Exception as e:
                print(f"FAILED {j['stim_id']}/{j['k']}/{j['source']}: {e!r}"[:200])
                return
        async with lock:
            with path.open("a") as f:
                f.write(json.dumps({k: j[k] for k in ("stim_id", "task", "cond", "k", "t", "source")}
                                   | {"box_model": args.box_model, **b}) + "\n")

    await asyncio.gather(*(one(j) for j in todo))
    s = summarize(list(map(json.loads, path.open())))
    (OUT / "truth.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s["by_cond"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
