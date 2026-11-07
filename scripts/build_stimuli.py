"""Build real/blurred/doctored stimuli from the downloaded RoboSpoof data, in the format the robot received them.

In the robot runs (inspect-robots, see data/raw/logs) the controlling model never saw video: each call's observation
carried one snapshot, three separate 224x224 images (camera 'top_cam', 'left_cam', 'right_cam'), and only the two most
recent snapshots stayed in context (image_horizon=2). The site stores each snapshot as a 672x224 webp composite in
explore/{cell}.json; here it is split back into its three camera images (lossless PNG crops of the decoded webp).

Sets
  first  the initial snapshot of each trial: exactly the input the controlling model refused or attempted on.
  seq    every snapshot of the trial in order (more than the controlling model ever had in context at once).
  hero   the published hero cut split into its real (left) and doctored (right) halves at --hero-fps: true video,
         higher resolution, overhead camera only.
By default trials are all of --source-model's (gpt-6-astra: 12 per task x condition), so each stimulus pairs with that
model's own behaviour in the robot setting.

Writes data/stimuli/<set>/<stim_id>/s<k>_<cam>.png and data/stimuli/manifest.jsonl, where each stimulus has
"snapshots": a list of snapshots, each a list of image paths (one per camera; hero frames are single images).
Usage: uv run python scripts/build_stimuli.py [--source-model gpt-6-astra | --source-model all --per-cell 12]
"""

import argparse
import base64
import io
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "stimuli"
TRUTH = {"n": "real", "b": "real_blurred", "d": "doctored"}
CAMS = ["top_cam", "left_cam", "right_cam"]  # left-to-right order of the composite, as in the observation messages


def round_robin(trials: list[dict], k: int, rng: random.Random) -> list[dict]:
    """Pick k trials spreading evenly over the source models that generated them."""
    by_model = defaultdict(list)
    for t in trials:
        by_model[t["model"]].append(t)
    for v in by_model.values():
        rng.shuffle(v)
    models = sorted(by_model)
    rng.shuffle(models)
    picked = []
    while len(picked) < k and any(by_model.values()):
        for m in models:
            if by_model[m] and len(picked) < k:
                picked.append(by_model[m].pop())
    return picked


def write_snapshots(dest: Path, imgs: list[str]) -> list[list[str]]:
    dest.mkdir(parents=True, exist_ok=True)
    snaps = []
    for k, b64 in enumerate(imgs):
        comp = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        w = comp.width // len(CAMS)
        paths = []
        for i, cam in enumerate(CAMS):
            p = dest / f"s{k:02d}_{cam}.png"
            comp.crop((i * w, 0, (i + 1) * w, comp.height)).save(p)
            paths.append(str(p.relative_to(ROOT)))
        snaps.append(paths)
    return snaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="gpt-6-astra", help="whose robot trials to use; 'all' samples across models")
    ap.add_argument("--per-cell", type=int, help="trials per (task, condition); default all")
    ap.add_argument("--hero-fps", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    runs = json.loads((RAW / "explore" / "runs.json").read_text())
    records = json.loads((RAW / "explore" / "records.json").read_text())
    cell_frames = {c: json.loads((RAW / "explore" / f"{c}.json").read_text()) for c in {r["cell"] for r in runs}}
    for r in runs:
        r["imgs"] = [c["img"] for c in cell_frames[r["cell"]].get(r["id"], []) if c.get("img")]
    instruction = {r["task"]: records[r["id"]]["task"] for r in runs}

    if OUT.exists():
        shutil.rmtree(OUT)
    manifest = []
    by_cell = defaultdict(list)
    for r in runs:
        if r["imgs"] and args.source_model in ("all", r["model"]):
            by_cell[(r["task"], r["cond"])].append(r)

    for (task, cond), trials in sorted(by_cell.items()):
        for r in round_robin(trials, args.per_cell or len(trials), rng):
            tid = r["id"].split("_")[-1]
            meta = {"task": task, "cond": cond, "truth": TRUTH[cond], "instruction": instruction[task],
                    "source_run": r["id"], "source_model": r["model"], "source_outcome": r["o"]}
            snaps = write_snapshots(OUT / "trial" / f"{task}-{cond}-{tid}", r["imgs"])
            manifest.append({"stim_id": f"first-{task}-{cond}-{tid}", "set": "first", **meta, "snapshots": snaps[:1]})
            manifest.append({"stim_id": f"seq-{task}-{cond}-{tid}", "set": "seq", **meta, "snapshots": snaps})

    hero = RAW / "media" / "hero-stab-wide.mp4"
    for side, crop, cond in (("left", "iw/2:ih:0:0", "n"), ("right", "iw/2:ih:iw/2:0", "d")):
        sid = f"hero-doll-{cond}"
        dest = OUT / "hero" / sid
        dest.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(hero), "-vf", f"crop={crop},fps={args.hero_fps}",
                        "-q:v", "2", str(dest / "frame_%02d.jpg")], check=True)
        frames = sorted(str(p.relative_to(ROOT)) for p in dest.glob("frame_*.jpg"))
        manifest.append({"stim_id": sid, "set": "hero", "task": "doll", "cond": cond, "truth": TRUTH[cond],
                         "instruction": instruction["doll"], "source_run": f"hero-stab-wide:{side}",
                         "source_model": None, "source_outcome": None, "snapshots": [[f] for f in frames]})

    with (OUT / "manifest.jsonl").open("w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    counts, lens = defaultdict(int), defaultdict(list)
    for m in manifest:
        counts[(m["set"], m["cond"])] += 1
        lens[(m["set"], m["cond"])].append(len(m["snapshots"]))
    print(f"{len(manifest)} stimuli:", dict(sorted(counts.items())))
    print("snapshots per stimulus (min/median/max):",
          {k: (min(v), sorted(v)[len(v) // 2], max(v)) for k, v in sorted(lens.items())})


if __name__ == "__main__":
    main()
