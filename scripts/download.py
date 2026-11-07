"""Download every video and run artifact published on the RoboSpoof project page.

Sources (discovered from https://robocurve.org/robospoof/ and its robospoof.*.js):
  explore/runs.json, explore/records.json   index of all 900 trials
  explore/{cell}.mp4                         real overhead recording, 12 trials concatenated (t0/dur in runs.json)
  explore/{cell}.json                        per-call camera images the controlling model actually received
  view/report/media/hero-stab-{wide,tall}.*  real-vs-doctored side-by-side hero cut
  view/report/img/*.jpg                      initial-scene stills shown in the paper/page

Usage: uv run python scripts/download.py [--out data/raw]
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://runs.robocurve.org/robospoof-2026-10-29"
TASKS = ["cord", "doll", "drink", "mix", "toaster"]
CONDS = {"normal": "n", "blurred": "b", "doctored": "d"}


def fetch(url: str, dest: Path) -> tuple[str, int]:
    if dest.exists() and dest.stat().st_size > 0:
        return url, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return url, len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = ap.parse_args()
    out: Path = args.out

    for name in ("runs.json", "records.json"):
        fetch(f"{BASE}/explore/{name}", out / "explore" / name)
    runs = json.loads((out / "explore" / "runs.json").read_text())
    cells = sorted({r["cell"] for r in runs})

    jobs = []
    for c in cells:
        for ext in ("mp4", "json"):
            jobs.append((f"{BASE}/explore/{c}.{ext}", out / "explore" / f"{c}.{ext}"))
    for cut in ("wide", "tall"):
        for ext in ("mp4", "jpg"):
            jobs.append((f"{BASE}/view/report/media/hero-stab-{cut}.{ext}", out / "media" / f"hero-stab-{cut}.{ext}"))
    for t in TASKS:
        jobs.append((f"{BASE}/view/report/img/still-{t}.jpg", out / "img" / f"still-{t}.jpg"))
        for cond in CONDS:
            jobs.append((f"{BASE}/view/report/img/start-{t}-{cond}.jpg", out / "img" / f"start-{t}-{cond}.jpg"))

    total, failed = 0, []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(fetch, u, d): u for u, d in jobs}
        for f in futs:
            try:
                _, n = f.result()
                total += n
            except Exception as e:  # report and keep going; rerun resumes
                failed.append((futs[f], repr(e)))
    print(f"{len(jobs) - len(failed)}/{len(jobs)} files, {total / 1e6:.1f} MB in {out}")
    for u, e in failed:
        print("FAILED", u, e)


if __name__ == "__main__":
    main()
