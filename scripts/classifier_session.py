"""Remove the session confound from the classifier result.

Unaltered trials were mostly recorded on 2026-10-24 and doctored ones on 10-25..27, so a classifier on the robot's
snapshots could be reading the session (lighting, layout, rig) instead of the edit. No unaltered/doctored trials share
a session (the closest share a rig and UTC date but are ~18 h apart). Fix: every trial has the real overhead mp4 frame
recorded at the same instant as its first snapshot (same session, lighting and layout), so classify the robot's top_cam
snapshot features *minus* that trial's mp4 frame features. Session cancels within the trial; what remains is the
pipeline difference: site WebP vs h264 for unaltered trials, edit + WebP vs h264 for doctored trials.

Comparisons (trial-grouped CV, leave-one-task-out, shuffled-label null; reuses data/classifier cache):
  paired_main          unaltered vs doctored on (snapshot - mp4) differences
  raw_main_top         unaltered vs doctored on the raw snapshot (the earlier result, for reference)
  session_null_raw     unaltered trials only, 10-24 vs 10-25 (cord / mix / drink), raw snapshot: is session visible?
  session_null_paired  the same split on (snapshot - mp4): does pairing remove it? (should be ~0.5)
  dsession_null_*      doctored trials only, 10-25 vs 10-26, raw and paired (same check on the doctored side)
  matched_view         unaltered vs doctored restricted to same rig + UTC date (cord rig-6, mix rig-2, drink rig-5
                       on 10-25), raw snapshot; matched_mp4 is the session-only baseline on that subset

Usage: uv run python scripts/classifier_session.py [--perms 200]   (run scripts/classifier.py first)
Writes results/e2_classifier/session.md and session.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import CACHE, OUT, RAW, evaluate  # noqa: E402

MATCHED = {("cord", "omen_rig-6"), ("mix", "omen_rig-2"), ("drink", "omen-2_rig-5")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    z = np.load(CACHE / "first_snapshot_features.npz", allow_pickle=True)
    meta = list(z["meta"])
    rec = json.loads((RAW / "explore" / "records.json").read_text())
    date = np.array([rec[m["id"]]["date"][:10] for m in meta])
    rig = np.array([rec[m["id"]]["log"].split("/")[-1].split("_robospoof")[0] for m in meta])
    cond = np.array([m["cond"] for m in meta])
    task = np.array([m["task"] for m in meta])
    idx_all = np.arange(len(meta))
    matched = np.array([(t, r) in MATCHED for t, r in zip(task, rig)]) & (date == "2026-10-25")

    results = {}
    for feat in ("forensic", "resnet"):
        view, mp4 = z[f"view_{feat}"][:, 0], z[f"mp4_{feat}"]  # top_cam snapshot and its same-instant mp4 frame
        paired = view - mp4
        comps = {
            "paired_main": (np.isin(cond, ["n", "d"]), cond == "d", paired),
            "raw_main_top": (np.isin(cond, ["n", "d"]), cond == "d", view),
            "session_null_raw": ((cond == "n") & np.isin(task, ["cord", "mix", "drink"]) & np.isin(date, ["2026-10-24", "2026-10-25"]),
                                 date == "2026-10-25", view),
            "session_null_paired": ((cond == "n") & np.isin(task, ["cord", "mix", "drink"]) & np.isin(date, ["2026-10-24", "2026-10-25"]),
                                    date == "2026-10-25", paired),
            "dsession_null_raw": ((cond == "d") & np.isin(date, ["2026-10-25", "2026-10-26"]), date == "2026-10-26", view),
            "dsession_null_paired": ((cond == "d") & np.isin(date, ["2026-10-25", "2026-10-26"]), date == "2026-10-26", paired),
            "matched_view": (matched & np.isin(cond, ["n", "d"]), cond == "d", view),
            "matched_mp4": (matched & np.isin(cond, ["n", "d"]), cond == "d", mp4),
        }
        for name, (keep, pos, X) in comps.items():
            idx = idx_all[keep]
            r = evaluate(X[idx], pos[idx].astype(int), idx, task[idx], args.perms, rng)
            results[f"{name}/{feat}"] = r
            print(f"{name:20} {feat:9} n={r['n_trials']:3} ({r['n_pos']:3}) AUROC cv {r['auroc_cv']:.3f}  "
                  f"LOTO {r['auroc_loto_mean'] if r['auroc_loto_mean'] is not None else float('nan'):.3f}  "
                  f"null p95 {r.get('null_p95', float('nan')):.3f}")

    (OUT / "session.json").write_text(json.dumps(results, indent=2))
    lines = ["# Session-confound control\n",
             "Trial-level AUROC (5-fold CV grouped by trial; leave-one-task-out mean where >1 task has both classes; "
             "shuffled-label null 95th pct). 'paired' = robot snapshot features minus same-instant mp4 frame features.\n",
             "| comparison | features | trials (pos) | AUROC CV | AUROC LOTO | null p95 | p |", "|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        name, feat = k.split("/")
        loto = f"{r['auroc_loto_mean']:.3f}" if r["auroc_loto_mean"] is not None else "–"
        lines.append(f"| {name} | {feat} | {r['n_trials']} ({r['n_pos']}) | {r['auroc_cv']:.3f} | {loto} | "
                     f"{r.get('null_p95', float('nan')):.3f} | {r.get('p_perm', float('nan')):.3g} |")
    (OUT / "session.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
