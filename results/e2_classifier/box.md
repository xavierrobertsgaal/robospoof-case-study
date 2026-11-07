# Box-only classifier (top_cam, pixels inside the target object's box)

| comparison | features | trials (pos) | AUROC CV | AUROC LOTO mean | LOTO per task | null p95 |
|---|---|---|---|---|---|---|
| box_main | forensic | 600 (300) | 0.985 | 0.917 | doll 0.78, cord 1.00, toaster 1.00, mix 0.98, drink 0.82 | 0.542 |
| box_session | forensic | 600 (300) | 0.892 | 0.561 | doll 0.58, cord 0.47, toaster 0.53, mix 0.53, drink 0.69 | 0.538 |
| box_main | resnet | 600 (300) | 0.999 | 0.733 | doll 1.00, cord 0.92, toaster 0.05, mix 0.76, drink 0.93 | 0.544 |
| box_session | resnet | 600 (300) | 0.992 | 0.871 | doll 0.90, cord 0.70, toaster 0.94, mix 0.90, drink 0.92 | 0.535 |

**Session-identical transfer** (trained on robot traces minus the 20 ablation bases; tested on each base vs our edits of it, cropped at the edit box; bases whose top_cam box was not visible are dropped)

| features | method | AUROC vs real | flagged (P>0.5) | real flagged |
|---|---|---|---|---|
| forensic | global-sunburst | 0.710 | 5/20 | 1/20 |
| forensic | global-flare | 0.818 | 5/20 | 1/20 |
| forensic | global-mini | 0.568 | 5/20 | 1/20 |
| forensic | local-blend | 0.652 | 5/20 | 1/20 |
| forensic | crude-paste | 0.650 | 2/20 | 1/20 |
| resnet | global-sunburst | 0.997 | 18/20 | 0/20 |
| resnet | global-flare | 0.995 | 18/20 | 0/20 |
| resnet | global-mini | 1.000 | 20/20 | 0/20 |
| resnet | local-blend | 0.997 | 19/20 | 0/20 |
| resnet | crude-paste | 1.000 | 20/20 | 0/20 |
