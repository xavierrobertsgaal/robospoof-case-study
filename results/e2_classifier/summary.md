# Small-classifier separability of first snapshots

Trial-level AUROC (5-fold CV grouped by trial; leave-one-task-out mean; shuffled-label null mean / 95th pct).

| comparison | features | trials (pos) | AUROC CV | balanced acc | AUROC LOTO mean | null mean | null p95 | p |
|---|---|---|---|---|---|---|---|---|
| main | forensic | 600 (300) | 1.000 | 1.000 | 1.000 | 0.480 | 0.532 | 0.00498 |
| main_top | forensic | 600 (300) | 1.000 | 1.000 | 1.000 | 0.484 | 0.540 | 0.00498 |
| session_top | forensic | 600 (300) | 0.859 | 0.777 | 0.657 | 0.483 | 0.532 | 0.00498 |
| pos_blur | forensic | 600 (300) | 1.000 | 1.000 | 1.000 | nan | nan | nan |
| main | resnet | 600 (300) | 1.000 | 1.000 | 1.000 | 0.489 | 0.546 | 0.00498 |
| main_top | resnet | 600 (300) | 1.000 | 1.000 | 1.000 | 0.496 | 0.538 | 0.00498 |
| session_top | resnet | 600 (300) | 0.984 | 0.937 | 0.601 | 0.496 | 0.549 | 0.00498 |
| pos_blur | resnet | 600 (300) | 1.000 | 1.000 | 1.000 | nan | nan | nan |

LOTO per task: main/forensic: doll 1.00, cord 1.00, toaster 1.00, mix 1.00, drink 1.00; main_top/forensic: doll 1.00, cord 1.00, toaster 1.00, mix 1.00, drink 1.00; session_top/forensic: doll 0.60, cord 0.79, toaster 0.64, mix 0.75, drink 0.51; pos_blur/forensic: doll 1.00, cord 1.00, toaster 1.00, mix 1.00, drink 1.00; main/resnet: doll 1.00, cord 1.00, toaster 1.00, mix 1.00, drink 1.00; main_top/resnet: doll 1.00, cord 1.00, toaster 1.00, mix 1.00, drink 1.00; session_top/resnet: doll 0.57, cord 0.69, toaster 0.49, mix 0.61, drink 0.64; pos_blur/resnet: doll 1.00, cord 1.00, toaster 1.00, mix 1.00, drink 1.00
