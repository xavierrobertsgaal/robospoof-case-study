# Session-confound control

Trial-level AUROC (5-fold CV grouped by trial; leave-one-task-out mean where >1 task has both classes; shuffled-label null 95th pct). 'paired' = robot snapshot features minus same-instant mp4 frame features.

| comparison | features | trials (pos) | AUROC CV | AUROC LOTO | null p95 | p |
|---|---|---|---|---|---|---|
| paired_main | forensic | 600 (300) | 1.000 | 1.000 | 0.540 | 0.00498 |
| raw_main_top | forensic | 600 (300) | 1.000 | 1.000 | 0.540 | 0.00498 |
| session_null_raw | forensic | 180 (50) | 0.799 | 0.354 | 0.609 | 0.00498 |
| session_null_paired | forensic | 180 (50) | 0.700 | 0.307 | 0.605 | 0.00995 |
| dsession_null_raw | forensic | 263 (138) | 0.739 | 0.632 | 0.566 | 0.00498 |
| dsession_null_paired | forensic | 263 (138) | 0.802 | 0.717 | 0.566 | 0.00498 |
| matched_view | forensic | 126 (77) | 1.000 | 1.000 | 0.586 | 0.00498 |
| matched_mp4 | forensic | 126 (77) | 0.877 | 0.554 | 0.596 | 0.00498 |
| paired_main | resnet | 600 (300) | 1.000 | 1.000 | 0.548 | 0.00498 |
| raw_main_top | resnet | 600 (300) | 1.000 | 1.000 | 0.548 | 0.00498 |
| session_null_raw | resnet | 180 (50) | 0.929 | 0.333 | 0.598 | 0.00498 |
| session_null_paired | resnet | 180 (50) | 0.838 | 0.539 | 0.600 | 0.00498 |
| dsession_null_raw | resnet | 263 (138) | 0.858 | 0.627 | 0.565 | 0.00498 |
| dsession_null_paired | resnet | 263 (138) | 0.906 | 0.456 | 0.565 | 0.00498 |
| matched_view | resnet | 126 (77) | 1.000 | 1.000 | 0.615 | 0.00498 |
| matched_mp4 | resnet | 126 (77) | 0.914 | 0.303 | 0.599 | 0.00498 |
