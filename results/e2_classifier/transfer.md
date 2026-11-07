# Session-identical test: classifier trained on robot traces, tested on edits of held-out robot snapshots

| features | method | AUROC vs real | flagged (P>0.5) | real bases flagged |
|---|---|---|---|---|
| forensic | global-sunburst | 0.997 | 15/20 | 0/20 |
| forensic | global-flare | 0.995 | 9/20 | 0/20 |
| forensic | global-mini | 0.580 | 1/20 | 0/20 |
| forensic | local-blend | 0.608 | 0/20 | 0/20 |
| forensic | crude-paste | 0.617 | 0/20 | 0/20 |
| resnet | global-sunburst | 0.907 | 0/20 | 0/20 |
| resnet | global-flare | 0.962 | 1/20 | 0/20 |
| resnet | global-mini | 0.995 | 14/20 | 0/20 |
| resnet | local-blend | 0.855 | 1/20 | 0/20 |
| resnet | crude-paste | 0.900 | 0/20 | 0/20 |
