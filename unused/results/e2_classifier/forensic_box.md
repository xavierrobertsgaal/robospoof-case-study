# Forensics-only box classifier (64x64 residual patch at the object, ring-referenced, no pixels/colour/shape)

| comparison | trials (pos) | AUROC CV | AUROC LOTO mean | LOTO per task | null p95 | p |
|---|---|---|---|---|---|---|
| fbox_main | 600 (300) | 0.996 | 0.834 | doll 0.63, cord 0.99, toaster 1.00, mix 0.57, drink 0.97 | 0.544 | 0.00498 |
| fbox_session | 600 (300) | 0.882 | 0.550 | doll 0.72, cord 0.61, toaster 0.65, mix 0.46, drink 0.31 | 0.546 | 0.00498 |

**Session-identical transfer** (trained on robot traces minus the ablation bases; each base vs our edits of it, same patch location)

| method | AUROC vs real | flagged (P>0.5) | real flagged |
|---|---|---|---|
| global-sunburst | 0.748 | 3/20 | 0/20 |
| global-flare | 0.802 | 4/20 | 0/20 |
| global-mini | 0.592 | 4/20 | 0/20 |
| local-blend | 0.785 | 5/20 | 0/20 |
| crude-paste | 0.772 | 4/20 | 0/20 |
