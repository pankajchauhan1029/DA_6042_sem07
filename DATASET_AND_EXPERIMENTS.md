# Dataset layout and run checklist

The scripts follow the upstream FreqNet / CNNDetection folder conventions. The root passed to `--dataroot` must contain `train/` and `val/` subdirectories. For the recommended four-class experiment, each split contains class folders, and each class folder contains real/fake folders:

```text
ForenSynths/
  train/
    car/
      0_real/
      1_fake/
    cat/
      0_real/
      1_fake/
    chair/
      0_real/
      1_fake/
    horse/
      0_real/
      1_fake/
  val/
    car/
      0_real/
      1_fake/
    ...
  test/
    progan/
      0_real/
      1_fake/
    stylegan/
      0_real/
      1_fake/
    ...
```

Folder names can vary by downloaded dataset. Before a long run, inspect the real folder names and update `--classes` to match. The ImageFolder label order must be `0_real` = 0 and `1_fake` = 1.

## Minimum smoke test

Run on a small dataset subset first, with a small batch and one epoch. Confirm that:
- the training loader has samples from both classes;
- the validation loader has both classes;
- loss is finite and decreases or at least changes;
- checkpoints are created under `checkpoints/<run-name>/`;
- evaluation can load the saved checkpoint.

## Final experiment table

Use identical splits, training budget, and seed per paired baseline/FCR run. Prefer seeds 1029, 2027, and 6042. Keep test data out of model selection.

| Variant | Flags |
|---|---|
| Baseline | no `--fcr_enabled` |
| Two-view ablation | `--fcr_enabled --fcr_lambda 0` |
| FCR low | `--fcr_enabled --fcr_lambda 0.01` |
| FCR medium | `--fcr_enabled --fcr_lambda 0.05` |
| FCR high | `--fcr_enabled --fcr_lambda 0.1` |

Do not put dataset images or trained weights into Git; the repository .gitignore excludes common data and checkpoint paths.
