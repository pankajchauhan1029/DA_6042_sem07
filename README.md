# DA_6042_sem07 — FreqNet Deepfake Detection

Team members:
- Pitta Abhiram — 230104070
- Pankaj Chauhan — 230104069

This repository contains the project proposal and a patch for adding Frequency-Consistency Regularization (FCR) to the FreqNet deepfake detector.

## Current status

- [x] Initialize repository and add experiment plan
- [x] Prepare proposed FCR trainer patch and CLI option additions
- [ ] Integrate with the full upstream FreqNet source and verify compatibility
- [ ] Reproduce the original baseline
- [ ] Run controlled ablations and robustness tests
- [ ] Replace report placeholders with measured results

**No training has been run and no project performance numbers are claimed.** Values quoted from the paper are reference results only.

## Files

- `patches/trainer_fcr.py`: proposed FCR trainer patch based on the upstream trainer API.
- `options/train_options_additions.txt`: two command-line options to add to upstream `TrainOptions`.
- `EXPERIMENT_PLAN.md`: controlled comparison, robustness protocol, and reporting requirements.
- `REPORT_TEMPLATE.tex`: one-page LaTeX report draft with unfilled project result fields.

## Proposed objective

For image x, label y, transformation T, and model logit z:

`L = 0.5 * [BCE(z(x), y) + BCE(z(T(x)), y)] + lambda * KL(Bernoulli(sigmoid(z(x))) || Bernoulli(sigmoid(z(T(x)))))`

The proposed transformation randomly uses mild downsample/upscale or a 3x3 average blur. JPEG robustness must be tested with actual JPEG encode/decode at evaluation time; the training helper does not simulate JPEG.

## Integration

1. Obtain the full upstream implementation from https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection and install its requirements.
2. Back up the upstream `networks/trainer.py`, then use `patches/trainer_fcr.py` as the proposed replacement.
3. Add the arguments in `options/train_options_additions.txt` inside `TrainOptions.initialize()`.
4. Run baseline without `--fcr_enabled`, then compare FCR with `--fcr_enabled --fcr_lambda 0.01`, `0.05`, and `0.1`. Also run `--fcr_enabled --fcr_lambda 0` as a two-view supervised ablation.
5. Use separate experiment names/checkpoint folders and keep data splits, training budget, and all other settings identical.

Example command shape (adapt to the upstream options and dataset paths):

```bash
python train.py --name fcr-lambda005 --dataroot /path/to/CNNDetection --classes car,cat,chair,horse --batch_size 32 --delr_freq 10 --lr 0.001 --niter 85 --fcr_enabled --fcr_lambda 0.05
```

## Evaluation rules

- Tune only on validation data; do not select checkpoints or lambda on the test set.
- Prefer three seeds and report mean ± standard deviation.
- Report accuracy, average precision, AUROC, F1, confusion matrix, parameter count, and latency where feasible.
- Evaluate the same held-out test images under JPEG qualities 95/75/50 and resize down/up.
- Keep generator/source identities disjoint between training and held-out tests when the dataset supports it.
- Record command lines, seed, commit, package versions, GPU, checkpoint path, and raw predictions.
- The upstream model code has CUDA allocations in its constructor; validate it in the intended GPU environment. The patch is not yet end-to-end tested.

FCR is a hypothesis and may improve, leave unchanged, or reduce performance. Only measured results from this project should be added to the report.
