# FreqNet + Frequency-Consistency Regularization (FCR)

## Status
The repository is initialized and the FCR trainer patch is proposed. Training has not yet been run; no project performance numbers are available.

## Proposed contribution
Keep the original FreqNet architecture and encourage prediction stability between an image and a mildly perturbed view. This is a research hypothesis, not a guaranteed improvement.

For logits z(x), binary label y, and a label-preserving transform T:

L = 0.5 * [BCEWithLogits(z(x), y) + BCEWithLogits(z(T(x)), y)] + lambda * KL(Bernoulli(sigmoid(z(x))) || Bernoulli(sigmoid(z(T(x))))).

The training transform samples either mild downsample/upscale or 3x3 average blur. JPEG is not simulated by this transform; evaluate JPEG robustness with actual encode/decode at test time.

## Experiment matrix
1. Original FreqNet baseline (FCR disabled).
2. Two-view supervised ablation (FCR enabled, lambda=0).
3. FCR lambda=0.01.
4. FCR lambda=0.05.
5. FCR lambda=0.1.
6. Robustness evaluation on the same held-out images under JPEG quality 95, 75, and 50, plus resize down/up.

## Valid protocol
- Fix and record seeds; use identical train/validation/test splits and the same test examples for every variant.
- Use validation only for checkpoint selection and lambda tuning. Do not tune on test data.
- Prefer three seeds and report mean ± standard deviation. If resources allow only one run, state that limitation.
- Report accuracy, average precision (AP), AUROC, F1, confusion matrix, parameter count, and inference latency where feasible.
- Keep source/generator identities disjoint across train and held-out tests when supported by the dataset. Avoid leakage from near-duplicates or related outputs.
- Save command lines, seed, commit hash, environment/package versions, GPU, checkpoint paths, and raw predictions.

## Important integration note
The upstream FreqNet implementation has architecture-specific CUDA allocations in model initialization. Test it in the intended GPU environment before assuming CPU compatibility. This patch is based on the upstream trainer API, but has not yet been run end-to-end against a training dataset.
