# FreqNet Deepfake Detection + Frequency-Consistency Regularization

**Team members**
- Pitta Abhiram — 230104070
- Pankaj Chauhan — 230104069

This project reproduces the official FreqNet deepfake detector and studies a proposed extension, **Frequency-Consistency Regularization (FCR)**. FreqNet learns in the frequency domain using high-frequency representations and frequency convolution modules. FCR is an experimental add-on that encourages stable predictions between an image and a mildly transformed view.

**Research status:** source code, training/evaluation scripts, tests, and the experiment protocol are in the repository. The model has **not** been trained here because the dataset and a training GPU have not been provided to this session. No project-specific performance result is claimed. Fill report metrics only after running the experiments.

## Project files

- `networks/freqnet.py` — FreqNet architecture adapted from the [official repository](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection).
- `networks/trainer.py` — baseline training path plus optional FCR loss.
- `train.py` — seeded training loop, validation, early stopping, TensorBoard logs, and best/last checkpoints.
- `evaluate.py` — held-out evaluation with accuracy, AP, AUROC, F1, precision, recall, and confusion matrix.
- `robustness_eval.py` — evaluates clean images, JPEG qualities 95/75/50, and resize down/up.
- `DATASET_AND_EXPERIMENTS.md` — dataset folder layout, smoke-test checklist, and experiment matrix.
- `REPORT_TEMPLATE.tex` — one-page report draft with project metrics left pending.
- `tests/test_fcr.py` — unit tests; GitHub Actions runs syntax checks and tests.

## 1. Environment setup

Python 3.10 or 3.11 is recommended. Use a CUDA-enabled PyTorch build compatible with your GPU if available.

```bash
git clone https://github.com/pankajchauhan1029/DA_6042_sem07.git
cd DA_6042_sem07
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CPU-only testing, pass `--gpu_ids -1`. The original FreqNet architecture is computationally heavy; use a supported CUDA GPU for practical training.

## 2. Dataset

Download and prepare the dataset following the [official FreqNet repository](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) and its linked [CNNDetection dataset](https://github.com/peterwang512/CNNDetection). The expected directory structure and class mapping are documented in [DATASET_AND_EXPERIMENTS.md](DATASET_AND_EXPERIMENTS.md).

For the four-class experiment, the root passed to `--dataroot` must have `train/` and `val/`; each split should contain class folders (for example `car, cat, chair, horse`) and within each class, `0_real/` and `1_fake/`. The ImageFolder class ordering must map `0_real` to label 0 and `1_fake` to label 1. Confirm the actual downloaded directory names before running.

## 3. Train baseline

First run a short smoke test (one epoch and a small batch) and verify data loading/checkpoint output. Then run the full baseline:

```bash
python train.py --name freqnet-baseline --dataroot /path/to/ForenSynths --classes car,cat,chair,horse --batch_size 32 --lr 0.001 --niter 85 --seed 1029 --gpu_ids 0
```

On CPU, replace `--gpu_ids 0` with `--gpu_ids -1` and use a small batch for a smoke test.

The run folder is created under `checkpoints/`. It contains the run configuration, `model_epoch_best.pth`, `model_epoch_last.pth`, TensorBoard logs, and a training summary. Best checkpoint selection uses validation average precision (AP), not test performance.

## 4. Train FCR

Run the same configuration and seed, changing only the run name and FCR flags:

```bash
python train.py --name freqnet-fcr-lambda005 --dataroot /path/to/ForenSynths --classes car,cat,chair,horse --batch_size 32 --lr 0.001 --niter 85 --seed 1029 --gpu_ids 0 --fcr_enabled --fcr_lambda 0.05
```

The objective is

`L = 0.5 * [BCE(z(x), y) + BCE(z(T(x)), y)] + lambda * KL(Bernoulli(sigmoid(z(x))) || Bernoulli(sigmoid(z(T(x)))))`

where `T` is a mild downsample/upscale or 3x3 average blur. This is a hypothesis: it may help, do nothing, or hurt. Compare the baseline against the two-view `lambda=0` ablation and `lambda=0.01, 0.05, 0.1`. Keep splits and training budgets identical.

## 5. Evaluate

For a held-out binary real/fake directory with `0_real/` and `1_fake/` subfolders:

```bash
python evaluate.py --model_path checkpoints/EXPERIMENT_FOLDER/model_epoch_best.pth --dataroot /path/to/held-out-set --gpu_ids 0 --output_json evaluation_results.json
python robustness_eval.py --model_path checkpoints/EXPERIMENT_FOLDER/model_epoch_best.pth --dataroot /path/to/held-out-set --gpu_ids 0 --output_json robustness_results.json
```

For a CPU run, use `--gpu_ids -1`. If the test root has class subdirectories, pass `--classes class1,class2` as appropriate. Use only held-out data for final metrics; do not tune the model or lambda on the test set.

## 6. Reproducibility and reporting

- Use seeds 1029, 2027, and 6042 when resources permit; report mean ± standard deviation.
- Compare baseline and FCR on identical splits and the same held-out examples.
- Report accuracy, AP, AUROC, F1, confusion matrix, parameter count, and latency.
- Report robustness separately under JPEG and resize perturbations.
- Keep datasets, checkpoints, and raw images out of Git (ignored by `.gitignore`).
- Do not report the paper's published scores as your own reproduction results.

The paper reports broad cross-generator evaluation; use its official paper and repository for method details:
- Tan et al., *Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning*, AAAI 2024. [Paper](https://arxiv.org/abs/2403.07240) · [Official code](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)
