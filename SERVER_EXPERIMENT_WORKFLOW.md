# Server Experiment Workflow

This project uses GitHub as the source of truth and the Linux server as the actual training environment.

## Branch Strategy

- `paper-baseline`
  - Reproduced DU-GAN paper baseline.
  - Use this as the checkpoint source for all later comparisons.
- `noise-d-v1`
  - Original experimental branch with NoiseD / MoE code.
- `noise-d-v2`
  - Recommended clean branch for fair `NoiseD-only` experiments.
  - Keep baseline DU-GAN losses intact and add only the NoiseD terms.

## Server-First Rules

1. Never edit code directly on the server without committing it back to GitHub.
2. Before each run, hard reset to the exact branch tip:

```bash
git fetch origin
git checkout <branch>
git reset --hard origin/<branch>
```

3. Keep datasets, logs, checkpoints, and outputs out of Git.
4. Every run should save `run_metadata.json` under its output directory.

## Dataset Layout

The server should provide these files locally:

```text
dataset/cmayo/train_64.npy
dataset/cmayo/test_512.npy
dataset/cmayo/train_id.csv
dataset/cmayo/test_id.csv
```

Only the CSV files should stay in Git. The NPY files are local server assets.

## Recommended Commands

### Baseline

```bash
bash train_dugan.sh
```

### NoiseD-only scratch

```bash
RUN_NAME=noise_d_clean \
MOE_PHASE=1 \
LAMBDA_NOISE_FIXED=0.1 \
LAMBDA_NOISE_REG=0.3 \
bash train_noise_d.sh
```

### NoiseD-only short smoke test

```bash
RUN_NAME=noise_d_smoke \
MAX_ITER=1 \
SAVE_FREQ=1 \
BATCH_SIZE=4 \
NUM_WORKERS=2 \
MOE_PHASE=1 \
LAMBDA_NOISE_FIXED=0.1 \
LAMBDA_NOISE_REG=0.3 \
bash train_noise_d.sh
```

## Warm-Start from `paper-baseline`

1. Copy baseline checkpoints into the new run folder:

```bash
mkdir -p output/DUGAN_MoE_noise_d_ft2k/save_models
cp output/DUGAN_official/save_models/*-100000 output/DUGAN_MoE_noise_d_ft2k/save_models/
```

2. Continue with a short run:

```bash
RUN_NAME=noise_d_ft2k \
RESUME_ITER=100000 \
MAX_ITER=102000 \
SAVE_FREQ=2000 \
MOE_PHASE=1 \
LAMBDA_NOISE_FIXED=0.05 \
LAMBDA_NOISE_REG=0.3 \
bash train_noise_d.sh
```

## Low-Cost Validation Order

1. `paper-baseline` metrics
2. `NoiseD-only` smoke test
3. `NoiseD-only` short scratch run
4. `NoiseD-only` 2k warm-start
5. Only if the above is stable, run a longer continuation

## What to Compare

Always compare against `paper-baseline` using:

- `SSIM`
- `PSNR`
- `RMSE`

Treat a run as promising only if:

- `SSIM` improves, and
- `PSNR` / `RMSE` do not clearly regress
