#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_NAME="${RUN_NAME:-noise_d_clean}"
RESUME_ITER="${RESUME_ITER:-0}"
MAX_ITER="${MAX_ITER:-100000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

LAMBDA_NOISE_FIXED="${LAMBDA_NOISE_FIXED:-0.1}"
LAMBDA_NOISE_REG="${LAMBDA_NOISE_REG:-0.3}"
LAMBDA_GATE_BALANCE="${LAMBDA_GATE_BALANCE:-0.01}"
MOE_PHASE="${MOE_PHASE:-1}"

python main.py \
  --model_name DUGAN_MoE \
  --train_dataset_name cmayo_train_64 \
  --test_dataset_name cmayo_test_512 \
  --batch_size "${BATCH_SIZE}" \
  --test_batch_size "${TEST_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --resume_iter "${RESUME_ITER}" \
  --max_iter "${MAX_ITER}" \
  --save_freq "${SAVE_FREQ}" \
  --cr_loss_weight 5.08720932695335 \
  --cutmix_prob 0.7615524094697519 \
  --cutmix_warmup_iter 1000 \
  --d_lr 7.122979672016055e-05 \
  --g_lr 0.00018083340390609657 \
  --grad_gen_loss_weight 0.11960717521104237 \
  --grad_loss_weight 35.310016043755894 \
  --img_gen_loss_weight 0.14178356036938378 \
  --pix_loss_weight 5.034293425614828 \
  --run_name "${RUN_NAME}" \
  --use_grad_discriminator true \
  --weight_decay 0. \
  --moe_phase "${MOE_PHASE}" \
  --lambda_noise_reg "${LAMBDA_NOISE_REG}" \
  --lambda_noise_fixed "${LAMBDA_NOISE_FIXED}" \
  --lambda_gate_balance "${LAMBDA_GATE_BALANCE}"
