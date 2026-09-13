#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_DQU="/root/autodl-tmp/dqucir_2026-08-05/repo"
CROSSPATH_DQU_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_CIRR_DQU="$CROSSPATH_DQU/data/CIRR"
CROSSPATH_OUTPUT="$CROSSPATH_DQU/checkpoints/e25_seed42"

export HF_HOME="/root/autodl-tmp/cache/hf"
export HF_HUB_OFFLINE=1

if [[ ! -f "$CROSSPATH_CIRR_DQU/captions/captions/cap.rc2.train.json" ]]; then
  echo "Missing prepared DQU CIRR dataset: $CROSSPATH_CIRR_DQU" >&2
  exit 1
fi

mkdir -p "$CROSSPATH_OUTPUT"
cd "$CROSSPATH_DQU/src"
exec "$CROSSPATH_DQU_PY" train.py \
  --dataset cirr \
  --cirr_path "$CROSSPATH_CIRR_DQU/" \
  --model_dir "$CROSSPATH_OUTPUT" \
  --i seed42 \
  --seed 42 \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 1e-4 \
  --clip_lr 1e-6 \
  --weight_decay 1e-2 \
  --dropout_rate 0.5 \
  --hidden_dim 1024 \
  --grad_ckpt 1
