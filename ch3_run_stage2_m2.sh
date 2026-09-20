#!/bin/bash
# Machine 2 half of stage 2: the layer sweep (figure 3.6) and the lambda sweep
# (table 3.6). Machine 1 pushes epoch25.pth here when stage 1 reaches it.
set -u
RUNS=/root/autodl-tmp/ch3/runs
FROZEN=/root/autodl-tmp/ch3/frozen_a9e5c558e2a9
INIT=$RUNS/epoch25.pth
PY=/root/miniconda3/bin/python
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/cache/huggingface PYTHONHASHSEED=0

echo "=== stage 2 (m2 half) waiting for $INIT ==="
# rsync writes a hidden temporary and renames only on success, so the file
# appearing under this name already means the push completed.
until [ -f "$INIT" ]; do sleep 120; done
echo "=== checkpoint present, starting $(date -Is) ==="

run_one() {
  local cfg=$1 keep=$2
  local out=$RUNS/$cfg
  if [ -f "$out/DONE" ]; then echo "skip $cfg (done)"; return; fi
  echo "=== $cfg  $(date -Is) ==="
  cd "$FROZEN"
  $PY -m ch3_mlaff.train_ch3 \
    --stage retrieval --config "$cfg" --init-from "$INIT" \
    --data-root /root/autodl-tmp/ch3/data/fashiongen_h5 \
    --cache-dir /root/autodl-tmp/ch3/data/index \
    --out-dir "$out" \
    --epochs 20 --batch-size 16 --num-workers 24 --amp bf16 \
    --eval-every 4 --keep-epochs "" --keep-last "$keep" --seed 42 --resume \
    >> "$RUNS/$cfg.log" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    rm -f "$out/resume.pth"
    touch "$out/DONE"
  else
    echo "!!! $cfg exited $status; leaving resume.pth so it can continue"
  fi
}

run_one L12 0
run_one L8 0
run_one L4_L8 0
run_one L8_L12 0
run_one lam0_1 0
run_one lam0_3 0
run_one lam1_0 0

echo "=== stage 2 (m2 half) complete $(date -Is) ==="
