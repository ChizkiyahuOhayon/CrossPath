#!/bin/bash
# Machine 1 half of stage 2: the ablation ladder, i.e. tables 3.3 and 3.5.
set -u
RUNS=/root/autodl-tmp/ch3/runs
FROZEN=/root/autodl-tmp/ch3/frozen_a9e5c558e2a9
INIT=$RUNS/stage1_pretrain/epoch25.pth
PY=/root/autodl-tmp/envs/procir-eval/bin/python
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/cache/huggingface PYTHONHASHSEED=0

echo "=== stage 2 (m1 half) waiting for $INIT ==="
until [ -f "$INIT" ]; do sleep 300; done

# machine 2 runs the other half off the same checkpoint; it may still be powered
# off, so keep offering the file until it lands there.
( until rsync -a "$INIT" m2:/root/autodl-tmp/ch3/runs/epoch25.pth; do sleep 300; done
  echo "epoch25.pth delivered to m2 $(date -Is)" ) >> "$RUNS/push_m2.log" 2>&1 &

until grep -q "training complete" "$RUNS/stage1.log" 2>/dev/null; do sleep 300; done
echo "=== stage 1 done, starting $(date -Is) ==="

run_one() {
  local cfg=$1 keep=$2 extra=${3:-}
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
    --eval-every 4 --keep-epochs "" --keep-last "$keep" --seed 42 --resume $extra \
    >> "$RUNS/$cfg.log" 2>&1
  local status=$?
  if [ $status -eq 0 ]; then
    rm -f "$out/resume.pth"
    touch "$out/DONE"
  else
    echo "!!! $cfg exited $status; leaving resume.pth so it can continue"
  fi
}

run_one m5 1 --full-rerank     # table 3.3 needs the Full-protocol rerank
run_one m1 1
run_one m2 1
run_one m3 1
run_one m4 1

echo "=== stage 2 (m1 half) complete $(date -Is) ==="
