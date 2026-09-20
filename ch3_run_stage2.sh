#!/bin/bash
# Stage-2 retrieval fine-tuning for every chapter-3 table.
#
#   m1..m5          table 3.5 ablation; m1 is table 3.2's Baseline+, m5 the full model
#   L12..L8_L12     table 3.6 layer combinations; L4+L8+L12 is m5
#   lam0_1..lam1_0  figure 3.6 lambda sweep; lambda=0 is m4 and lambda=0.5 is m5
#
# Runs are sequential: two on one 24 GB card fit but leave no headroom.  Each
# run resumes if interrupted, and its optimizer state is deleted once it
# finishes so that twelve runs fit in the free space on this disk.
#
# The Sample protocol scores 202,000 candidate pairs through the cross-attention
# blocks, so it runs every fourth epoch as a divergence check rather than every
# epoch.  The last epoch is always scored, and that is the one that is reported.
set -u

FROZEN=${FROZEN:?set FROZEN to the pinned code snapshot}
RUNS=/root/autodl-tmp/ch3/runs
INIT=$RUNS/stage1_pretrain/epoch25.pth
PY=/root/autodl-tmp/envs/procir-eval/bin/python

export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/cache/huggingface PYTHONHASHSEED=0

echo "=== stage 2 waiting for $INIT ==="
until [ -f "$INIT" ] && grep -q "training complete" "$RUNS/stage1.log" 2>/dev/null; do sleep 300; done
echo "=== stage 1 done, starting $(date -Is) ==="

# keep the final weights of the ablation ladder only; the sweeps are read from their logs
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
    rm -f "$out/resume.pth"          # ~4 GB of optimizer state, only needed to restart
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
run_one L12 0
run_one L8 0
run_one L4_L8 0
run_one L8_L12 0
run_one lam0_1 0
run_one lam0_3 0
run_one lam1_0 0

echo "=== stage 2 complete $(date -Is) ==="
$PY /root/autodl-tmp/ch3/repo/ch3_collect_results.py --runs "$RUNS" --out "$RUNS/ch3_results.json"
