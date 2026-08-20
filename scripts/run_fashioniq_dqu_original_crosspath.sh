#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_DQU="/root/autodl-tmp/dqucir_2026-08-05"
CROSSPATH_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_SOURCE="/root/autodl-tmp/weave/runs/CrossPath_FashionIQ_DQU_20260820_v1"
CROSSPATH_RUN="/root/autodl-tmp/weave/runs/CrossPath_FashionIQ_DQU_original_20260820_v1"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/root/autodl-tmp/cache/hf"
export PYTHONHASHSEED=0
export PYTHONPATH="$CROSSPATH_WEAVE"

test ! -e "$CROSSPATH_RUN"
mkdir -p "$CROSSPATH_RUN"

for CROSSPATH_CAT in dress shirt toptee; do
  CROSSPATH_BASE="$CROSSPATH_DQU/repo/src/checkpoints/${CROSSPATH_CAT}_s42_best_state.pt"
  CROSSPATH_CORRECTION="$CROSSPATH_DQU/gc128/${CROSSPATH_CAT}_gc128_best_state.pt"
  CROSSPATH_OFFICIAL="$CROSSPATH_RUN/$CROSSPATH_CAT/official"
  mkdir -p "$CROSSPATH_OFFICIAL"

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_extract_crosspath_dqu.py" \
    --repo-src "$CROSSPATH_DQU/repo/src" \
    --fashioniq-path "$CROSSPATH_DQU/repo/data/FashionIQ" \
    --base-checkpoint "$CROSSPATH_BASE" \
    --correction-checkpoint "$CROSSPATH_CORRECTION" \
    --category "$CROSSPATH_CAT" \
    --split val \
    --gallery-protocol original-split \
    --output-dir "$CROSSPATH_OFFICIAL/embeddings" \
    --batch-size 16 \
    > "$CROSSPATH_OFFICIAL/extract.log" 2>&1

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_build_crosspath_cache.py" \
    --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
    --output-dir "$CROSSPATH_OFFICIAL/cache" \
    --query-batch-size 16 \
    --cutoffs 1 10 50 \
    --exclude-source \
    --device cuda \
    > "$CROSSPATH_OFFICIAL/cache.log" 2>&1

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_eval_crosspath_gate.py" \
    --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
    --cache-dir "$CROSSPATH_OFFICIAL/cache" \
    --gate-dir "$CROSSPATH_SOURCE/$CROSSPATH_CAT/internal/gate" \
    --output-dir "$CROSSPATH_OFFICIAL/eval" \
    --bootstrap-samples 100 \
    --bootstrap-seed 20260820 \
    --device cuda \
    > "$CROSSPATH_OFFICIAL/eval.log" 2>&1
done

"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_aggregate_crosspath_fashioniq.py" \
  --run-root "$CROSSPATH_RUN" \
  > "$CROSSPATH_RUN/summary.log" 2>&1

echo "CrossPath FashionIQ original-split pipeline completed"
