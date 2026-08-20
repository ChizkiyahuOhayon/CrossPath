#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_SOURCE="/root/autodl-tmp/weave/runs/CrossPath_FashionIQ_DQU_20260820_v1"
CROSSPATH_OFFICIAL_SOURCE="/root/autodl-tmp/weave/runs/CrossPath_FashionIQ_DQU_original_20260820_v1"
CROSSPATH_CORRECTION_PATH="${CROSSPATH_CORRECTION_PATH:-cross-mean}"
CROSSPATH_RUN="${CROSSPATH_RUN:-/root/autodl-tmp/weave/runs/CrossPath_FashionIQ_CrossMatrix_original_20260820_v1}"

export PYTHONHASHSEED=0
export PYTHONPATH="$CROSSPATH_WEAVE"

test ! -e "$CROSSPATH_RUN"
mkdir -p "$CROSSPATH_RUN"

for CROSSPATH_CAT in dress shirt toptee; do
  CROSSPATH_CAT_ROOT="$CROSSPATH_RUN/$CROSSPATH_CAT"
  CROSSPATH_INTERNAL="$CROSSPATH_CAT_ROOT/internal"
  CROSSPATH_OFFICIAL="$CROSSPATH_CAT_ROOT/official"
  mkdir -p "$CROSSPATH_INTERNAL" "$CROSSPATH_OFFICIAL"
  ln -s "$CROSSPATH_SOURCE/$CROSSPATH_CAT/internal/embeddings" "$CROSSPATH_INTERNAL/embeddings"
  ln -s "$CROSSPATH_OFFICIAL_SOURCE/$CROSSPATH_CAT/official/embeddings" "$CROSSPATH_OFFICIAL/embeddings"

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_build_crosspath_cache.py" \
    --embedding-dir "$CROSSPATH_INTERNAL/embeddings" \
    --output-dir "$CROSSPATH_INTERNAL/cache" \
    --query-batch-size 16 \
    --cutoffs 1 10 50 \
    --exclude-source \
    --correction-path "$CROSSPATH_CORRECTION_PATH" \
    --device cuda \
    > "$CROSSPATH_INTERNAL/cache.log" 2>&1

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_train_crosspath_gate.py" \
    --embedding-dir "$CROSSPATH_INTERNAL/embeddings" \
    --cache-dir "$CROSSPATH_INTERNAL/cache" \
    --output-dir "$CROSSPATH_INTERNAL/gate" \
    --widths 128 256 \
    --epochs 3 \
    --batch-records 64 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --grad-clip 1.0 \
    --regression-cost 2.0 \
    --seed 20260820 \
    --feature-mode full \
    --device cuda \
    > "$CROSSPATH_INTERNAL/gate.log" 2>&1

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_build_crosspath_cache.py" \
    --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
    --output-dir "$CROSSPATH_OFFICIAL/cache" \
    --query-batch-size 16 \
    --cutoffs 1 10 50 \
    --exclude-source \
    --correction-path "$CROSSPATH_CORRECTION_PATH" \
    --device cuda \
    > "$CROSSPATH_OFFICIAL/cache.log" 2>&1

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_eval_crosspath_gate.py" \
    --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
    --cache-dir "$CROSSPATH_OFFICIAL/cache" \
    --gate-dir "$CROSSPATH_INTERNAL/gate" \
    --output-dir "$CROSSPATH_OFFICIAL/eval" \
    --bootstrap-samples 100 \
    --bootstrap-seed 20260820 \
    --device cuda \
    > "$CROSSPATH_OFFICIAL/eval.log" 2>&1
done

"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_aggregate_crosspath_fashioniq.py" \
  --run-root "$CROSSPATH_RUN" \
  > "$CROSSPATH_RUN/summary.log" 2>&1

echo "CrossPath FashionIQ $CROSSPATH_CORRECTION_PATH pipeline completed"
