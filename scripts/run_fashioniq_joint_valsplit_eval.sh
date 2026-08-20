#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_EMBEDDINGS="$CROSSPATH_WEAVE/runs/CrossPath_FashionIQ_DQU_20260820_v1"
CROSSPATH_GATES="$CROSSPATH_WEAVE/runs/CrossPath_FashionIQ_JointMatrix_original_20260820_v1"
CROSSPATH_RUN="${CROSSPATH_RUN:-$CROSSPATH_WEAVE/runs/CrossPath_FashionIQ_JointMatrix_valsplit_20260820_v1}"

export PYTHONHASHSEED=0
export PYTHONPATH="$CROSSPATH_WEAVE"

test ! -e "$CROSSPATH_RUN"
mkdir -p "$CROSSPATH_RUN"

for CROSSPATH_CAT in dress shirt toptee; do
  CROSSPATH_OFFICIAL="$CROSSPATH_RUN/$CROSSPATH_CAT/official"
  mkdir -p "$CROSSPATH_OFFICIAL"
  ln -s "$CROSSPATH_EMBEDDINGS/$CROSSPATH_CAT/official/embeddings" "$CROSSPATH_OFFICIAL/embeddings"

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_build_crosspath_cache.py" \
    --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
    --output-dir "$CROSSPATH_OFFICIAL/cache" \
    --query-batch-size 16 \
    --cutoffs 1 10 50 \
    --exclude-source \
    --correction-path joint \
    --device cuda \
    > "$CROSSPATH_OFFICIAL/cache.log" 2>&1

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_eval_crosspath_gate.py" \
    --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
    --cache-dir "$CROSSPATH_OFFICIAL/cache" \
    --gate-dir "$CROSSPATH_GATES/$CROSSPATH_CAT/internal/gate" \
    --output-dir "$CROSSPATH_OFFICIAL/eval" \
    --bootstrap-samples 100 \
    --bootstrap-seed 20260820 \
    --device cuda \
    > "$CROSSPATH_OFFICIAL/eval.log" 2>&1
done

"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_aggregate_crosspath_fashioniq.py" \
  --run-root "$CROSSPATH_RUN" \
  > "$CROSSPATH_RUN/summary.log" 2>&1

echo "CrossPath FashionIQ joint-matrix val-split evaluation completed"
