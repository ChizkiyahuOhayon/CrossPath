#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_OUT="$CROSSPATH_WEAVE/runs/CrossPath_efficiency_20260820"
CROSSPATH_FG_INTERNAL="$CROSSPATH_WEAVE/runs/CrossPath_A1seedpair_20260816_internal_v1"
CROSSPATH_FG_OFFICIAL="$CROSSPATH_WEAVE/runs/CrossPath_A1seedpair_20260816_official_v1"
CROSSPATH_FIQ="$CROSSPATH_WEAVE/runs/CrossPath_FashionIQ_DQU_20260820_v1"

export PYTHONHASHSEED=0
export PYTHONPATH="/root/autodl-tmp/fastpath_pkgs:$CROSSPATH_WEAVE"

test ! -e "$CROSSPATH_OUT"
mkdir -p "$CROSSPATH_OUT"

"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_benchmark_crosspath_efficiency.py" \
  --embedding-dir "$CROSSPATH_FG_OFFICIAL/embeddings" \
  --cache-dir "$CROSSPATH_FG_OFFICIAL/cache" \
  --gate-dir "$CROSSPATH_FG_INTERNAL/gate" \
  --output "$CROSSPATH_OUT/fashiongen.json" \
  --max-queries 500 \
  --device cuda

for CROSSPATH_CAT in dress shirt toptee; do
  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_benchmark_crosspath_efficiency.py" \
    --embedding-dir "$CROSSPATH_FIQ/$CROSSPATH_CAT/official/embeddings" \
    --cache-dir "$CROSSPATH_FIQ/$CROSSPATH_CAT/official/cache" \
    --gate-dir "$CROSSPATH_FIQ/$CROSSPATH_CAT/internal/gate" \
    --output "$CROSSPATH_OUT/fashioniq_${CROSSPATH_CAT}.json" \
    --max-queries 500 \
    --device cuda
done

echo "CrossPath efficiency benchmark completed"
