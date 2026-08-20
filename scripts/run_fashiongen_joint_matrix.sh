#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_SOURCE_INTERNAL="$CROSSPATH_WEAVE/runs/CrossPath_A1seedpair_20260816_internal_v1"
CROSSPATH_SOURCE_OFFICIAL="$CROSSPATH_WEAVE/runs/CrossPath_A1seedpair_20260816_official_v1"
CROSSPATH_RUN="${CROSSPATH_RUN:-$CROSSPATH_WEAVE/runs/CrossPath_FashionGen_JointMatrix_20260820_v1}"

export PYTHONHASHSEED=0
export PYTHONPATH="/root/autodl-tmp/fastpath_pkgs:$CROSSPATH_WEAVE"

test ! -e "$CROSSPATH_RUN"
mkdir -p "$CROSSPATH_RUN/internal" "$CROSSPATH_RUN/official"
ln -s "$CROSSPATH_SOURCE_INTERNAL/embeddings" "$CROSSPATH_RUN/internal/embeddings"
ln -s "$CROSSPATH_SOURCE_OFFICIAL/embeddings" "$CROSSPATH_RUN/official/embeddings"

"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_build_crosspath_cache.py" \
  --embedding-dir "$CROSSPATH_RUN/internal/embeddings" \
  --output-dir "$CROSSPATH_RUN/internal/cache" \
  --query-batch-size 128 \
  --cache-workers 8 \
  --cutoffs 1 5 10 \
  --correction-path joint \
  --device cuda \
  > "$CROSSPATH_RUN/internal/cache.log" 2>&1

"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_train_crosspath_gate.py" \
  --embedding-dir "$CROSSPATH_RUN/internal/embeddings" \
  --cache-dir "$CROSSPATH_RUN/internal/cache" \
  --output-dir "$CROSSPATH_RUN/internal/gate" \
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
  > "$CROSSPATH_RUN/internal/gate.log" 2>&1

"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_build_crosspath_cache.py" \
  --embedding-dir "$CROSSPATH_RUN/official/embeddings" \
  --output-dir "$CROSSPATH_RUN/official/cache" \
  --query-batch-size 128 \
  --cache-workers 8 \
  --cutoffs 1 5 10 \
  --correction-path joint \
  --device cuda \
  > "$CROSSPATH_RUN/official/cache.log" 2>&1

"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_eval_crosspath_gate.py" \
  --embedding-dir "$CROSSPATH_RUN/official/embeddings" \
  --cache-dir "$CROSSPATH_RUN/official/cache" \
  --gate-dir "$CROSSPATH_RUN/internal/gate" \
  --output-dir "$CROSSPATH_RUN/official/eval" \
  --bootstrap-samples 100 \
  --bootstrap-seed 20260820 \
  --device cuda \
  > "$CROSSPATH_RUN/official/eval.log" 2>&1

echo "CrossPath FashionGen joint-matrix pipeline completed"
