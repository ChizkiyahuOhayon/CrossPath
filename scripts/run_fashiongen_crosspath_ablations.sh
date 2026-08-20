#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_INTERNAL="/root/autodl-tmp/weave/runs/CrossPath_A1seedpair_20260816_internal_v1"
CROSSPATH_OFFICIAL="/root/autodl-tmp/weave/runs/CrossPath_A1seedpair_20260816_official_v1"

export PYTHONHASHSEED=0
export PYTHONPATH="/root/autodl-tmp/fastpath_pkgs:$CROSSPATH_WEAVE"

for CROSSPATH_MODE in query_only margin_only; do
  CROSSPATH_GATE="$CROSSPATH_INTERNAL/gate_${CROSSPATH_MODE}"
  CROSSPATH_EVAL="$CROSSPATH_OFFICIAL/eval_${CROSSPATH_MODE}"
  test ! -e "$CROSSPATH_GATE"
  test ! -e "$CROSSPATH_EVAL"

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_train_crosspath_gate.py" \
    --embedding-dir "$CROSSPATH_INTERNAL/embeddings" \
    --cache-dir "$CROSSPATH_INTERNAL/cache" \
    --output-dir "$CROSSPATH_GATE" \
    --widths 128 \
    --epochs 3 \
    --batch-records 64 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --grad-clip 1.0 \
    --regression-cost 2.0 \
    --seed 20260802 \
    --feature-mode "$CROSSPATH_MODE" \
    --device cuda \
    > "$CROSSPATH_INTERNAL/gate_${CROSSPATH_MODE}.log" 2>&1

  "$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_eval_crosspath_gate.py" \
    --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
    --cache-dir "$CROSSPATH_OFFICIAL/cache" \
    --gate-dir "$CROSSPATH_GATE" \
    --output-dir "$CROSSPATH_EVAL" \
    --bootstrap-samples 2000 \
    --bootstrap-seed 20260820 \
    --device cuda \
    > "$CROSSPATH_OFFICIAL/eval_${CROSSPATH_MODE}.log" 2>&1
done

echo "CrossPath FashionGen ablations completed"
