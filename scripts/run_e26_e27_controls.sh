#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_DQU_RUN="$CROSSPATH_WEAVE/runs/CrossPath_FashionIQ_DQU_20260820_v1"
CROSSPATH_E24_RUN="$CROSSPATH_WEAVE/runs/E24_Heterogeneous_CrossPath_FashionIQ_20260824_v1"
CROSSPATH_FASHIONGEN="$CROSSPATH_WEAVE/runs/CrossPath_A1seedpair_20260816_official_v1/embeddings"
CROSSPATH_OUTPUT="$CROSSPATH_WEAVE/runs/E26_Controls_E27_Matrix3_20260824_v1"
CROSSPATH_PYTHON="/root/autodl-tmp/envs/procir-eval/bin/python"

mkdir -p "$CROSSPATH_OUTPUT"

"$CROSSPATH_PYTHON" "$CROSSPATH_WEAVE/analyze_crosspath_controls.py" \
  --embedding-dir "$CROSSPATH_FASHIONGEN" \
  --output "$CROSSPATH_OUTPUT/fashiongen_controls.json" \
  --baseline-path q0_g0 --method-path cross_mean \
  --device cuda --batch-size 128 --cutoffs 1 5 10

"$CROSSPATH_PYTHON" "$CROSSPATH_WEAVE/analyze_crosspath_controls.py" \
  --run-root "$CROSSPATH_E24_RUN/dqu_gc_mcot" \
  --output "$CROSSPATH_OUTPUT/fashioniq_dqu_gc_mcot_controls.json" \
  --baseline-path q1_g1 --method-path cross_mean \
  --device cuda --batch-size 128 --cutoffs 1 10 50 --exclude-source

"$CROSSPATH_PYTHON" "$CROSSPATH_WEAVE/eval_fashioniq_matrix3.py" \
  --dqu-root "$CROSSPATH_DQU_RUN" \
  --mcot-root "$CROSSPATH_E24_RUN/mcot" \
  --output "$CROSSPATH_OUTPUT/fashioniq_matrix3.json" \
  --device cuda --batch-size 128 --cutoffs 1 10 50

echo "E26 controls and E27 matrix completed"
