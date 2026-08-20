#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXTRACTION_PID" >&2
  exit 64
fi

CROSSPATH_EXTRACTION_PID="$1"
CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_FASTPATH="/root/autodl-tmp/fastpath_pkgs:/root/autodl-tmp/weave"
CROSSPATH_FASHIONMV="/root/autodl-tmp/third_party/FashionMV"
CROSSPATH_IMAGES="/root/autodl-tmp/data/fashionmv/images"
CROSSPATH_DATA="/root/autodl-tmp/data/fashionmv/data"
CROSSPATH_BASE="/root/autodl-tmp/weave/runs/A1_full_fgen_official_seed20260718/checkpoint-final"
CROSSPATH_CORRECTION="/root/autodl-tmp/weave/runs/A1_full_fgen_official_seed20260722/checkpoint-final"
CROSSPATH_INTERNAL="/root/autodl-tmp/weave/runs/CrossPath_A1seedpair_20260816_internal_v1"
CROSSPATH_OFFICIAL="/root/autodl-tmp/weave/runs/CrossPath_A1seedpair_20260816_official_v1"

export PYTHONHASHSEED=0
export PYTHONPATH="$CROSSPATH_FASTPATH"

echo "Waiting for embedding extraction PID $CROSSPATH_EXTRACTION_PID"
while kill -0 "$CROSSPATH_EXTRACTION_PID" 2>/dev/null; do
  sleep 45
done

test -f "$CROSSPATH_INTERNAL/embeddings/manifest.json"
test ! -e "$CROSSPATH_INTERNAL/cache"
echo "Building internal CrossPath cache"
"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_build_crosspath_cache.py" \
  --embedding-dir "$CROSSPATH_INTERNAL/embeddings" \
  --output-dir "$CROSSPATH_INTERNAL/cache" \
  --query-batch-size 16 \
  --device cuda \
  > "$CROSSPATH_INTERNAL/cache.log" 2>&1

test -f "$CROSSPATH_INTERNAL/cache/manifest.json"
test ! -e "$CROSSPATH_INTERNAL/gate"
echo "Training internal CrossPath gate"
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
  --seed 20260802 \
  --device cuda \
  > "$CROSSPATH_INTERNAL/gate.log" 2>&1

test -f "$CROSSPATH_INTERNAL/gate/manifest.json"
if ! "$CROSSPATH_PY" -c 'import json, sys; raise SystemExit(not json.load(open(sys.argv[1]))["report"]["internal_go"])' "$CROSSPATH_INTERNAL/gate/manifest.json"; then
  echo "Internal go/no-go failed; official evaluation was not started"
  exit 2
fi

test ! -e "$CROSSPATH_OFFICIAL"
mkdir -p "$CROSSPATH_OFFICIAL"
echo "Internal go/no-go passed; exporting official validation embeddings"
"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_extract_crosspath_embeddings.py" \
  --fashionmv-repo "$CROSSPATH_FASHIONMV" \
  --base-model-path "$CROSSPATH_BASE" \
  --correction-model-path "$CROSSPATH_CORRECTION" \
  --image-root "$CROSSPATH_IMAGES" \
  --data-dir "$CROSSPATH_DATA" \
  --output-dir "$CROSSPATH_OFFICIAL/embeddings" \
  --dataset fashiongen_val \
  --batch-size 10 \
  --num-workers 2 \
  > "$CROSSPATH_OFFICIAL/extract.log" 2>&1

test -f "$CROSSPATH_OFFICIAL/embeddings/manifest.json"
echo "Building official CrossPath cache"
"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_build_crosspath_cache.py" \
  --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
  --output-dir "$CROSSPATH_OFFICIAL/cache" \
  --query-batch-size 16 \
  --device cuda \
  > "$CROSSPATH_OFFICIAL/cache.log" 2>&1

test -f "$CROSSPATH_OFFICIAL/cache/manifest.json"
echo "Evaluating frozen gate on official validation data"
"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_eval_crosspath_gate.py" \
  --embedding-dir "$CROSSPATH_OFFICIAL/embeddings" \
  --cache-dir "$CROSSPATH_OFFICIAL/cache" \
  --gate-dir "$CROSSPATH_INTERNAL/gate" \
  --output-dir "$CROSSPATH_OFFICIAL/eval" \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260802 \
  --device cuda \
  > "$CROSSPATH_OFFICIAL/eval.log" 2>&1

test -f "$CROSSPATH_OFFICIAL/eval/manifest.json"
echo "CrossPath A1 seed-pair pipeline completed"
