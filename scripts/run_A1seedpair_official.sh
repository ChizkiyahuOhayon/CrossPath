#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_INTERNAL="/root/autodl-tmp/weave/runs/CrossPath_A1seedpair_20260816_internal_v1"
CROSSPATH_OFFICIAL="/root/autodl-tmp/weave/runs/CrossPath_A1seedpair_20260816_official_v1"

export PYTHONHASHSEED=0
export PYTHONPATH="/root/autodl-tmp/fastpath_pkgs:/root/autodl-tmp/weave"

test -f "$CROSSPATH_INTERNAL/gate/manifest.json"
"$CROSSPATH_PY" -c 'import json, sys; raise SystemExit(not json.load(open(sys.argv[1]))["report"]["internal_go"])' "$CROSSPATH_INTERNAL/gate/manifest.json"
test ! -e "$CROSSPATH_OFFICIAL"
mkdir -p "$CROSSPATH_OFFICIAL"

echo "Exporting official FashionGen validation embeddings"
"$CROSSPATH_PY" "$CROSSPATH_WEAVE/weave_extract_crosspath_embeddings.py" \
  --fashionmv-repo /root/autodl-tmp/third_party/FashionMV \
  --base-model-path /root/autodl-tmp/weave/runs/A1_full_fgen_official_seed20260718/checkpoint-final \
  --correction-model-path /root/autodl-tmp/weave/runs/A1_full_fgen_official_seed20260722/checkpoint-final \
  --image-root /root/autodl-tmp/data/fashionmv/images \
  --data-dir /root/autodl-tmp/data/fashionmv/data \
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
echo "Evaluating the frozen gate with 10,000 paired bootstrap samples"
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
echo "CrossPath A1 seed-pair official evaluation completed"
