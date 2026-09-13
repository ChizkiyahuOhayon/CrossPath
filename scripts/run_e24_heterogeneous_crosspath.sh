#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_MCOT="/root/autodl-tmp/mcot_mvs"
CROSSPATH_DQU_RUN="/root/autodl-tmp/weave/runs/CrossPath_FashionIQ_DQU_20260820_v1"
CROSSPATH_RUN="/root/autodl-tmp/weave/runs/E24_Heterogeneous_CrossPath_FashionIQ_20260824_v1"
CROSSPATH_MCOT_PY="/root/miniconda3/bin/python"
CROSSPATH_EVAL_PY="/root/autodl-tmp/envs/procir-eval/bin/python"

export HF_HOME="/root/autodl-tmp/cache/hf"
export HF_HUB_OFFLINE=1

checkpoint_for() {
  case "$1" in
    dress) echo "$CROSSPATH_MCOT/checkpoints/mcot_mvs_dress.pt" ;;
    shirt) echo "$CROSSPATH_MCOT/checkpoints/mcot_mvs_shirt.pt" ;;
    toptee) echo "$CROSSPATH_MCOT/checkpoints/mcot_mvs_toptee.pt" ;;
    *) return 1 ;;
  esac
}

link_pair() {
  local category="$1"
  local dqu_prefix="$2"
  local pair_name="$3"
  local dqu_dir="$CROSSPATH_DQU_RUN/$category/official/embeddings"
  local mcot_dir="$CROSSPATH_RUN/mcot/$category"
  local pair_dir="$CROSSPATH_RUN/$pair_name/$category/official/embeddings"

  mkdir -p "$pair_dir"
  ln -sfn "$dqu_dir/${dqu_prefix}_gallery.npy" "$pair_dir/base_gallery.npy"
  ln -sfn "$dqu_dir/${dqu_prefix}_queries.npy" "$pair_dir/base_queries.npy"
  ln -sfn "$mcot_dir/gallery.npy" "$pair_dir/correction_gallery.npy"
  ln -sfn "$mcot_dir/queries.npy" "$pair_dir/correction_queries.npy"
  ln -sfn "$dqu_dir/gallery_ids.json" "$pair_dir/gallery_ids.json"
  ln -sfn "$dqu_dir/queries.jsonl" "$pair_dir/queries.jsonl"
}

mkdir -p "$CROSSPATH_RUN/mcot"

for category in dress shirt toptee; do
  mcot_dir="$CROSSPATH_RUN/mcot/$category"
  if [[ ! -f "$mcot_dir/manifest.json" ]]; then
    if [[ -d "$mcot_dir" ]] && [[ -n "$(find "$mcot_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Incomplete non-empty MCoT output: $mcot_dir" >&2
      exit 1
    fi
    (
      cd "$CROSSPATH_MCOT/src"
      "$CROSSPATH_MCOT_PY" "$CROSSPATH_WEAVE/weave_extract_crosspath_mcot.py" \
        --repo-src "$CROSSPATH_MCOT/src" \
        --fashioniq-path "$CROSSPATH_MCOT/data/fiq/fashionIQ_dataset" \
        --checkpoint "$(checkpoint_for "$category")" \
        --alignment-dir "$CROSSPATH_DQU_RUN/$category/official/embeddings" \
        --output-dir "$mcot_dir" \
        --category "$category" \
        --batch-size 32 \
        --workers 8
    ) > "$CROSSPATH_RUN/mcot/${category}.log" 2>&1
  fi

  link_pair "$category" base dqu_base_mcot
  link_pair "$category" correction dqu_gc_mcot
done

for pair_name in dqu_base_mcot dqu_gc_mcot; do
  "$CROSSPATH_EVAL_PY" "$CROSSPATH_WEAVE/eval_cross_compatibility.py" \
    --run-root "$CROSSPATH_RUN/$pair_name" \
    --output "$CROSSPATH_RUN/${pair_name}_summary.json" \
    --device cuda \
    --batch-size 128 \
    --exclude-source \
    > "$CROSSPATH_RUN/${pair_name}.log" 2>&1

  "$CROSSPATH_EVAL_PY" "$CROSSPATH_WEAVE/eval_cross_compatibility.py" \
    --run-root "$CROSSPATH_RUN/$pair_name" \
    --output "$CROSSPATH_RUN/${pair_name}_include_source_summary.json" \
    --device cuda \
    --batch-size 128 \
    --include-source \
    > "$CROSSPATH_RUN/${pair_name}_include_source.log" 2>&1
done

echo "E24 heterogeneous CrossPath completed"
