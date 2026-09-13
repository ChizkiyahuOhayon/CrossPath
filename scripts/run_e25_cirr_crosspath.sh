#!/usr/bin/env bash
set -euo pipefail

CROSSPATH_WEAVE="/root/autodl-tmp/weave"
CROSSPATH_DQU="/root/autodl-tmp/dqucir_2026-08-05/repo"
CROSSPATH_MCOT="/root/autodl-tmp/mcot_mvs"
CROSSPATH_RUN="$CROSSPATH_WEAVE/runs/E25_CIRR_CrossPath_20260901_v1"
CROSSPATH_DQU_DATA="$CROSSPATH_DQU/data/CIRR"
CROSSPATH_MCOT_DATA="$CROSSPATH_MCOT/data/cirr/cirr_dataset"
CROSSPATH_DQU_CHECKPOINT="$CROSSPATH_DQU/checkpoints/e25_seed42/cirr_seed42_best_model.pt"
CROSSPATH_MCOT_CHECKPOINT="$CROSSPATH_MCOT/checkpoints/mcot_mvs_cirr.pt"
CROSSPATH_DQU_PY="/root/autodl-tmp/envs/procir-eval/bin/python"
CROSSPATH_MCOT_PY="/root/miniconda3/bin/python"
CROSSPATH_EVAL_PY="/root/autodl-tmp/envs/procir-eval/bin/python"

export HF_HOME="/root/autodl-tmp/cache/hf"
export HF_HUB_OFFLINE=1

for required in \
  "$CROSSPATH_DQU_CHECKPOINT" \
  "$CROSSPATH_MCOT_CHECKPOINT" \
  "$CROSSPATH_DQU_DATA/captions/captions/cap.rc2.val.json" \
  "$CROSSPATH_MCOT_DATA/cirr/captions/cap.rc2.val.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

link_pair() {
  local split="$1"
  local pair_dir="$CROSSPATH_RUN/$split/embeddings"
  mkdir -p "$pair_dir"
  ln -sfn "$CROSSPATH_RUN/dqu/$split/gallery.npy" "$pair_dir/base_gallery.npy"
  ln -sfn "$CROSSPATH_RUN/dqu/$split/queries.npy" "$pair_dir/base_queries.npy"
  ln -sfn "$CROSSPATH_RUN/mcot/$split/gallery.npy" "$pair_dir/correction_gallery.npy"
  ln -sfn "$CROSSPATH_RUN/mcot/$split/queries.npy" "$pair_dir/correction_queries.npy"
  ln -sfn "$CROSSPATH_RUN/dqu/$split/gallery_ids.json" "$pair_dir/gallery_ids.json"
  ln -sfn "$CROSSPATH_RUN/dqu/$split/queries.jsonl" "$pair_dir/queries.jsonl"
}

extract_split() {
  local split="$1"
  local dqu_dir="$CROSSPATH_RUN/dqu/$split"
  local mcot_dir="$CROSSPATH_RUN/mcot/$split"
  mkdir -p "$CROSSPATH_RUN/dqu" "$CROSSPATH_RUN/mcot"

  if [[ ! -f "$dqu_dir/manifest.json" ]]; then
    (
      cd "$CROSSPATH_DQU/src"
      "$CROSSPATH_DQU_PY" "$CROSSPATH_WEAVE/weave_extract_crosspath_dqu_cirr.py" \
        --repo-src "$CROSSPATH_DQU/src" \
        --cirr-path "$CROSSPATH_DQU_DATA" \
        --checkpoint "$CROSSPATH_DQU_CHECKPOINT" \
        --split "$split" \
        --output-dir "$dqu_dir" \
        --batch-size 16
    ) > "$CROSSPATH_RUN/dqu_${split}.log" 2>&1
  fi

  if [[ ! -f "$mcot_dir/manifest.json" ]]; then
    (
      cd "$CROSSPATH_MCOT/src"
      "$CROSSPATH_MCOT_PY" "$CROSSPATH_WEAVE/weave_extract_crosspath_mcot_cirr.py" \
        --repo-src "$CROSSPATH_MCOT/src" \
        --cirr-path "$CROSSPATH_MCOT_DATA" \
        --checkpoint "$CROSSPATH_MCOT_CHECKPOINT" \
        --alignment-dir "$dqu_dir" \
        --split "$split" \
        --output-dir "$mcot_dir" \
        --batch-size 16 \
        --workers 8
    ) > "$CROSSPATH_RUN/mcot_${split}.log" 2>&1
  fi
  link_pair "$split"
}

mkdir -p "$CROSSPATH_RUN"
extract_split val
"$CROSSPATH_EVAL_PY" "$CROSSPATH_WEAVE/scripts/eval_cirr_cross_compatibility.py" \
  --embedding-dir "$CROSSPATH_RUN/val/embeddings" \
  --output "$CROSSPATH_RUN/val_summary.json" \
  --batch-size 128 \
  > "$CROSSPATH_RUN/val_eval.log" 2>&1

CROSSPATH_BEST_PATH="$($CROSSPATH_EVAL_PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_path"])' "$CROSSPATH_RUN/val_summary.json")"
printf 'Selected CIRR path on validation Avg: %s\n' "$CROSSPATH_BEST_PATH"

extract_split test1
"$CROSSPATH_EVAL_PY" "$CROSSPATH_WEAVE/scripts/eval_cirr_cross_compatibility.py" \
  --embedding-dir "$CROSSPATH_RUN/test1/embeddings" \
  --output "$CROSSPATH_RUN/test1_submission_manifest.json" \
  --batch-size 128 \
  --submission-path "$CROSSPATH_BEST_PATH" \
  --submission-dir "$CROSSPATH_RUN/submission" \
  > "$CROSSPATH_RUN/test1_submission.log" 2>&1

echo "E25 CIRR CrossPath completed: $CROSSPATH_RUN"
