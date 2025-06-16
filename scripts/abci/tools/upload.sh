#!/bin/bash

set -e

upload_base_dir=/groups/gag51395/fujii/checkpoints/nemo-to-hf/Llama-3.1-8b-v0.5/lmsys-chat-1m-gemma-3-ja/hf

upload_checkpoint() {
  local upload_dir=$1
  local repo_name=$2
  local max_retries=5
  local retry_count=0

  while [ $retry_count -lt $max_retries ]; do
    if python scripts/abci/tools/upload.py \
        --ckpt-path "$upload_dir" \
        --repo-name "$repo_name"; then
        echo "Successfully uploaded $repo_name"
        return 0
    else
        echo "Upload failed for $repo_name. Retrying..."
        ((retry_count++))
        sleep 5
    fi
  done

  echo "Failed to upload $repo_name after $max_retries attempts"
  return 1
}

upload_dir=$upload_base_dir
repo_name="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5-nemo-aliner-lmsys-chat-1m-gemma-3-ja"

if ! upload_checkpoint "$upload_dir" "$repo_name"; then
  echo "Skipping to next checkpoint after repeated failures for $repo_name"
fi
