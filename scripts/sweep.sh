#!/usr/bin/env bash
# One size's runs, one after another, on one pod. See docs/runbook.md, step 6.
#
#   scripts/sweep.sh SIZE EPOCHS RECIPE...
#
# A recipe is rank:learning_rate:volume:seed, volume "all" for every training filing, so
# the 2B base recipe is 16:1e-4:all:0. Needs VOLUME_ID and DATACENTER in the environment
# with the S3 key, and HF_HOME on the container disk.
#
# Each run checkpoints to the network volume under its own name and, when it finishes,
# copies its adapter and run.json there too: the pod's disk is not somewhere a finished
# run is kept. A run whose run.json is already on the volume is skipped, so the same
# command on a replacement pod carries on where the lost one stopped.
set -euo pipefail

size=$1
epochs=$2
shift 2
: "${VOLUME_ID:?}" "${DATACENTER:?}"
s3=(--region "$DATACENTER" --endpoint-url "https://s3api-${DATACENTER,,}.runpod.io/")
mkdir -p logs runs

for recipe in "$@"; do
  IFS=: read -r rank lr volume seed <<<"$recipe"
  name="$size-r$rank-lr$lr-n$volume-s$seed-e$epochs"
  done_uri="s3://$VOLUME_ID/adapters/$name"
  if aws s3 ls "${s3[@]}" "$done_uri/run.json" >/dev/null 2>&1; then
    echo "=== $name already on the volume, skipped"
    continue
  fi
  args=(--size "$size" --epochs "$epochs" --rank "$rank" --learning-rate "$lr" --seed "$seed")
  if [ "$volume" != all ]; then args+=(--volume "$volume"); fi
  echo "=== $name started $(date -u +%FT%TZ)"
  uv run --no-sync smallprint train run --dataset-dir data/train/full "${args[@]}" \
    --out "runs/$name" --checkpoint-uri "s3://$VOLUME_ID/checkpoints/$name" \
    --s3-datacenter "$DATACENTER" >"logs/$name.log" 2>&1
  aws s3 sync "${s3[@]}" --quiet "runs/$name/adapter" "$done_uri/adapter"
  aws s3 cp "${s3[@]}" --quiet "runs/$name/run.json" "$done_uri/run.json"
  echo "=== $name finished $(date -u +%FT%TZ)"
done
echo "=== ALL-DONE"
