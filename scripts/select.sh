#!/usr/bin/env bash
# Choosing each size's recipe: sweep runs merged, served and graded on the validation set.
# See docs/runbook.md, step 7.
#
#   scripts/select.sh NAME...
#
# NAME is a run's name on the volume, as scripts/sweep.sh wrote it under adapters/NAME.
# Needs what scripts/evaluate.sh needs, plus the train extra for the merge.
#
# The same merge and the same evaluate.sh as a published model, on the validation split,
# which no headline number is taken from. The merged weights are the size of the base,
# 10 to 16 GB, and a candidate that is not chosen is never published, so they are made on
# the pod's disk, graded, and deleted; only the grades go to the volume, under evals/, and
# a run already graded there is skipped.
set -euo pipefail
: "${VOLUME_ID:?}" "${DATACENTER:?}"
s3=(--region "$DATACENTER" --endpoint-url "https://s3api-${DATACENTER,,}.runpod.io/")
mkdir -p logs models

for name in "$@"; do
  if aws s3 ls "${s3[@]}" "s3://$VOLUME_ID/evals/$name-bf16-validation/summary.json" >/dev/null 2>&1; then
    echo "=== $name already graded, skipped"
    continue
  fi
  echo "=== $name started $(date -u +%FT%TZ)"
  [ -f "runs/$name/adapter/adapter_config.json" ] ||
    aws s3 sync "${s3[@]}" --quiet "s3://$VOLUME_ID/adapters/$name" "runs/$name"
  [ -f "models/$name/bf16/merge.json" ] ||
    uv run --no-sync smallprint quantise merge --run-dir "runs/$name" --out "models/$name/bf16" \
      --dataset-dir data/train/full >"logs/$name-merge.log" 2>&1
  scripts/evaluate.sh "$name" bf16 validation
  rm -rf "models/$name"
  echo "=== $name finished $(date -u +%FT%TZ)"
done
echo "=== ALL-DONE"
