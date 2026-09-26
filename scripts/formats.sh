#!/usr/bin/env bash
# Every published format of one finished run, on one pod. See docs/runbook.md, step 8.
#
#   scripts/formats.sh NAME [FORMAT...]
#
# FORMAT is any of bf16, awq, gptq and gguf, all four when none is named; bf16 is made
# whichever are named, because the others are made from it.
#
# NAME is the run's name on the volume, as scripts/sweep.sh wrote it under adapters/NAME,
# like 2b-r16-lr1e-4-nall-s0-e1. Needs VOLUME_ID and DATACENTER in the environment with the
# S3 key, HF_HOME on the container disk, the train and quant extras installed, and, for
# GGUF, LLAMA_CPP pointing at a llama.cpp checkout with llama-quantize built in build/bin.
#
# The merged bf16 weights come first, because every other format is made from them and
# judged against them. They stay on the pod: they are the size of the base, the merge
# makes them again in two minutes from the adapter on the volume and checks them, and three
# sizes of them filled the volume until it refused the formats that are published
# (QuotaExceeded, 2026-09-26). Each other format goes to the volume under
# formats/NAME/<format> as soon as it is made, weights first and its record last, so a
# record on the volume means the weights beside it are whole; a format whose record is
# already there is skipped. The same command on a replacement pod carries on where the
# lost one stopped. Every store command is tried four times, as the training's are
# (smallprint/train/checkpoint.py), and its errors are printed, not hidden.
set -euo pipefail

name=$1
shift
wanted=" ${*:-bf16 awq gptq gguf} "
: "${VOLUME_ID:?}" "${DATACENTER:?}"
s3=(--region "$DATACENTER" --endpoint-url "https://s3api-${DATACENTER,,}.runpod.io/")
remote="s3://$VOLUME_ID/formats/$name"
dir=models/$name
mkdir -p logs "$dir"

retry() {
  local wait
  for wait in 10 30 90 0; do
    "$@" && return 0
    if [ "$wait" -gt 0 ]; then sleep "$wait"; fi
  done
  return 1
}
have() { aws s3 ls "${s3[@]}" "$remote/$1" >/dev/null 2>&1; }
# Weights, then the record: the record is the marker that the upload finished.
push() {
  retry aws s3 sync "${s3[@]}" --only-show-errors --exclude "$2" "$dir/$1" "$remote/$1"
  retry aws s3 cp "${s3[@]}" --only-show-errors "$dir/$1/$2" "$remote/$1/$2"
}
sp() { uv run --no-sync smallprint "$@"; }

retry aws s3 sync "${s3[@]}" --only-show-errors "s3://$VOLUME_ID/adapters/$name" "runs/$name"

echo "=== $name bf16 $(date -u +%FT%TZ)"
if [ ! -f "$dir/bf16/merge.json" ]; then
  sp quantise merge --run-dir "runs/$name" --out "$dir/bf16" \
    --dataset-dir data/train/full >"logs/$name-merge.log" 2>&1
fi

for fmt in awq gptq; do
  [[ $wanted == *" $fmt "* ]] || continue
  echo "=== $name $fmt $(date -u +%FT%TZ)"
  if have "$fmt/quant.json"; then continue; fi
  sp quantise format --format "$fmt" --model "$dir/bf16" --out "$dir/$fmt" \
    --dataset-dir data/train/full >"logs/$name-$fmt.log" 2>&1
  push "$fmt" quant.json
done

echo "=== $name gguf $(date -u +%FT%TZ)"
if [[ $wanted == *" gguf "* ]] && ! have "gguf/$name-q4_k_m.json"; then
  : "${LLAMA_CPP:?}"
  sp quantise gguf --model "$dir/bf16" --out "$dir/gguf" --name "$name" \
    --llama-cpp "$LLAMA_CPP" --quantize-binary "$LLAMA_CPP/build/bin/llama-quantize" \
    >"logs/$name-gguf.log" 2>&1
  # The bf16 GGUF is the converter's intermediate, not a published format.
  rm -f "$dir/gguf/$name-bf16.gguf"
  retry aws s3 sync "${s3[@]}" --only-show-errors --exclude "*.json" "$dir/gguf" "$remote/gguf"
  retry aws s3 sync "${s3[@]}" --only-show-errors --exclude "*" --include "*.json"     "$dir/gguf" "$remote/gguf"
fi
echo "=== ALL-DONE"
