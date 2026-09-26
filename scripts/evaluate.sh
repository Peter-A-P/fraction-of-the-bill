#!/usr/bin/env bash
# One format of one fine-tune, served and run over held-out filings through the gateway.
# See docs/runbook.md, step 7.
#
#   scripts/evaluate.sh NAME FORMAT [SPLIT...]
#
# NAME is the run's name, as formats.sh wrote it under formats/NAME; FORMAT is bf16, awq,
# gptq, gguf-q8_0 or gguf-q4_k_m; SPLIT defaults to test_post_cutoff, the headline. The
# contamination gap is the same command with test_pre_cutoff, on bf16. Needs VOLUME_ID
# and DATACENTER with the S3 key, the gateway extra, vllm or llama-server on the PATH, and
# the build in data/build/full.
#
# The server is started with the command line `smallprint serve argv` prints, so the flags
# that decide a measurement are the tested ones; the gateway is given the served model as a
# self-hosted provider by `smallprint serve config`; and the accuracy run is the same
# `baseline run` every frontier number came from, at temperature 0. Each finished run goes
# to the volume under evals/, and a split already there is skipped.
set -euo pipefail

name=$1
fmt=$2
shift 2
splits=("${@:-test_post_cutoff}")
: "${VOLUME_ID:?}" "${DATACENTER:?}"
s3=(--region "$DATACENTER" --endpoint-url "https://s3api-${DATACENTER,,}.runpod.io/")
port=8000
served="$name-$fmt"
mkdir -p logs data/finetuned

case "$fmt" in
  gguf-*)
    weights="models/$name/gguf/$name-${fmt#gguf-}.gguf"
    [ -f "$weights" ] || aws s3 sync "${s3[@]}" --quiet "s3://$VOLUME_ID/formats/$name/gguf" "models/$name/gguf"
    ;;
  *)
    weights="models/$name/$fmt"
    [ -d "$weights" ] || aws s3 sync "${s3[@]}" --quiet "s3://$VOLUME_ID/formats/$name/$fmt" "$weights"
    ;;
esac

# KV_TOKENS sizes llama.cpp's shared cache to the card; vLLM sizes its own.
kv=()
if [[ $fmt == gguf-* && -n ${KV_TOKENS:-} ]]; then kv=(--kv-tokens "$KV_TOKENS"); fi
argv=$(uv run --no-sync smallprint serve argv --model "$weights" --run "$name" --format "$fmt" \
  --port "$port" "${kv[@]}")
echo "=== serving: $argv"
# exec, so that $! is the server itself: killing the subshell that started it left the
# server running and holding the card, and the next merge ran on the CPU (2026-09-24).
eval "exec $argv" >"logs/serve-$served.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null; wait "$server" 2>/dev/null || true' EXIT
# Both servers answer /health with 200 once the weights are loaded.
until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
  kill -0 "$server" 2>/dev/null || { echo "the server exited; see logs/serve-$served.log"; exit 1; }
  sleep 5
done

uv run --no-sync smallprint serve config --base-url "http://127.0.0.1:$port/v1"

for split in "${splits[@]}"; do
  out="data/finetuned/$served-$split"
  done_uri="s3://$VOLUME_ID/evals/$served-$split"
  if aws s3 ls "${s3[@]}" "$done_uri/summary.json" >/dev/null 2>&1; then
    echo "=== $served on $split already on the volume, skipped"
    continue
  fi
  echo "=== $served on $split started $(date -u +%FT%TZ)"
  uv run --no-sync smallprint baseline run --config boundary-served.yaml \
    --model "selfhosted/$served" --split "$split" --temperature 0 \
    --build-dir data/build/full --out "$out" --run-id "$served-$split" \
    >"logs/eval-$served-$split.log" 2>&1
  aws s3 sync "${s3[@]}" --quiet --exclude summary.json "$out" "$done_uri"
  aws s3 cp "${s3[@]}" --quiet "$out/summary.json" "$done_uri/summary.json"
  echo "=== $served on $split finished $(date -u +%FT%TZ)"
done
echo "=== ALL-DONE"
