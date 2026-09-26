#!/usr/bin/env bash
# The load test for one format of one fine-tune, on the card it is priced on. See
# docs/runbook.md, step 9.
#
#   scripts/loadtest.sh NAME FORMAT
#
# Needs what scripts/evaluate.sh needs, and the price of this card read that day, in
# GPU, PROVIDER, KIND, USD_PER_HOUR, CHECKED and SOURCE, which go into the record beside
# the timings, so a throughput can never be quoted without the rate it was bought at.
# KV_TOKENS sizes llama.cpp's shared cache to the card, as in evaluate.sh.
#
# The weights come from the volume, except bf16, which is merged here from the adapter
# (scripts/formats.sh keeps merged weights off the volume). The server is started with
# the tested command line, the gateway is given it with retries off, and `bench run`
# sweeps the concurrency levels through the gateway, streamed. Each finished load test
# goes to the volume under bench/, and one already there is skipped.
set -euo pipefail

name=$1
fmt=$2
: "${VOLUME_ID:?}" "${DATACENTER:?}"
: "${GPU:?}" "${PROVIDER:?}" "${KIND:?}" "${USD_PER_HOUR:?}" "${CHECKED:?}" "${SOURCE:?}"
s3=(--region "$DATACENTER" --endpoint-url "https://s3api-${DATACENTER,,}.runpod.io/")
port=8000
served="$name-$fmt"
out="data/bench/$served-$GPU"
done_uri="s3://$VOLUME_ID/bench/$served-$GPU"
mkdir -p logs data/bench

retry() {
  local wait
  for wait in 10 30 90 0; do
    "$@" && return 0
    if [ "$wait" -gt 0 ]; then sleep "$wait"; fi
  done
  return 1
}

if aws s3 ls "${s3[@]}" "$done_uri/bench.json" >/dev/null 2>&1; then
  echo "=== $served on $GPU already on the volume, skipped"
  exit 0
fi

case "$fmt" in
  bf16)
    weights="models/$name/bf16"
    if [ ! -f "$weights/merge.json" ]; then
      retry aws s3 sync "${s3[@]}" --only-show-errors "s3://$VOLUME_ID/adapters/$name" "runs/$name"
      uv run --no-sync smallprint quantise merge --run-dir "runs/$name" --out "$weights" \
        --dataset-dir data/train/full >"logs/$name-merge.log" 2>&1
    fi
    ;;
  gguf-*)
    weights="models/$name/gguf/$name-${fmt#gguf-}.gguf"
    [ -f "$weights" ] || retry aws s3 sync "${s3[@]}" --only-show-errors \
      "s3://$VOLUME_ID/formats/$name/gguf" "models/$name/gguf"
    ;;
  *)
    weights="models/$name/$fmt"
    [ -d "$weights" ] || retry aws s3 sync "${s3[@]}" --only-show-errors \
      "s3://$VOLUME_ID/formats/$name/$fmt" "$weights"
    ;;
esac

kv=()
if [[ $fmt == gguf-* && -n ${KV_TOKENS:-} ]]; then kv=(--kv-tokens "$KV_TOKENS"); fi
argv=$(uv run --no-sync smallprint serve argv --model "$weights" --run "$name" --format "$fmt" \
  --port "$port" "${kv[@]}")
echo "=== serving: $argv"
eval "exec $argv" >"logs/serve-$served-$GPU.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null; wait "$server" 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$port/health" >/dev/null; do
  kill -0 "$server" 2>/dev/null || { echo "the server exited; see logs/serve-$served-$GPU.log"; exit 1; }
  sleep 5
done

uv run --no-sync smallprint serve config --base-url "http://127.0.0.1:$port/v1"
echo "=== $served on $GPU started $(date -u +%FT%TZ)"
uv run --no-sync smallprint bench run --model "selfhosted/$served" --out "$out" \
  --gpu "$GPU" --provider "$PROVIDER" --kind "$KIND" --usd-per-hour "$USD_PER_HOUR" \
  --checked "$CHECKED" --source "$SOURCE" --config boundary-served.yaml \
  --build-dir data/build/full >"logs/bench-$served-$GPU.log" 2>&1
retry aws s3 sync "${s3[@]}" --only-show-errors --exclude bench.json "$out" "$done_uri"
retry aws s3 cp "${s3[@]}" --only-show-errors "$out/bench.json" "$done_uri/bench.json"
echo "=== $served on $GPU finished $(date -u +%FT%TZ)"
