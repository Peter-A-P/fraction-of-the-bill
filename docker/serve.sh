#!/bin/sh
# Start vLLM on MODEL (a Hugging Face repository or a path), named after RUN and FORMAT.
set -eu
: "${MODEL:?set MODEL}" "${RUN:?set RUN}" "${FORMAT:?set FORMAT, one of bf16, awq, gptq}"
argv=$(/opt/smallprint/bin/smallprint serve argv --model "$MODEL" --run "$RUN" --format "$FORMAT" \
  --port "${PORT:-8000}")
echo "$argv"
exec sh -c "exec $argv --host 0.0.0.0"
