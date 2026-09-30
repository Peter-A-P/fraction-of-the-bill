# The served fine-tune: vLLM started with the command line `smallprint serve argv` prints,
# the one every number in this project was measured with (docs/serving.md).
#
#   docker build -f docker/Dockerfile -t smallprint-serve .
#   docker run --gpus all -p 8000:8000 -e HF_TOKEN \
#     -e MODEL=Peter-A-P/smallprint-7b-gptq -e RUN=7b-r64-lr1e-4-nall-s0-e1 -e FORMAT=gptq \
#     smallprint-serve
#
# vLLM at the version the load tests ran on. smallprint gets its own Python beside it,
# because it needs 3.13 and the image's vLLM runs on the image's own Python; it is used
# only to print the command line, so the flags that decide a measurement come from the
# tested code and not from a copy of them here. GGUF is not served from this image:
# llama.cpp is the format for one machine, not for a server (docs/serving.md).
FROM vllm/vllm-openai:v0.30.0

RUN pip install --no-cache-dir uv==0.9.* \
    && uv python install 3.13 \
    && uv venv -p 3.13 /opt/smallprint
COPY pyproject.toml uv.lock README.md LICENSE /opt/src/
COPY smallprint /opt/src/smallprint
RUN uv pip install --python /opt/smallprint/bin/python --no-cache /opt/src
COPY docker/serve.sh /usr/local/bin/smallprint-serve
RUN chmod +x /usr/local/bin/smallprint-serve

ENV PORT=8000
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/smallprint-serve"]
