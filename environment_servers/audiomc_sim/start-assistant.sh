#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch the audio-patched local vLLM server using an existing installation.
#
# Setup:
#   No manual exports are needed with this host's configured env.sh. For another
#   installation, prepare <NANO_VLLM_ROOT>/env.sh with the paths below. The script
#   sources that file automatically; it does not install vLLM or download weights.
#
# Installation settings:
#   NANO_VLLM_ROOT    Directory containing env.sh, the prepared .venv, server.pid
#                    and default traces directory. Used first to locate env.sh.
#                    Default: /home/yongqiangw/work/tmp/nemo_vllm
#   NANO_VLLM_SERVER  Directory containing the audio-patched serve.py entry point.
#                    Required after env.sh is sourced; no launcher default.
#   NANO_VLLM_MODEL   Hugging Face checkpoint directory to load into GPU memory.
#                    Required after env.sh is sourced; no launcher default.
#
#   The current host's env.sh assigns all three installation settings directly.
#   Those assignments replace pre-existing exports. To change them, edit env.sh
#   or select another installation's env.sh with NANO_VLLM_ROOT.
#
# Optional launch settings (inherited from the shell unless env.sh changes them):
#   PORT                 HTTP listening port on 127.0.0.1. Default: 8000.
#                        Set assistant.api_base in backend.yaml to the matching
#                        URL, for example http://127.0.0.1:8001/v1.
#   AUDIO_LIMIT          Maximum audio clips accepted in one model request.
#                        Default: 50. Set at least as high as collection max-turns
#                        because requests include the full user audio history.
#   NANO_VLLM_TRACE_DIR  Diagnostic trace directory used by the patched server.
#                        Default: <NANO_VLLM_ROOT>/traces.
#   CUDA_VISIBLE_DEVICES GPU visibility/selection, for example 0 or a GPU UUID.
#                        If unset, uses the runtime's normal GPU visibility.
#                        This launcher serves on one GPU (tensor parallelism 1).
#
# Runtime settings normally supplied by env.sh:
#   PATH / LD_LIBRARY_PATH   Select the prepared Python environment and libraries.
#   CUDA_HOME / CUDNN_PATH   CUDA toolkit and cuDNN installation locations.
#   NVRTC_PATH / CURAND_PATH CUDA runtime-compilation and random-number libraries.
#   PYTHONUNBUFFERED         Enables immediate Python log output when set to 1.
#   TOKENIZERS_PARALLELISM   Controls tokenizer threading; this host sets false.
#
# Examples (run from the repository root):
#   ./environment_servers/audiomc_sim/start-assistant.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8001 AUDIO_LIMIT=50 \
#     NANO_VLLM_TRACE_DIR=/path/to/traces \
#     ./environment_servers/audiomc_sim/start-assistant.sh
#
# Extra command-line arguments are appended to the vLLM arguments after `--`.
# Other serving limits below are fixed in this launcher, not environment options.
# server.pid records the launcher PID; exec replaces it with the Python process.
set -euo pipefail
NANO_VLLM_ROOT="${NANO_VLLM_ROOT:-/home/yongqiangw/work/tmp/nemo_vllm}"
source "$NANO_VLLM_ROOT/env.sh"
printf '%s\n' "$$" > "$NANO_VLLM_ROOT/server.pid"
export NANO_VLLM_TRACE_DIR="${NANO_VLLM_TRACE_DIR:-$NANO_VLLM_ROOT/traces}"
exec python "$NANO_VLLM_SERVER/serve.py" \
    --model "$NANO_VLLM_MODEL" --allow-untested-vllm \
    --host 127.0.0.1 --port "${PORT:-8000}" \
    --audio-limit "${AUDIO_LIMIT:-50}" --image-limit 0 --video-limit 0 \
    --max-num-seqs 32 --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 -- --enforce-eager \
    --no-enable-prefix-caching --mm-processor-cache-gb 0 "$@"
