#!/bin/bash
#SBATCH --job-name=qwen35-35b-gaia-frames
#SBATCH --partition=normal
#SBATCH --nodelist=Slurm-GPU-Node-[1-49]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/ASearcher/logs/qwen35-35b-gaia-frames.log
#SBATCH --error=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/ASearcher/logs/qwen35-35b-gaia-frames.log
#
# Prerequisites:
#   conda create -n qwen35 python=3.12
#   conda activate qwen35_35b
#   uv pip install vllm --torch-backend=auto

set -euo pipefail

# ─── Kill any existing GPU worker processes ──────────────────────────────────
PIDS=$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader |
  awk -F"," '/Worker_TP|VLLM/ { gsub(/ /, "", $1); print $1 }' |
  tr '\n' ' '
)

if [ -z "$PIDS" ]; then
    echo "No existing worker processes found locally"
else
    echo "Found PIDs locally: $PIDS"
    sudo kill -9 $PIDS
    echo "Killed. Waiting for cleanup..."
    sleep 20
    echo "Done."
fi

# ─── Model configuration ────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/mnt/weka/post_training/checkpoints/Qwen3.5-35B-A3B}"
MODEL_NAME="${MODEL_NAME:-${MODEL_PATH##*/}}"
MODEL_URL=http://localhost:8000/v1

# Qwen3.5 recommended params (thinking mode)
TEMPERATURE=0.6
TOP_P=0.95
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-81920}"

# ─── Paths ────────────────────────────────────────────────────────────────────
EVAL_ROOT=/mnt/weka/post_training/pt2-search-agent/evaluation
SCRIPT_DIR="${EVAL_ROOT}/orig_repo/ASearcher/evaluation"
PROJECT_ROOT="${EVAL_ROOT}/orig_repo/ASearcher"

DATA_DIR="${PROJECT_ROOT}/data"
LOG_DIR="${PROJECT_ROOT}/logs"
OUTPUT_DIR="${SCRIPT_DIR}/output/${MODEL_NAME}"

# ─── Eval config ──────────────────────────────────────────────────────────────
DATA_NAMES=GAIA,frames
AGENT_TYPE=oss-tool-calling
PROMPT_TYPE=tool-calling
SEARCH_CLIENT_TYPE=async-web-search-access

# ─── Web search API keys ─────────────────────────────────────────────────────
# Keys are loaded from eval_config.yaml by config_loader.py

# ─── Environment ──────────────────────────────────────────────────────────────
source /mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh

export FI_PROVIDER=tcp
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=1800

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ─── Launch vLLM server ──────────────────────────────────────────────────────
cd "${EVAL_ROOT}"
conda activate qwen35_35b

vllm serve "${MODEL_PATH}" \
  --port 8000 \
  --tensor-parallel-size 8 \
  ${MAX_MODEL_LEN:+--max-model-len ${MAX_MODEL_LEN}} \
  --served-model-name "${MODEL_NAME}" \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  > "${LOG_DIR}/vllm_qwen35_35b.log" 2>&1 &

# ─── Wait for vLLM to be ready ───────────────────────────────────────────────
echo "Waiting for vLLM server at ${MODEL_URL} ..."
WAIT=0
until curl -s http://localhost:8000/health >/dev/null 2>&1; do
  sleep 5
  WAIT=$((WAIT + 5))
  if [ "${WAIT}" -ge 1800 ]; then
    echo "ERROR: vLLM server did not become ready within 30 minutes"
    exit 1
  fi
done
echo "vLLM ready after ${WAIT}s"

# ─── Run evaluation (Chat Completions API) ────────────────────────────────────
echo "MODEL PATH:  ${MODEL_PATH}"
echo "OUTPUT DIR:  ${OUTPUT_DIR}"
echo "Temperature: ${TEMPERATURE}"
echo "top_p:       ${TOP_P}"

cd "${SCRIPT_DIR}"

TOKENIZERS_PARALLELISM=false \
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
python3 oss_eval_async_clean_glm.py \
    --data_names          "${DATA_NAMES}" \
    --model_name_or_path  "${MODEL_NAME}" \
    --model-url           "${MODEL_URL}" \
    --output_dir          "${OUTPUT_DIR}" \
    --prompt_type         "${PROMPT_TYPE}" \
    --agent-type          "${AGENT_TYPE}" \
    --data_dir            "${DATA_DIR}" \
    --split               test \
    --search-client-type  "${SEARCH_CLIENT_TYPE}" \
    --max-tokens-per-call "${MAX_GEN_TOKENS}" \
    --top-k-docs          5 \
    --max-iterations      100 \
    --n_sampling          1 \
    --temperature         "${TEMPERATURE}" \
    --top_p               "${TOP_P}" \
    --start               0 \
    --end                 -1 \
    --seed                1 \
    --parallel-mode       seed \
    --concurrent          64 \
    --pass-at-k           4 \
    --get-document \
    --use-jina \
    --query-template      QUERY_TEMPLATE \
    --no-extra-body \
    "$@"
