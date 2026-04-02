#!/bin/bash
#SBATCH --job-name=glm47-gaia-frames
#SBATCH --partition=normal
#SBATCH --nodelist=Slurm-GPU-Node-[75-90]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/ASearcher/logs/glm47-gaia-frames.log
#SBATCH --error=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/ASearcher/logs/glm47-gaia-frames.log

set -euo pipefail

# ─── Kill any existing GPU worker processes ──────────────────────────────────
PIDS=$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader |
  awk -F"," '/Worker_TP|sglang/ { gsub(/ /, "", $1); print $1 }' |
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
MODEL_PATH=/mnt/weka/post_training/checkpoints/GLM-4.7   # Full BF16 (8x H200)
MODEL_NAME=GLM-4.7
MODEL_URL=http://localhost:8000/v1
TP_SIZE=8

# GLM 4.7 official eval params (default / agentic tasks)
TEMPERATURE=1.0
TOP_P=0.95
MAX_GEN_TOKENS=131072

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

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ─── Launch SGLang server (glm47 conda env) ──────────────────────────────────
# SGLang dev branch required for GLM 4.7 support.
# Preserved thinking (clear_thinking=false) is only supported in SGLang.
cd "${EVAL_ROOT}"
conda activate glm47

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size "${TP_SIZE}" \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.8 \
  --served-model-name "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  > "${LOG_DIR}/sglang_glm47.log" 2>&1 &

# ─── Wait for SGLang to be ready ──────────────────────────────────────────────
echo "Waiting for SGLang server at ${MODEL_URL} ..."
WAIT=0
until curl -s http://localhost:8000/health >/dev/null 2>&1; do
  sleep 5
  WAIT=$((WAIT + 5))
  if [ "${WAIT}" -ge 1800 ]; then
    echo "ERROR: SGLang server did not become ready within 30 minutes"
    exit 1
  fi
done
echo "SGLang ready after ${WAIT}s"

# ─── Run evaluation (GLM variant — Chat Completions API) ─────────────────────
# Uses oss_eval_async_clean_glm.py which:
#   - Uses client.chat.completions.create() (compatible with SGLang)
#   - Passes extra_body with chat_template_kwargs for preserved thinking
#   - Web search via Serper API + Jina for page browsing
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
    "$@"
