#!/bin/bash
#SBATCH --job-name=solar-open-eval
#SBATCH --partition=omni
#SBATCH --nodelist=Slurm-GPU-Node-[88]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/ASearcher/logs/solar-open-eval.log
#SBATCH --error=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/ASearcher/logs/solar-open-eval.log

set -euo pipefail

# ─── Kill any existing VLLM worker processes ──────────────────────────────────
PIDS=$(
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader |
  awk -F"," '/VLLM::Worker_TP/ { gsub(/ /, "", $1); print $1 }' |
  tr '\n' ' '
)

if [ -z "$PIDS" ]; then
    echo "No VLLM::Worker_TP processes found locally"
else
    echo "Found PIDs locally: $PIDS"
    sudo kill -9 $PIDS
    echo "Killed. Waiting for vLLM processes cleanup..."
    sleep 20
    echo "Done."
fi

# ─── Paths ────────────────────────────────────────────────────────────────────
EVAL_ROOT=/mnt/weka/post_training/pt2-search-agent/evaluation
SCRIPT_DIR="${EVAL_ROOT}/orig_repo/ASearcher/evaluation"
PROJECT_ROOT="${EVAL_ROOT}/orig_repo/ASearcher"

MODEL_PATH=/mnt/weka/post_training/checkpoints/Solar-Open-100B
MODEL_NAME="${MODEL_PATH##*/}"
MODEL_URL=http://localhost:8000/v1

DATA_DIR="${PROJECT_ROOT}/data"
LOG_DIR="${PROJECT_ROOT}/logs"
OUTPUT_DIR="${SCRIPT_DIR}/output/${MODEL_NAME}"

# ─── Eval config ──────────────────────────────────────────────────────────────
MAX_GEN_TOKENS=4096
DATA_NAMES=GAIA,frames
AGENT_TYPE=oss-tool-calling
PROMPT_TYPE=tool-calling
SEARCH_CLIENT_TYPE=async-search-access
temperature=0.8
top_p=0.95
top_k=50

RETRIEVER_PORT=5201
RETRIEVER_ADDR=$(python3 -c "import socket; print(socket.gethostbyname(socket.gethostname()))")

# ─── Environment ──────────────────────────────────────────────────────────────
source /mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh

export FI_PROVIDER=tcp
export VLLM_CACHE_ROOT=/mnt/weka/post_training_tmp/pt2-search-agent/vllm
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# ─── 1. Patch eval_config.yaml with retriever address/port ───────────────────
# Done before any server starts so config_loader reads the right values.
python3 - <<EOF
import yaml, pathlib
cfg_path = pathlib.Path("${SCRIPT_DIR}/eval_config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg.setdefault("settings", {}).setdefault("local_server", {})
cfg["settings"]["local_server"]["address"] = "${RETRIEVER_ADDR}"
cfg["settings"]["local_server"]["port"]    = "${RETRIEVER_PORT}"
cfg_path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
print(f"eval_config.yaml updated: local_server = ${RETRIEVER_ADDR}:${RETRIEVER_PORT}")
EOF

# ─── 2. Launch retriever server (conda retriever, background) ─────────────────
conda activate retriever

cd "${PROJECT_ROOT}"
bash scripts/launch_local_server.sh "${RETRIEVER_PORT}" "${LOG_DIR}/retriever_addr.txt" \
    > "${LOG_DIR}/retriever.log" 2>&1 &
RETRIEVER_PID=$!
echo "Retriever PID: ${RETRIEVER_PID}"

conda deactivate

# ─── 3. Wait for retriever to be ready ────────────────────────────────────────
echo "Waiting for retriever at ${RETRIEVER_ADDR}:${RETRIEVER_PORT} ..."
WAIT=0
until curl -s http://${RETRIEVER_ADDR}:${RETRIEVER_PORT}/retrieve >/dev/null 2>&1; do
  sleep 5
  WAIT=$((WAIT + 5))
  if [ "${WAIT}" -ge 1200 ]; then
    echo "ERROR: retriever did not become ready within 20 minutes"
    exit 1
  fi
done
echo "Retriever ready after ${WAIT}s"

# ─── 4. Launch vLLM server (eval venv, background) ────────────────────────────
# evaluation/.venv contains both vllm and all eval client packages,
# so no environment switch is needed before or after this step.
cd "${EVAL_ROOT}"
source .venv/bin/activate

vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.7 \
  --enable-auto-tool-choice \
  --tool-call-parser solar_open \
  --reasoning-parser solar_open \
  --logits-processors vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor \
  --logits-processors vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor \
  > "${LOG_DIR}/vllm_solar_open.log" 2>&1 &

# ─── 5. Wait for vLLM to be ready ─────────────────────────────────────────────
echo "Waiting for vLLM server at ${MODEL_URL} ..."
WAIT=0
until curl -s http://localhost:8000/health >/dev/null 2>&1; do
  sleep 5
  WAIT=$((WAIT + 5))
  if [ "${WAIT}" -ge 600 ]; then
    echo "ERROR: vLLM server did not become ready within 10 minutes"
    exit 1
  fi
done
echo "vLLM ready after ${WAIT}s"

# ─── 6. Run evaluation (same eval venv, no switch needed) ─────────────────────
echo "MODEL PATH:  ${MODEL_PATH}"
echo "OUTPUT DIR:  ${OUTPUT_DIR}"
echo "Temperature: ${temperature}"
echo "top_p:       ${top_p}"
echo "top_k:       ${top_k}"

cd "${SCRIPT_DIR}"

TOKENIZERS_PARALLELISM=false \
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
python3 oss_eval_async_clean.py \
    --data_names          "${DATA_NAMES}" \
    --model_name_or_path  "${MODEL_PATH}" \
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
    --reasoning-effort    high \
    --n_sampling          1 \
    --temperature         "${temperature}" \
    --top_p               "${top_p}" \
    --top_k               "${top_k}" \
    --start               0 \
    --end                 -1 \
    --seed                1 \
    --parallel-mode       seed \
    --concurrent          64 \
    --pass-at-k           4 \
    --get-document \
    --query-template      QUERY_TEMPLATE \
    "$@"
