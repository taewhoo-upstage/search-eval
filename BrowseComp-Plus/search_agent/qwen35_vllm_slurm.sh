#!/bin/bash
#SBATCH --job-name=qwen35-397b-browsecomp
#SBATCH --partition=normal
#SBATCH --nodelist=Slurm-GPU-Node-[1-49]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/qwen35-397b-browsecomp.log
#SBATCH --error=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/qwen35-397b-browsecomp.log

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
MODEL_PATH=/mnt/weka/post_training/checkpoints/Qwen3.5-397B-A17B
MODEL_NAME=Qwen3.5-397B-A17B
QUERY_FILE="${QUERY_FILE:-topics-qrels/queries.tsv}"

# Qwen3.5 recommended params (thinking mode)
TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=81920

BC_ROOT=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus

# ─── Environment ──────────────────────────────────────────────────────────────
source /mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh

export FI_PROVIDER=tcp
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ENGINE_READY_TIMEOUT_S=1800

mkdir -p "${BC_ROOT}/logs"

# ─── Launch vLLM server ──────────────────────────────────────────────────────
cd /mnt/weka/post_training/pt2-search-agent/evaluation
conda activate qwen35_35b

vllm serve "${MODEL_PATH}" \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --served-model-name "${MODEL_NAME}" \
  --gpu-memory-utilization 0.8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --generation-config vllm \
  > "${BC_ROOT}/logs/vllm_qwen35_397b.log" 2>&1 &

echo "Waiting for vLLM server..."
WAIT=0
until curl -s http://localhost:8000/health >/dev/null 2>&1; do
  sleep 5
  WAIT=$((WAIT+5))
  if [ $WAIT -ge 1800 ]; then echo "vLLM failed to start"; exit 1; fi
done
echo "vLLM ready after ${WAIT}s"

# ─── Run BrowseComp evaluation ───────────────────────────────────────────────
cd "${BC_ROOT}"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

python -u search_agent/glm_zai_client.py \
  --model "${MODEL_NAME}" \
  --model-url "http://localhost:8000/v1" \
  --api-key "EMPTY" \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --output-dir "runs/${MODEL_NAME}/oss_client" \
  --get-document \
  --searcher-type faiss \
  --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize
