#!/bin/bash
#SBATCH --job-name=qwen35-browsecomp
#SBATCH --partition=preemptible
#SBATCH --nodelist=Slurm-GPU-Node-[1-49]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/qwen35-browsecomp.log
#SBATCH --error=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/qwen35-browsecomp.log

set -euo pipefail

# ─── Model configuration ────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/mnt/weka/post_training/checkpoints/Qwen3.5-397B-A17B}"
MODEL_NAME="${MODEL_PATH##*/}"
QUERY_FILE="${QUERY_FILE:-topics-qrels/queries.tsv}"
TP_SIZE=8

TEMPERATURE=0.6
TOP_P=0.95
MAX_TOKENS=131072

BC_ROOT=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus

# ─── Environment ──────────────────────────────────────────────────────────────
source /mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh

cd /mnt/weka/post_training/pt2-search-agent/evaluation
conda activate qwen35_35b

export FI_PROVIDER=tcp
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p "${BC_ROOT}/logs"

# ─── Launch SGLang server ────────────────────────────────────────────────────
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size "${TP_SIZE}" \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mem-fraction-static 0.8 \
  --allow-auto-truncate \
  --served-model-name "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  > "${BC_ROOT}/logs/sglang_${MODEL_NAME}.log" 2>&1 &

echo "Waiting for SGLang server..."
WAIT=0
until curl -s http://localhost:8000/health >/dev/null 2>&1; do
  sleep 5
  WAIT=$((WAIT+5))
  if [ $WAIT -ge 1800 ]; then echo "SGLang failed to start"; exit 1; fi
done
echo "SGLang ready after ${WAIT}s"

conda deactivate 2>/dev/null || true

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
