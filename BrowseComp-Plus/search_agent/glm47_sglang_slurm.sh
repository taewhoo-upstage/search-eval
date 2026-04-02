#!/bin/bash
#SBATCH --job-name=glm47-browsecomp
#SBATCH --partition=omni
#SBATCH --nodelist=Slurm-GPU-Node-[75-90]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/glm47-browsecomp.log
#SBATCH --error=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/glm47-browsecomp.log

set -euo pipefail

# ─── Model configuration ────────────────────────────────────────────────────
MODEL_PATH=/mnt/weka/post_training/checkpoints/GLM-4.7   # Full BF16 (8x H200)
MODEL_NAME=glm-4.7
TP_SIZE=8

# GLM 4.7 official eval params (default / agentic tasks)
TEMPERATURE=1.0
TOP_P=0.95
MAX_TOKENS=131072

# ─── Environment ─────────────────────────────────────────────────────────────
source /mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh

cd /mnt/weka/post_training/pt2-search-agent/evaluation
conda activate glm47

export FI_PROVIDER=tcp
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p ./logs

# ─── Launch SGLang server ────────────────────────────────────────────────────
# SGLang dev branch required for GLM 4.7 support.
# Preserved thinking (clear_thinking=false) is only supported in SGLang.
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
  --allow-auto-truncate \
  --served-model-name "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  > ./logs/sglang_glm47.log 2>&1 &

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
cd /mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

python -u search_agent/glm_zai_client.py \
  --model "${MODEL_NAME}" \
  --model-url "http://localhost:8000/v1" \
  --api-key "EMPTY" \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --output-dir "runs/glm-4.7/oss_client" \
  --get-document \
  --searcher-type faiss \
  --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize
