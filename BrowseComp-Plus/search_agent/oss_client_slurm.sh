#!/bin/bash
#SBATCH --job-name=solar-open-eval
#SBATCH --partition=omni
#SBATCH --nodelist=Slurm-GPU-Node-[75-90]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/solar-open-100b-ckpt50_oss_client_getdoc.log
#SBATCH --error=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/solar-open-100b-ckpt50_oss_client_getdoc.log

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/weka/post_training/checkpoints/Solar-Open-100B}"
MODEL_NAME="${MODEL_PATH##*/}"
QUERY_FILE="${QUERY_FILE:-topics-qrels/queries.tsv}"

BC_ROOT=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus

source /mnt/weka/post_training_tmp/pt2-search-agent/miniconda3/etc/profile.d/conda.sh

cd /mnt/weka/post_training/pt2-search-agent/evaluation
source .venv/bin/activate

export FI_PROVIDER=tcp
export VLLM_CACHE_ROOT=/mnt/weka/post_training_tmp/pt2-search-agent/vllm
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p ./logs

vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.8 \
  --enable-auto-tool-choice \
  --tool-call-parser solar_open \
  --reasoning-parser solar_open \
  --logits-processors vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor \
  --logits-processors vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor \
  > "./logs/vllm_${MODEL_NAME}.log" 2>&1 &

echo "Waiting for vLLM server..."
WAIT=0
until curl -s http://localhost:8000/health >/dev/null 2>&1; do
  sleep 5
  WAIT=$((WAIT+5))
  if [ $WAIT -ge 600 ]; then echo "vLLM failed to start"; exit 1; fi
done
echo "vLLM ready"

conda deactivate 2>/dev/null || true

cd "${BC_ROOT}"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

python -u search_agent/oss_client.py \
  --model "${MODEL_PATH}" \
  --query "${QUERY_FILE}" \
  --output-dir "runs/${MODEL_NAME}/oss_client" \
  --get-document \
  --searcher-type faiss \
  --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --model-name "Qwen/Qwen3-Embedding-8B" \
  --normalize
