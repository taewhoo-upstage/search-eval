#!/bin/bash
#SBATCH --job-name=solar-open-eval
#SBATCH --partition=omni
#SBATCH --nodelist=Slurm-GPU-Node-[75-90]
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --output=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/solar-open-100b-oss_client_getdoc_eval.log
#SBATCH --error=/mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus/logs/solar-open-100b-oss_client_getdoc_eval.log

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

cd /mnt/weka/post_training/pt2-search-agent/evaluation/orig_repo/BrowseComp-Plus
source .venv/bin/activate

export FI_PROVIDER=tcp
export VLLM_CACHE_ROOT=/mnt/weka/post_training_tmp/pt2-search-agent/vllm
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p ./logs

python -u scripts_evaluation/evaluate_run.py --input_dir runs/solar-open-100b/oss_client
