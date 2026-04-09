# Search Agent Evaluation

Evaluation scripts for GAIA, frames, and BrowseComp-Plus benchmarks with web search tool calling.

## Benchmarks

| Benchmark | Directory | Description |
|---|---|---|
| GAIA + frames | `ASearcher/evaluation/` | Web search QA (pass@4, 4 seeds) |
| BrowseComp-Plus | `BrowseComp-Plus/search_agent/` | Document retrieval with local FAISS index |

## Supported Models

| Model Family | GAIA/frames Script | BrowseComp Script | Engine | Tool Parser | Reasoning Parser |
|---|---|---|---|---|---|
| Solar Open 100B | `run_oss_eval_slurm.sh` | `oss_client_slurm.sh` | vLLM | `solar_open` | `solar_open` |
| GLM-4.7 | `run_glm47_eval_sglang_slurm.sh` | `glm47_sglang_slurm.sh` | SGLang | `glm47` | `glm45` |
| Qwen3.5 (35B/122B/397B) | `run_qwen35_eval_vllm_slurm.sh` | `qwen35_vllm_slurm.sh` | vLLM / SGLang | `qwen3_coder` | `qwen3` |
| Qwen3 SFT checkpoints | `run_qwen3_sft_hermes_slurm.sh` | `qwen3_sft_hermes_slurm.sh` | vLLM / SGLang | `hermes` | `deepseek_r1` |

## Running GAIA / frames Evaluation

Scripts are in `ASearcher/evaluation/`. Each script launches an inference server, waits for it to be ready, then runs the eval client.

```bash
# Default model (uses MODEL_PATH in script)
sbatch ASearcher/evaluation/run_qwen35_eval_vllm_slurm.sh

# Override model path via env vars
MODEL_PATH=/path/to/checkpoint \
MODEL_NAME=my-model \
sbatch --export=ALL ASearcher/evaluation/run_qwen35_eval_vllm_slurm.sh
```

### Eval scripts

- **`oss_eval_async_clean.py`** — OpenAI Responses API (used by Solar Open with vLLM)
- **`oss_eval_async_clean_glm.py`** — Chat Completions API (used by GLM, Qwen3, Qwen3.5). Supports `--no-extra-body` for vLLM (skips SGLang-specific `chat_template_kwargs`).

### Output

Results are saved to `ASearcher/evaluation/output/{MODEL_NAME}/agent_eval_{MAX_GEN_TOKENS}/`. Per-sample JSON files enable resume on restart. Final aggregate is `aggregate_results_*.json`.

### API Keys

Serper and Jina API keys are loaded from `ASearcher/evaluation/eval_config.yaml`.

## Running BrowseComp-Plus Evaluation

Scripts are in `BrowseComp-Plus/search_agent/`. Each script launches an inference server, then runs the BrowseComp client against a local FAISS index.

```bash
# Default model
sbatch BrowseComp-Plus/search_agent/qwen35_vllm_slurm.sh

# Override model path
MODEL_PATH=/path/to/checkpoint \
sbatch --export=ALL BrowseComp-Plus/search_agent/qwen35_vllm_slurm.sh
```

### Clients

- **`oss_client.py`** — Responses API client (Solar Open)
- **`glm_zai_client.py`** — Chat Completions API client (GLM, Qwen3, Qwen3.5). Saves partial results on error instead of crashing.

### Output

Raw results saved to `BrowseComp-Plus/runs/{MODEL_NAME}/oss_client/run_*.json`. Supports resume — existing query IDs are skipped on restart.

### Scoring

After runs complete, score with the LLM judge:

```bash
sbatch --gres=gpu:8 --wrap="cd BrowseComp-Plus && source .venv/bin/activate && \
  python -u scripts_evaluation/evaluate_run.py --input_dir runs/{MODEL_NAME}/oss_client"
```

Results are saved to `BrowseComp-Plus/evals/{MODEL_NAME}/oss_client/evaluation_summary.json`.

## Qwen3 vs Qwen3.5 Tool Calling

These model families use different tool call formats:

| | Qwen3 | Qwen3.5 |
|---|---|---|
| Tool format | `<tool_call>{"name": ..., "arguments": ...}</tool_call>` (Hermes JSON) | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` (XML) |
| vLLM parser | `--tool-call-parser hermes` | `--tool-call-parser qwen3_coder` |
| SGLang parser | `--tool-call-parser hermes` | `--tool-call-parser qwen3_coder` |
| vLLM reasoning | `--reasoning-parser deepseek_r1` | `--reasoning-parser qwen3` |
| SGLang reasoning | `--reasoning-parser qwen3` | `--reasoning-parser qwen3` |

### SFT Checkpoint Notes

When evaluating SFT checkpoints based on Qwen3, ensure:
- `generation_config.json` has `eos_token_id: [151645, 151643]` (both `<|im_end|>` and `<|endoftext|>`)
- `config.json` has `eos_token_id: [151645, 151643]`
- `tokenizer_config.json` has `eos_token: "<|im_end|>"`
- The chat template includes `<think>\n` in the generation prompt (copy from the base Qwen3 model if missing)
