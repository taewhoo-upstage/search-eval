#!/usr/bin/env python3
"""
Tool-calling evaluation — drop-in replacement for search_eval_async.py.

Differs from search_eval_async.py in exactly two ways:
  1. LLM backend: sglang  →  vLLM via OpenAI Responses API
  2. Agent:       XML-span parsing (AsearcherAgent / SearchR1Agent)
              →  OpenAI native tool-calling

Prompt templates and tool definitions mirror BrowseComp-Plus/search_agent/oss_client.py.
Everything else is identical: --search-client-type, data loading, pass@k,
metrics (F1/EM/CEM), and LLM-as-judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
import openai
from tqdm import tqdm
from transformers import AutoTokenizer

import evaluate as eval_metrics
from config_loader import load_config_and_set_env
from utils import set_seed, prepare_data, load_jsonl

try:
    from prettytable import PrettyTable
    PRETTYTABLE_AVAILABLE = True
except ImportError:
    PRETTYTABLE_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt templates  (copied from BrowseComp-Plus/search_agent/prompts.py)
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_TEMPLATE = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search and browse tools provided. Please perform reasoning and use the tools step by step, in an interleaved manner. You may use the search and browse tools multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer.}}
Exact Answer: {{your succinct, final answer}}
""".strip()

QUERY_TEMPLATE_NO_GET_DOCUMENT = """
You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.

Question: {Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer.}}
Exact Answer: {{your succinct, final answer}}
""".strip()

QUERY_TEMPLATES = {
    "QUERY_TEMPLATE":                          QUERY_TEMPLATE,
    "QUERY_TEMPLATE_NO_GET_DOCUMENT":          QUERY_TEMPLATE_NO_GET_DOCUMENT,
}


def format_query(question: str, query_template: str) -> str:
    tmpl = QUERY_TEMPLATES.get(query_template)
    if tmpl is None:
        raise ValueError(f"Unknown query_template: {query_template!r}. "
                         f"Choose from: {list(QUERY_TEMPLATES)}")
    return tmpl.format(Question=question)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool handlers  (one per search-client-type)
#
#  Tool names and parameter schemas mirror SearchToolHandler in
#  BrowseComp-Plus/search_agent/oss_client.py.
# ═══════════════════════════════════════════════════════════════════════════════

class LocalSearchToolHandler:
    """
    Wraps AsyncSearchBrowserClient for synchronous OpenAI tool-calling.

    Tool names and parameter schemas mirror SearchToolHandler in
    BrowseComp-Plus/search_agent/oss_client.py:
      - ``local_knowledge_base_retrieval``  (user_query)
      - ``get_document``                    (docid → fetched as URL via access_async)
    """

    def __init__(self, search_client, k: int = 5, include_get_document: bool = True):
        self.search_client = search_client
        self.k = k
        self.include_get_document = include_get_document

    def get_tool_definitions(self) -> list:
        tools = [
            {
                "type": "function",
                "name": "local_knowledge_base_retrieval",
                "description": (
                    f"Search the local knowledge base for documents relevant to the query. "
                    f"Returns the top {self.k} results with their docids and text snippets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "Query to search the local knowledge base for relevant information",
                        }
                    },
                    "required": ["user_query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]
        if self.include_get_document:
            tools.append({
                "type": "function",
                "name": "get_document",
                "description": "Retrieve the full text of a document by its URL (used as docid).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {
                            "type": "string",
                            "description": "Document URL to retrieve",
                        }
                    },
                    "required": ["docid"],
                    "additionalProperties": False,
                },
                "strict": True,
            })
        return tools

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "local_knowledge_base_retrieval":
            return self._search(arguments["user_query"])
        if tool_name == "get_document":
            return self._get_document(arguments["docid"])
        raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str) -> str:
        req = {"queries": [query], "topk": self.k, "return_scores": False}
        # Bridge async → sync; AsyncSearchBrowserClient opens a fresh
        # aiohttp session per call so asyncio.run() is safe here.
        result = asyncio.run(self.search_client.query_async(req))
        if not result:
            return json.dumps([])
        first = result[0]
        docs = first.get("documents", []) or []
        urls = first.get("urls", []) or []
        items = []
        for doc, url in zip(docs, urls):
            items.append({"docid": url, "score": None, "snippet": doc})
        return json.dumps(items, indent=2)

    def _get_document(self, docid: str) -> str:
        # docid is the URL; fetch via the server's access endpoint
        result = asyncio.run(self.search_client.access_async([docid]))
        if not result:
            return json.dumps({"error": f"Document '{docid}' not found"})
        page = (result[0] or {}).get("page", "") or ""
        return json.dumps({"docid": docid, "text": page}, indent=2)


class WebSearchToolHandler:
    """
    Synchronous web search using Serper + optional Jina.

    Tools exposed to the model:
      - ``search``  (queries: list[str] → Serper, returns title/url/snippet per query)
      - ``browse``  (url, query → Jina / requests, returns page text)
    """

    SEARCH_TOOL = {
        "type": "function",
        "name": "search",
        "description": "Perform web search queries. Provide an array 'queries'. Returns (title,url,snippet) for each query.",
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of search query strings.",
                }
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    BROWSE_TOOL = {
        "type": "function",
        "name": "browse",
        "description": "Extract specific information from a webpage at 'url' for a given 'query'.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Target URL to browse."},
                "query": {"type": "string", "description": "Question to answer based on the page content."},
            },
            "required": ["url", "query"],
            "additionalProperties": False,
        },
        "strict": True,
    }

    def __init__(self, k: int = 5, use_jina: bool = True, include_get_document: bool = True):
        from config_loader import get_api_key
        self.serper_api_key = get_api_key("serper_api_key") or os.environ.get("SERPER_API_KEY", "")
        self.jina_api_key = get_api_key("jina_api_key") or os.environ.get("JINA_API_KEY", "")
        self.k = k
        self.use_jina = use_jina and bool(self.jina_api_key)
        self.include_get_document = include_get_document
        # Per-thread API usage tracking (thread-local to avoid races with concurrent=64)
        self._thread_local = threading.local()

    def reset_api_usage(self):
        self._thread_local.api_usage = {"serper_queries": 0, "jina_requests": 0, "jina_total_chars": 0}

    def _usage(self) -> dict:
        if not hasattr(self._thread_local, "api_usage"):
            self._thread_local.api_usage = {"serper_queries": 0, "jina_requests": 0, "jina_total_chars": 0}
        return self._thread_local.api_usage

    def get_api_usage(self) -> dict:
        usage = dict(self._usage())
        usage["jina_estimated_tokens"] = usage["jina_total_chars"] // 4
        return usage

    def get_tool_definitions(self) -> list:
        tools = [self.SEARCH_TOOL]
        if self.include_get_document:
            tools.append(self.BROWSE_TOOL)
        return tools

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        import requests
        if tool_name == "search":
            if not self.serper_api_key:
                return json.dumps({"error": "SERPER_API_KEY not configured"})
            results = []
            queries = arguments.get("queries", [])
            self._usage()["serper_queries"] += len(queries)
            for query in queries:
                try:
                    resp = requests.post(
                        "https://google.serper.dev/search",
                        headers={"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"},
                        json={"q": query[:2000], "num": self.k},
                        timeout=20,
                    )
                    resp.raise_for_status()
                    items = [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("link", ""),
                            "snippet": r.get("snippet", ""),
                        }
                        for r in resp.json().get("organic", [])[:self.k]
                    ]
                    results.append({"query": query, "results": items})
                except Exception as e:
                    results.append({"query": query, "error": str(e)})
            return json.dumps(results, indent=2)

        if tool_name == "browse":
            url = arguments["url"]
            try:
                if self.use_jina:
                    resp = requests.get(
                        f"https://r.jina.ai/{url}",
                        headers={"Authorization": f"Bearer {self.jina_api_key}"},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        self._usage()["jina_requests"] += 1
                        self._usage()["jina_total_chars"] += len(resp.text)
                    text = resp.text[:25000] if resp.status_code == 200 else f"HTTP {resp.status_code}"
                else:
                    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                    text = resp.text[:25000]
                return json.dumps({"url": url, "query": arguments.get("query", ""), "text": text}, indent=2)
            except Exception as e:
                return json.dumps({"error": str(e)})

        raise ValueError(f"Unknown tool: {tool_name}")


def make_tool_handler(search_client_type: str, args):
    """Return the right tool handler for the given search-client-type."""
    include_get_document = args.get_document
    if search_client_type == "async-web-search-access":
        print("Tool handler: WebSearchToolHandler (Serper + Jina)")
        return WebSearchToolHandler(
            k=args.top_k_docs,
            use_jina=args.use_jina,
            include_get_document=include_get_document,
        )
    else:
        # Local RAG server
        from tools.search_utils import make_search_client
        search_client = make_search_client(search_client_type, args.use_jina, args.jina_api_key)
        print(f"Tool handler: LocalSearchToolHandler ({search_client_type})")
        return LocalSearchToolHandler(
            search_client=search_client,
            k=args.top_k_docs,
            include_get_document=include_get_document,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool-calling conversation loop  (from oss_client.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_conversation_with_tools(
    client: openai.OpenAI,
    initial_request: dict,
    tool_handler,
    max_iterations: int = 50,
    verbose: bool = False,
) -> tuple[list, dict, str]:
    """Drive a Responses-API tool-calling conversation to completion.

    Returns (messages, tool_usage, status).
    """
    tool_usage: dict[str, int] = {}
    messages = list(initial_request["input"])

    for iteration in range(1, max_iterations + 1):
        t_iter = time.time()
        try:
            response = client.responses.create(**{**initial_request, "input": messages})
        except Exception as e:
            print(f"  [iter {iteration}] API error ({time.time()-t_iter:.1f}s): {e}", flush=True)
            continue

        elapsed_iter = time.time() - t_iter
        output = response.model_dump(mode="python")["output"]
        output_types = [item["type"] for item in output]
        usage = getattr(response, "usage", None)
        usage_str = ""
        if usage:
            usage_str = f" | tokens: in={getattr(usage, 'input_tokens', '?')} out={getattr(usage, 'output_tokens', '?')}"
        print(f"  [iter {iteration}] {elapsed_iter:.1f}s | output_types={output_types}{usage_str}", flush=True)
        print(f"  [iter {iteration}] output: {json.dumps(output, ensure_ascii=False, default=str)}", flush=True)

        messages.extend(output)

        # Reasoning-only turn → keep looping
        if output and output[-1]["type"] == "reasoning":
            messages.pop()
            continue

        function_calls = [item for item in output if item["type"] == "function_call"]
        if not function_calls:
            return messages, tool_usage, "completed"

        new_messages = messages.copy()
        for tc in function_calls:
            try:
                result = tool_handler.execute_tool(tc["name"], json.loads(tc["arguments"]))
                tool_usage[tc["name"]] = tool_usage.get(tc["name"], 0) + 1
            except Exception as e:
                result = f"Error executing {tc['name']}: {e}"
            new_messages.append({
                "type": "function_call_output",
                "call_id": tc["call_id"],
                "output": result,
            })
        messages = new_messages

    return messages, tool_usage, "incomplete"


# ═══════════════════════════════════════════════════════════════════════════════
#  Answer extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_answer(messages: list) -> str:
    """Extract the predicted answer from the final model message.

    Matches the format requested by QUERY_TEMPLATE:
      Exact Answer: {answer}
    Falls back to <answer>...</answer> then full text.
    """
    for item in reversed(messages):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        parts = [
            str(p.get("text", ""))
            for p in (item.get("content") or [])
            if isinstance(p, dict) and p.get("type") == "output_text"
        ]
        text = "\n".join(parts).strip()
        if not text:
            continue
        m = re.search(r"Exact Answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-item evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_metric(fn, answer: str, gt: str) -> float:
    result = fn(answer, gt)
    if isinstance(result, tuple):
        return float(result[1] if result[0] is None else result[0])
    return float(result)


def process_single_item(
    client: openai.OpenAI,
    tool_handler,
    item: dict,
    args,
    out_dir: str,
) -> dict:
    """Run one question through the tool-calling agent, score it, and save."""

    user_prompt = format_query(item["question"], args.query_template)

    request: dict = {
        "model": args.model_name_or_path,
        "max_output_tokens": args.max_tokens_per_call,
        "input": [{"role": "user", "content": user_prompt}],
        "tools": tool_handler.get_tool_definitions(),
        "truncation": "auto",
    }
    if args.reasoning_effort and args.reasoning_effort != "none":
        request["reasoning"] = {"effort": args.reasoning_effort, "summary": "detailed"}
    if args.temperature > 0:
        request["temperature"] = args.temperature

    if hasattr(tool_handler, "reset_api_usage"):
        tool_handler.reset_api_usage()

    t0 = time.time()
    messages, tool_usage, status = run_conversation_with_tools(
        client, request, tool_handler, args.max_iterations, args.verbose,
    )
    elapsed = time.time() - t0

    api_usage = tool_handler.get_api_usage() if hasattr(tool_handler, "get_api_usage") else {}

    pred_answer = extract_answer(messages)

    gt = item["gt"]
    gt_list = gt if isinstance(gt, (list, tuple)) else [gt]
    scored: dict[str, float] = {}
    for name, fn in [
        ("F1",  eval_metrics.compute_score_f1),
        ("EM",  eval_metrics.compute_score_em),
        ("CEM", eval_metrics.cover_exact_match_score_1),
    ]:
        scored[name] = max(_safe_metric(fn, pred_answer, g) for g in gt_list) if gt_list else 0.0

    result = {
        **item,
        "pred_answer": pred_answer,
        "status": status,
        "tool_usage": tool_usage,
        "api_usage": api_usage,
        "elapsed_seconds": round(elapsed, 2),
        "history": messages,
        **scored,
    }

    with open(os.path.join(out_dir, f"{item['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset evaluation  (one seed × one dataset)
# ═══════════════════════════════════════════════════════════════════════════════

async def eval_dataset(client: openai.OpenAI, tool_handler, data_name: str, args) -> dict:
    """Evaluate data_name for the current args.seed.  Returns metric dict."""
    processes, out_dir = prepare_data(data_name, args, save_async=True)

    # Separate already-completed items from pending ones
    done: list[dict] = []
    todo: list[dict] = []
    for p in processes:
        if p.get("pred_answer") is not None:
            done.append(p)
        else:
            todo.append(p)

    print(f"  {data_name}: {len(todo)} pending, {len(done)} already done")

    t0 = time.time()
    lock = threading.Lock()
    results: list[dict] = list(done)

    def _handle(item, pbar=None):
        result = process_single_item(client, tool_handler, item, args, out_dir)
        with lock:
            results.append(result)
            if pbar:
                pbar.set_postfix(F1=f"{result.get('F1', 0):.2f}")

    # Use ThreadPoolExecutor for concurrent requests to the vLLM endpoint
    semaphore = asyncio.Semaphore(args.concurrent)

    async def _handle_async(item, pbar=None):
        async with semaphore:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _handle, item, pbar)

    with tqdm(total=len(todo), desc=data_name, unit="q") as pbar:
        tasks = [_handle_async(item, pbar) for item in todo]
        for coro in asyncio.as_completed(tasks):
            await coro
            pbar.update(1)

    print(f"  Finished {data_name} in {time.time()-t0:.1f}s")

    # Write final JSONL
    out_jsonl = out_dir + ".jsonl"
    results.sort(key=lambda r: int(r.get("id", 0)))
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Compute metrics
    metric_fns = {
        "F1":  eval_metrics.compute_score_f1,
        "EM":  eval_metrics.compute_score_em,
        "CEM": eval_metrics.cover_exact_match_score_1,
    }
    result_json: dict = {k: [] for k in metric_fns}
    for r in results:
        pred = r.get("pred_answer", "")
        gt_list = r["gt"] if isinstance(r["gt"], (list, tuple)) else [r["gt"]]
        for k, fn in metric_fns.items():
            r[k] = max(_safe_metric(fn, pred, g) for g in gt_list) if gt_list else 0.0
            result_json[k].append(r[k])
    for k in metric_fns:
        result_json[k] = float(np.mean(result_json[k]))

    # Re-write JSONL with per-item scores
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    result_json["time_use_in_second"] = time.time() - t0

    # Aggregate API usage across all samples
    total_serper = sum(r.get("api_usage", {}).get("serper_queries", 0) for r in results)
    total_jina_req = sum(r.get("api_usage", {}).get("jina_requests", 0) for r in results)
    total_jina_chars = sum(r.get("api_usage", {}).get("jina_total_chars", 0) for r in results)
    total_jina_tokens = total_jina_chars // 4
    jina_cost = total_jina_tokens * 0.05 / 1_000_000
    api_summary = {
        "serper_queries_total": total_serper,
        "jina_requests_total": total_jina_req,
        "jina_total_chars": total_jina_chars,
        "jina_estimated_tokens": total_jina_tokens,
        "jina_estimated_cost_usd": round(jina_cost, 4),
        "serper_queries_per_sample": round(total_serper / len(results), 2) if results else 0,
        "jina_requests_per_sample": round(total_jina_req / len(results), 2) if results else 0,
    }
    result_json["api_usage"] = api_summary
    print(f"  {data_name} API usage: {json.dumps(api_summary, indent=2)}")

    print(f"  {data_name}: {result_json}")
    return result_json


# ═══════════════════════════════════════════════════════════════════════════════
#  Aggregation (Max@k across seeds)  — same as search_eval_async.py
# ═══════════════════════════════════════════════════════════════════════════════

def compute_average(results, metric):
    values = []
    for v in results.values():
        val = v[metric]
        if isinstance(val, list):
            values.extend(val)
        elif isinstance(val, (int, float)):
            values.append(val)
    return np.mean(values) if values else np.nan


def compute_max(results, metric, n):
    ret = []
    for v in results.values():
        val = v[metric]
        if isinstance(val, list):
            ret.append(val[:n] if len(val) >= n else val)
        else:
            ret.append([val])
    if not ret:
        return np.nan
    return np.mean([max(q) for q in ret if q])


def aggregate_multiple_runs(data_name: str, base_dir: str, args, n_sampling: int,
                            tokenizer=None) -> dict:
    eval_dir = f"agent_eval_{args.max_tokens_per_call}"
    cur_dir = os.path.join(base_dir, eval_dir, data_name)
    file_n_sampling = 1

    if args.parallel_mode == "seed":
        pattern = (
            f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}"
            f"_{args.num_test_sample}_seed*"
            f"_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
            f"_s{args.start}_e{args.end}_n{file_n_sampling}.jsonl"
        )
    else:
        pattern = (
            f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}"
            f"_{args.num_test_sample}_split*"
            f"_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
            f"_s{args.start}_e{args.end}_n{file_n_sampling}.jsonl"
        )

    files = glob(os.path.join(cur_dir, pattern))
    if not files:
        return {}

    aggregated_results = defaultdict(lambda: defaultdict(list))
    metrics = ["F1", "EM", "CEM"]

    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    if not data:
                        continue
                    qid = data.get("id", "unknown")
                    for m in metrics:
                        if m in data:
                            aggregated_results[qid][m].append(data[m])
        except Exception:
            continue

    if args.llm_as_judge:
        metrics.append("MBE")
        if args.parallel_mode == "seed":
            judge_pattern = (
                f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}"
                f"_{args.num_test_sample}_seed*"
                f"_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
                f"_s{args.start}_e{args.end}_n{file_n_sampling}"
                f"-llm_as_judge_{args.judge_prompt}_use_openai_{args.use_openai}.jsonl"
            )
        else:
            judge_pattern = (
                f"{args.split}_{args.agent_type}_{args.prompt_type}_{args.search_client_type}"
                f"_{args.num_test_sample}_split*"
                f"_t{args.temperature:.1f}_topp{args.top_p:.2f}_topk{args.top_k}"
                f"_s{args.start}_e{args.end}_n{file_n_sampling}"
                f"-llm_as_judge_{args.judge_prompt}_use_openai_{args.use_openai}.jsonl"
            )
        for jf in glob(os.path.join(cur_dir, judge_pattern)):
            try:
                with open(jf, encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data and "MBE" in data:
                            aggregated_results[data.get("id", "unknown")]["MBE"].append(data["MBE"])
            except Exception:
                continue

    if not aggregated_results:
        return {}

    final: dict = {"num": len(aggregated_results)}
    for m in metrics:
        final[m] = compute_average(aggregated_results, m)
        final[f"{m}.Max@{n_sampling}"] = compute_max(aggregated_results, m, n_sampling)
    return final


# ═══════════════════════════════════════════════════════════════════════════════
#  Results table  — same as search_eval_async.py
# ═══════════════════════════════════════════════════════════════════════════════

def format_results_table(all_results: dict) -> str:
    if not all_results:
        return "No results to display"
    if not PRETTYTABLE_AVAILABLE:
        return json.dumps(all_results, indent=2, default=str)

    table = PrettyTable()
    first = next(iter(all_results.values()))
    ordered = ["num", "avg_gen_len", "avg_doc_len", "avg_num_searchs",
               "avg_num_access", "F1", "EM", "CEM", "MBE"]
    max_fields = sorted(
        (k for k in first if ".Max@" in k),
        key=lambda x: (
            0 if x.startswith("F1.Max@") else
            1 if x.startswith("EM.Max@") else
            2 if x.startswith("CEM.Max@") else
            3 if x.startswith("MBE.Max@") else 4
        ),
    )
    field_names = ["dataset"] + [f for f in ordered + max_fields if f in first]
    table.field_names = field_names

    for name, result in all_results.items():
        row = [name]
        for f in field_names[1:]:
            v = result.get(f, "-")
            row.append(f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else str(v))
        table.add_row(row)

    if len(all_results) > 1:
        row = ["Average"]
        for f in field_names[1:]:
            vals = [v.get(f) for v in all_results.values() if isinstance(v.get(f), (int, float))]
            row.append(f"{np.nanmean(vals):.3f}" if vals else "-")
        table.add_row(row)

    return str(table)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI  — mirrors search_eval_async.py, minus sglang args, plus vLLM args
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── identical to search_eval_async.py ──────────────────────────────
    parser.add_argument("--data_names", default="hotpotqa_500", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="/path/to/model", type=str,
                        help="Model served by vLLM; also passed to llm_as_judge.py")
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-calling", type=str,
                        help="Used only for output file naming (no prompt template applied)")
    parser.add_argument("--agent-type", dest="agent_type", default="oss-tool-calling", type=str,
                        help="Used only for output file naming")
    parser.add_argument("--search-client-type", dest="search_client_type",
                        default="async-search-access",
                        choices=["async-search-access", "async-web-search-access"],
                        type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-splits", dest="n_splits", default=1, type=int)
    parser.add_argument("--split-id", dest="split_id", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max-tokens-per-call", dest="max_tokens_per_call", default=4096, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parallel-mode", dest="parallel_mode", type=str,
                        default="seed", choices=["seed", "split"])
    parser.add_argument("--use-jina", dest="use_jina", action="store_true")
    parser.add_argument("--jina-api-key", dest="jina_api_key", type=str, default=None)
    parser.add_argument("--concurrent", type=int, default=64,
                        help="Max concurrent requests to vLLM endpoint")
    parser.add_argument("--llm_as_judge", action="store_true")
    parser.add_argument("--judge-prompt", dest="judge_prompt", type=str, default="default")
    parser.add_argument("--use-openai", dest="use_openai", default=False,
                        type=eval, choices=[True, False])
    parser.add_argument("--pass-at-k", dest="pass_at_k", type=int, default=1)
    parser.add_argument("--aggregate-only", dest="aggregate_only",
                        action="store_true")

    # ── vLLM / tool-calling specific (new vs search_eval_async.py) ─────
    parser.add_argument("--model-url", dest="model_url",
                        default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible endpoint URL")
    parser.add_argument("--max-iterations", dest="max_iterations", type=int, default=50,
                        help="Max tool-calling rounds per question")
    parser.add_argument("--reasoning-effort", dest="reasoning_effort", default="none",
                        choices=["low", "medium", "high", "none"],
                        help="Extended thinking effort ('none' to disable)")
    parser.add_argument("--top-k-docs", dest="top_k_docs", type=int, default=5,
                        help="Number of documents / search results to retrieve")
    parser.add_argument("--query-template", dest="query_template",
                        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
                        choices=list(QUERY_TEMPLATES),
                        help="Prompt template (from BrowseComp-Plus/search_agent/prompts.py)")
    parser.add_argument("--get-document", dest="get_document", action="store_true",
                        help="Also register the get_document tool (use with QUERY_TEMPLATE)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 1.0
        args.top_k = -1

    # prompt_type must exist in PROMPT_TYPES for prepare_data to work.
    # We inject a passthrough entry at runtime.
    from utils import PROMPT_TYPES
    if args.prompt_type not in PROMPT_TYPES:
        PROMPT_TYPES[args.prompt_type] = "{question}"

    return args


# ═══════════════════════════════════════════════════════════════════════════════
#  Main  — mirrors search_eval_async.py's pass@k + aggregation + judge flow
# ═══════════════════════════════════════════════════════════════════════════════

async def main(args):
    load_config_and_set_env()
    data_list = args.data_names.split(",")

    # ── pass@k seed generation (same as search_eval_async.py) ──────────
    base_seed = args.seed
    random.seed(base_seed)
    sampling_seeds = [base_seed]
    if args.pass_at_k > 1:
        while len(sampling_seeds) < args.pass_at_k:
            s = random.randint(0, int(1e7))
            if s not in sampling_seeds:
                sampling_seeds.append(s)

    # ── aggregate-only shortcut ─────────────────────────────────────────
    if args.aggregate_only:
        print("Aggregating existing results...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        except Exception:
            tokenizer = None
        all_results = {}
        for data_name in data_list:
            result = aggregate_multiple_runs(data_name, args.output_dir, args, args.pass_at_k, tokenizer)
            if result:
                all_results[data_name] = result
        if all_results:
            print("\n" + "=" * 60 + "\nAggregated Results:\n" + "=" * 60)
            print(format_results_table(all_results))
        return

    print(f"Pass@{args.pass_at_k} evaluation  (seeds: {sampling_seeds})")

    # ── build shared client and tool handler ────────────────────────────
    client = openai.OpenAI(base_url=args.model_url, api_key="EMPTY")
    tool_handler = make_tool_handler(args.search_client_type, args)

    original_n_sampling = args.n_sampling
    args.n_sampling = 1  # always 1 per run for pass@k

    for i, seed in enumerate(sampling_seeds):
        print(f"\n{'='*60}\n  Run {i+1}/{args.pass_at_k}  (seed={seed})\n{'='*60}")
        args.seed = seed
        tasks = [eval_dataset(client, tool_handler, data_name, args) for data_name in data_list]
        await asyncio.gather(*tasks)

    args.n_sampling = original_n_sampling
    args.seed = base_seed

    # ── aggregate Max@k ─────────────────────────────────────────────────
    print(f"\nAggregating {args.pass_at_k} sampling results...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except Exception:
        tokenizer = None

    all_results = {}
    for data_name in data_list:
        result = aggregate_multiple_runs(data_name, args.output_dir, args, args.pass_at_k, tokenizer)
        if result:
            all_results[data_name] = result

    if not all_results:
        print("\nNo results to display")
        return

    print("\n" + "=" * 60)
    print(f"Pass@{args.pass_at_k} Final Results:")
    print("=" * 60)
    print(format_results_table(all_results))

    eval_dir = f"agent_eval_{args.max_tokens_per_call}"
    result_path = os.path.join(
        args.output_dir, eval_dir,
        f"aggregate_results_{args.agent_type}_{args.prompt_type}_{args.search_client_type}"
        f"_t{args.temperature:.1f}.json",
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {result_path}")

    # ── LLM-as-judge (same subprocess pattern as search_eval_async.py) ──
    if not args.llm_as_judge:
        return

    print("\n" + "=" * 50 + "\nStarting LLM-as-judge...\n" + "=" * 50)
    seeds_str = ",".join(str(s) for s in sampling_seeds)

    judge_env = os.environ.copy()
    judge_env.setdefault("CUDA_VISIBLE_DEVICES", "0")  # llm_as_judge.py reads this in parse_args

    cmd = [
        "python", "llm_as_judge.py",
        "--data_names",          ",".join(data_list),
        "--data_dir",            args.data_dir,
        "--model_name_or_path",  args.model_name_or_path,
        "--output_dir",          args.output_dir,
        "--prompt_type",         args.prompt_type,
        "--agent-type",          args.agent_type,
        "--search-client-type",  args.search_client_type,
        "--split",               args.split,
        "--num_test_sample",     str(args.num_test_sample),
        "--seeds",               seeds_str,
        "--n-splits",            str(args.n_splits),
        "--split-id",            str(args.split_id),
        "--start",               str(args.start),
        "--end",                 str(args.end),
        "--temperature",         str(args.temperature),
        "--n_sampling",          "1",
        "--top_p",               str(args.top_p),
        "--top_k",               str(args.top_k),
        "--max-tokens-per-call", str(args.max_tokens_per_call),
        "--parallel-mode",       args.parallel_mode,
        "--tensor_parallel_size", "1",
        "--judge-prompt",        args.judge_prompt,
        "--use-openai",          str(args.use_openai),
    ]
    if args.shuffle:
        cmd.append("--shuffle")
    if args.overwrite:
        cmd.append("--overwrite")

    print(f"Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), env=judge_env)
    if ret.returncode != 0:
        print(f"LLM-as-judge failed (exit code {ret.returncode})")
        return

    # Re-aggregate with MBE scores
    print("\nRe-aggregating to include MBE scores...")
    all_results = {}
    for data_name in data_list:
        result = aggregate_multiple_runs(data_name, args.output_dir, args, args.pass_at_k, tokenizer)
        if result:
            all_results[data_name] = result
    if all_results:
        print(f"\nFinal Results with MBE  (Pass@{args.pass_at_k}):")
        print(format_results_table(all_results))
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    asyncio.run(main(args))
