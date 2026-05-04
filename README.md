# humanLLM

A personal memory system for LLMs that stores facts about you in model weights — not context files.

Inspired by how human memory works: short-term compression during conversation, long-term consolidation during sleep.

## How it works

**Phase 1 — Hierarchical context compression**

During conversation, when context grows too large, it's compressed into a tiered hierarchy of abstract facts. Facts are stored in a ledger with importance scores that increase with repetition — just like human memory strengthens through reinforcement.

**Phase 2 — Weight consolidation**

At the end of each conversation, the system generates synthetic training data from the ledger facts, filters out hallucinations, and annotates per-example importance weights. After stopping inference, a LoRA adapter is trained on this data. The model now *knows* your facts — no context injection, no retrieval, no tokens spent.

## Architecture

```
conversation.py   — main chat loop, triggers compression on context overflow
compressor.py     — hierarchical fact extraction + contradiction detection
memory.py         — importance ledger with reinforcement and weighted replay
trainer.py        — synthetic data generation, hallucination filtering, LoRA training
train.py          — standalone training script (run after stopping vLLM)
```

## Workflow

```bash
# 1. Start vLLM with your adapter (first run: omit --enable-lora flags)
python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen2.5-7B-Instruct-AWQ \
  --port 8000 --quantization awq --max-model-len 8192 --enforce-eager \
  --enable-lora --max-lora-rank 64 \
  --lora-modules personal=/path/to/humanLLM/adapter

# 2. Have a conversation
python app/conversation.py
# On exit: facts are compressed, ledger updated, synthetic data generated

# 3. Stop vLLM, then train
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python app/train.py
# Adapter saved to adapter/

# 4. Restart vLLM with updated adapter and repeat
```

## Requirements

- ROCm 7.x (AMD GPU) or CUDA
- vLLM built from source
- HuggingFace `peft`, `transformers`, `datasets`
- `openai` Python package

## Models

Tested with `Qwen/Qwen2.5-7B-Instruct-AWQ` for inference (fits in 16GB VRAM) and `Qwen/Qwen2.5-7B-Instruct` (fp16) for LoRA training.

## Prior art

Closest academic work: [Language Models Need Sleep](https://openreview.net/forum?id=iiZy6xyVVE) and [SCM: Sleep-Consolidated Memory with Algorithmic Forgetting](https://arxiv.org/html/2604.20943v1). This project independently arrived at the same core idea and implements it as a practical local system.
