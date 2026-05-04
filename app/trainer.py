import json
import torch
from pathlib import Path
from openai import OpenAI
from compressor import Fact
from transformers import Trainer

ADAPTER_PATH = Path(__file__).parent.parent / "adapter"
BASE_MODEL = str(Path(__file__).parent.parent / "models" / "Qwen2.5-7B-Instruct")
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = str(Path(__file__).parent.parent / "models" / "Qwen2.5-7B-Instruct-AWQ")

SYNTH_PROMPT = """\
Given these facts about a user named Turner, generate {n} diverse conversational examples that teach an AI assistant to know these facts.

Each example must follow this structure:
- "prompt": a user message asking the assistant something related to these facts (e.g. "What's my wife's name?", "Where am I from?", "Do I have any pets?")
- "response": the assistant's answer in first-person assistant voice, demonstrating knowledge of the user (e.g. "Your wife's name is Katherine.", "You were born and raised in Memphis, Tennessee.")

The assistant should answer naturally and confidently, as if it already knows the user personally.

Facts:
{facts}

Return JSON: {{"examples": [{{"prompt": "...", "response": "..."}}]}}"""


class _WeightedTrainer(Trainer):
    """HuggingFace Trainer that scales loss by per-sample importance weights."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight").to(model.device)
        labels = inputs["labels"].clone()
        outputs = model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        token_counts = (shift_labels != -100).sum(dim=-1).float().clamp(min=1)
        sample_losses = token_losses.sum(dim=-1) / token_counts
        loss = (sample_losses * weights).mean()

        return (loss, outputs) if return_outputs else loss


SYNTH_DATA_PATH = Path(__file__).parent.parent / "memory" / "synth_data.json"


class LoRATrainer:
    def __init__(self, adapter_path: str | None = None):
        self.adapter_path = Path(adapter_path) if adapter_path else ADAPTER_PATH
        self.client = OpenAI(base_url=VLLM_BASE_URL, api_key="placeholder")

    def generate_and_save(self, new_facts: list[Fact], replay: list[dict]):
        """Call while vLLM is running. Generates synthetic data and saves to disk."""
        all_facts = list({f["text"] for f in replay} | {f.text for f in new_facts})
        if not all_facts:
            return

        # Build per-fact importance scores keyed by text
        fact_score: dict[str, float] = {}
        for f in new_facts:
            fact_score[f.text] = f.importance
        for f in replay:
            score = f["importance"] * (1 + f["reinforcements"] * 0.1)
            fact_score[f["text"]] = max(fact_score.get(f["text"], 0), score)

        print(f"[trainer] generating synthetic data from {len(all_facts)} facts...")
        training_data = self._generate_synthetic_data(all_facts)
        if not training_data:
            print("[trainer] no synthetic data generated")
            return

        print(f"[trainer] annotating per-example fact coverage...")
        coverage = self._annotate_coverage(training_data, all_facts)
        weights = self._coverage_weights(coverage, all_facts, fact_score)

        payload = [{"prompt": ex["prompt"], "response": ex["response"], "weight": w}
                   for ex, w in zip(training_data, weights)]
        SYNTH_DATA_PATH.parent.mkdir(exist_ok=True)
        with open(SYNTH_DATA_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[trainer] saved {len(payload)} examples to {SYNTH_DATA_PATH}")

    def train_from_saved(self):
        """Call after stopping vLLM. Loads saved synthetic data and trains the adapter."""
        if not SYNTH_DATA_PATH.exists():
            print("[trainer] no saved synthetic data found — run a conversation first")
            return
        with open(SYNTH_DATA_PATH) as f:
            payload = json.load(f)
        training_data = [{"prompt": p["prompt"], "response": p["response"]} for p in payload]
        weights = [p["weight"] for p in payload]
        print(f"[trainer] training on {len(training_data)} examples...")
        self._train(training_data, weights)
        SYNTH_DATA_PATH.unlink()  # clear after training

    def _generate_synthetic_data(self, facts: list[str]) -> list[dict]:
        facts_str = "\n".join(f"- {f}" for f in facts)
        prompt = SYNTH_PROMPT.format(n=len(facts) * 4, facts=facts_str)
        try:
            response = self.client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=2048,
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("examples", [])
        except Exception as e:
            print(f"[trainer] synthetic data generation failed: {e}")
            return []

    def _annotate_coverage(self, examples: list[dict], facts: list[str]) -> list[list[int]]:
        """Returns list of fact-index lists, one per example."""
        indexed_facts = "\n".join(f"{i}: {f}" for i, f in enumerate(facts))
        indexed_examples = "\n".join(
            f"{i}: prompt={ex['prompt']!r} response={ex['response']!r}"
            for i, ex in enumerate(examples)
        )
        prompt = (
            "For each example below, list which fact indices (0-based) it covers.\n\n"
            f"Facts:\n{indexed_facts}\n\n"
            f"Examples:\n{indexed_examples}\n\n"
            'Return JSON: {"coverage": [[fact_indices_for_example_0], [fact_indices_for_example_1], ...]}'
        )
        try:
            response = self.client.chat.completions.create(
                model=VLLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            data = json.loads(response.choices[0].message.content)
            coverage = data.get("coverage", [])
            # Pad/trim to match example count
            while len(coverage) < len(examples):
                coverage.append([])
            return coverage[:len(examples)]
        except Exception as e:
            print(f"[trainer] coverage annotation failed: {e}, using uniform weights")
            return [list(range(len(facts)))] * len(examples)

    def _coverage_weights(self, coverage: list[list[int]], facts: list[str], fact_score: dict[str, float]) -> list[float]:
        weights = []
        for indices in coverage:
            if indices:
                weights.append(max(fact_score.get(facts[i], 0.5) for i in indices if i < len(facts)))
            else:
                weights.append(sum(fact_score.values()) / len(fact_score) if fact_score else 1.0)
        return weights

    def _compute_weights(self, new_facts: list[Fact], replay: list[dict], n: int) -> list[float]:
        fact_weights = [f.importance for f in new_facts]
        fact_weights += [f["importance"] * (1 + f["reinforcements"] * 0.1) for f in replay]
        avg = sum(fact_weights) / len(fact_weights) if fact_weights else 1.0
        return [avg] * n

    def _train(self, training_data: list[dict], weights: list[float]):
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import get_peft_model, LoraConfig, TaskType, PeftModel
        from datasets import Dataset

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        if self.adapter_path.exists():
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
            model = PeftModel.from_pretrained(base, str(self.adapter_path), is_trainable=True)
        else:
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
            model = get_peft_model(base, LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
            ))
            model.print_trainable_parameters()

        def tokenize(example, weight):
            prompt_ids = tokenizer(example["prompt"] + "\n", add_special_tokens=True)["input_ids"]
            full = tokenizer(
                example["prompt"] + "\n" + example["response"],
                add_special_tokens=True,
                truncation=True,
                max_length=1024,
            )
            labels = [-100] * len(prompt_ids) + full["input_ids"][len(prompt_ids):]
            return {
                "input_ids": full["input_ids"],
                "attention_mask": full["attention_mask"],
                "labels": labels,
                "weight": weight,
            }

        dataset = Dataset.from_list([tokenize(ex, w) for ex, w in zip(training_data, weights)])

        trainer = _WeightedTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(self.adapter_path),
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
            ),
            train_dataset=dataset,
        )
        trainer.train()

        self.adapter_path.mkdir(exist_ok=True)
        model.save_pretrained(str(self.adapter_path))
        tokenizer.save_pretrained(str(self.adapter_path))
        print(f"[trainer] adapter saved to {self.adapter_path}")
