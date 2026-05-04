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
Given these facts about a user, generate {n} diverse conversational examples where these facts appear naturally.
Each example should look like a real exchange where the assistant demonstrates knowledge of the user.

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


class LoRATrainer:
    # NOTE: vLLM must be stopped before calling update_adapter — both need GPU VRAM.
    def __init__(self, adapter_path: str | None = None):
        self.adapter_path = Path(adapter_path) if adapter_path else ADAPTER_PATH
        self.client = OpenAI(base_url=VLLM_BASE_URL, api_key="placeholder")

    def update_adapter(self, new_facts: list[Fact], replay: list[dict]):
        all_facts = list({f["text"] for f in replay} | {f.text for f in new_facts})
        if not all_facts:
            return

        print(f"[trainer] generating synthetic data from {len(all_facts)} facts...")
        training_data = self._generate_synthetic_data(all_facts)
        if not training_data:
            print("[trainer] no training data generated, skipping")
            return

        weights = self._compute_weights(new_facts, replay, len(training_data))
        print(f"[trainer] training on {len(training_data)} examples...")
        self._train(training_data, weights)

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
