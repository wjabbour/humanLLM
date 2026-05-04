import json
from pathlib import Path
from compressor import Fact

ADAPTER_PATH = Path(__file__).parent.parent / "adapter"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYNTH_PROMPT = """\
Given these facts about a user, generate {n} diverse conversational examples where these facts naturally appear.
Each example should be a realistic assistant response or exchange that embeds the facts organically.

Facts:
{facts}

Return JSON: {{"examples": [{{"prompt": "...", "response": "..."}}]}}"""


class LoRATrainer:
    def __init__(self, adapter_path: str | None = None):
        self.adapter_path = Path(adapter_path) if adapter_path else ADAPTER_PATH
        self._model = None
        self._tokenizer = None

    def update_adapter(self, new_facts: list[Fact], replay: list[dict]):
        all_facts = list({f["text"] for f in replay} | {f.text for f in new_facts})
        if not all_facts:
            return

        training_data = self._generate_synthetic_data(all_facts)
        importance_weights = self._compute_weights(new_facts, replay)
        self._train(training_data, importance_weights)
        self.adapter_path.mkdir(exist_ok=True)
        self._save()

    def _generate_synthetic_data(self, facts: list[str]) -> list[dict]:
        # Lazy import — only load transformers when training
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        facts_str = "\n".join(f"- {f}" for f in facts)
        prompt = SYNTH_PROMPT.format(n=len(facts) * 4, facts=facts_str)

        if self._model is None:
            self._load_model()

        pipe = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer)
        result = pipe(prompt, max_new_tokens=2048, return_full_text=False)[0]["generated_text"]

        try:
            data = json.loads(result)
            return data["examples"]
        except Exception:
            return []

    def _compute_weights(self, new_facts: list[Fact], replay: list[dict]) -> list[float]:
        weights = [f.importance for f in new_facts]
        weights += [f["importance"] * (1 + f["reinforcements"] * 0.1) for f in replay]
        return weights

    def _train(self, training_data: list[dict], weights: list[float]):
        from peft import get_peft_model, LoraConfig, TaskType
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        import torch

        if self._model is None:
            self._load_model()

        if not hasattr(self._model, "peft_config"):
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.05,
            )
            self._model = get_peft_model(self._model, config)

        # TODO: convert training_data + weights into a Dataset and run Trainer
        # Placeholder until Dataset wiring is implemented
        print(f"[trainer] {len(training_data)} synthetic examples ready for training")

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        adapter = self.adapter_path if self.adapter_path.exists() else BASE_MODEL
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self._model = AutoModelForCausalLM.from_pretrained(
            adapter,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def _save(self):
        if hasattr(self._model, "save_pretrained"):
            self._model.save_pretrained(self.adapter_path)
            self._tokenizer.save_pretrained(self.adapter_path)
