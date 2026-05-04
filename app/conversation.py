from openai import OpenAI
from compressor import HierarchicalCompressor
from memory import ImportanceLedger
from trainer import LoRATrainer

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL = "personal"
CONTEXT_COMPRESS_THRESHOLD = 6000  # approx tokens before compression fires


class Conversation:
    def __init__(self):
        self.client = OpenAI(base_url=VLLM_BASE_URL, api_key="placeholder")
        self.compressor = HierarchicalCompressor(self.client, MODEL)
        self.ledger = ImportanceLedger()
        self.trainer = LoRATrainer()
        self.messages: list[dict] = []

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        if self._estimate_tokens() > CONTEXT_COMPRESS_THRESHOLD:
            self._compress_context()

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def end_session(self):
        if not self.messages:
            return
        print("Compressing context and saving to memory ledger...")
        result = self.compressor.compress(self.messages, existing_facts=self.ledger.facts)
        self.ledger.apply_corrections(result.corrections)
        self.ledger.add_facts(result.facts)
        self.ledger.save()
        replay = self.ledger.sample_replay(n=32)
        self.trainer.generate_and_save(result.facts, replay)
        print(f"Saved {len(result.facts)} facts. Stop vLLM then run `train.py` to consolidate into adapter.")

    def _compress_context(self):
        result = self.compressor.compress(self.messages, existing_facts=self.ledger.facts)
        self.ledger.apply_corrections(result.corrections)
        self.ledger.add_facts(result.facts)
        summary = "\n".join(f.text for f in result.facts if f.tier >= 1)
        self.messages = [{"role": "system", "content": f"[Context summary]\n{summary}"}]

    def _estimate_tokens(self) -> int:
        return sum(len(m["content"].split()) * 4 // 3 for m in self.messages)


if __name__ == "__main__":
    conv = Conversation()
    print("humanLLM — type 'quit' to end\n")
    try:
        while True:
            user = input("You: ").strip()
            if user.lower() in ("quit", "exit"):
                break
            print(f"Assistant: {conv.chat(user)}\n")
    finally:
        print("Consolidating memory...")
        conv.end_session()
        print("Done.")
