import json
import random
from pathlib import Path
from compressor import Fact

LEDGER_PATH = Path(__file__).parent.parent / "memory" / "ledger.json"


class ImportanceLedger:
    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)
        self.facts: list[dict] = self._load()

    def add_facts(self, facts: list[Fact]):
        for fact in facts:
            existing = self._find(fact.text)
            if existing:
                # Reinforce: importance grows with repetition
                existing["importance"] = min(1.0, existing["importance"] + 0.1)
                existing["reinforcements"] += 1
            else:
                self.facts.append({
                    "text": fact.text,
                    "tier": fact.tier,
                    "importance": fact.importance,
                    "reinforcements": 1,
                })

    def sample_replay(self, n: int) -> list[dict]:
        if not self.facts:
            return []
        weights = [f["importance"] * (1 + f["reinforcements"] * 0.1) for f in self.facts]
        k = min(n, len(self.facts))
        return random.choices(self.facts, weights=weights, k=k)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.facts, f, indent=2)

    def _load(self) -> list[dict]:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return []

    def _find(self, text: str) -> dict | None:
        for f in self.facts:
            if f["text"].strip().lower() == text.strip().lower():
                return f
        return None
