import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from memory import ImportanceLedger
from trainer import LoRATrainer


def main():
    ledger = ImportanceLedger()
    if not ledger.facts:
        print("No facts in ledger yet — run a conversation first.")
        return

    print(f"Loaded {len(ledger.facts)} facts from ledger.")

    # Sample high-importance facts for new_facts arg (tier 2 = abstract personal facts)
    top_facts_raw = sorted(ledger.facts, key=lambda f: f["importance"], reverse=True)[:20]

    # Convert to Fact objects for the trainer interface
    from compressor import Fact
    new_facts = [Fact(text=f["text"], tier=f["tier"], importance=f["importance"]) for f in top_facts_raw]
    replay = ledger.sample_replay(n=32)

    trainer = LoRATrainer()
    trainer.update_adapter(new_facts, replay)


if __name__ == "__main__":
    main()
