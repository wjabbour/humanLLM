import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trainer import LoRATrainer


if __name__ == "__main__":
    LoRATrainer().train_from_saved()
