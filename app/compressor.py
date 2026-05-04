import json
from dataclasses import dataclass
from openai import OpenAI

# Tier 0 = low-level detail, tier 1 = summary, tier 2 = abstract fact
TIERS = [0, 1, 2]


@dataclass
class Fact:
    text: str
    tier: int
    importance: float  # 0.0 - 1.0


COMPRESS_PROMPT = """\
You are a memory compression system. Given a conversation, extract facts at three tiers:
- Tier 2 (abstract): stable personal facts, preferences, identity (e.g. "User's name is Turner", "User lives in Denver")
- Tier 1 (summary): session-level context, goals, topics discussed
- Tier 0 (detail): specific exchanges, temporary context

Return JSON: {"facts": [{"text": "...", "tier": 0|1|2, "importance": 0.0-1.0}]}

Higher importance = more likely to be remembered long-term. Tier 2 facts should generally have high importance.

Conversation:
{conversation}"""


class HierarchicalCompressor:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def compress(self, messages: list[dict]) -> list[Fact]:
        conversation = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        prompt = COMPRESS_PROMPT.format(conversation=conversation)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return [Fact(**f) for f in data["facts"]]
