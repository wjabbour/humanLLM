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
You are a memory compression system. Extract only facts ABOUT THE USER from this conversation — not general world knowledge, not things the assistant said about topics.

Tiers:
- Tier 2 (abstract): stable personal facts about the user — identity, location, preferences, background. Write in third person (e.g. "User's name is Turner", "User lives in Denver", "User prefers Python")
- Tier 1 (summary): what the user was doing or discussing this session, third person (e.g. "User was discussing their move from Memphis to Denver")
- Tier 0 (detail): specific things the user said that may be relevant later but aren't stable facts, third person

Do NOT include: general facts about cities, topics, or things the assistant said that aren't user-specific.

Return JSON: {{"facts": [{{"text": "...", "tier": 0|1|2, "importance": 0.0-1.0}}]}}

Higher importance = more stable and personal. Tier 2 facts should be 0.8+.

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
