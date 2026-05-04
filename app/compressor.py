import json
from dataclasses import dataclass, field
from openai import OpenAI

# Tier 0 = low-level detail, tier 1 = summary, tier 2 = abstract fact


@dataclass
class Fact:
    text: str
    tier: int
    importance: float  # 0.0 - 1.0


@dataclass
class CompressResult:
    facts: list[Fact]
    corrections: list[dict]  # [{"old": "...", "new": "..."}]


COMPRESS_PROMPT = """\
You are a memory compression system. Extract only facts ABOUT THE USER from this conversation — not general world knowledge, not things the assistant said about topics.

Tiers:
- Tier 2 (abstract): stable personal facts about the user — identity, location, preferences, background. Write in third person (e.g. "User's name is Turner", "User lives in Denver", "User prefers Python")
- Tier 1 (summary): what the user was doing or discussing this session, third person (e.g. "User was discussing their move from Memphis to Denver")
- Tier 0 (detail): specific things the user said that may be relevant later but aren't stable facts, third person

Do NOT include: general facts about cities, topics, or things the assistant said that aren't user-specific.

{existing_section}Return JSON:
{{
  "facts": [{{"text": "...", "tier": 0|1|2, "importance": 0.0-1.0}}],
  "corrections": [{{"old": "exact text of known fact being corrected", "new": "corrected fact text"}}]
}}

"corrections" should only appear when the conversation explicitly contradicts or updates a known fact.
Higher importance = more stable and personal. Tier 2 facts should be 0.8+.

Conversation:
{conversation}"""

EXISTING_FACTS_SECTION = """\
IMPORTANT: The following facts are already known.
- If this conversation CONFIRMS or REPEATS one, use the EXACT same text so it can be matched.
- If this conversation CORRECTS one (e.g. user says "actually my dog's name is Max not Moby"), \
add it to "corrections" with the old text and the corrected text.
- Only add truly NEW facts to "facts".

Known facts:
{facts}

"""


class HierarchicalCompressor:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def compress(self, messages: list[dict], existing_facts: list[dict] | None = None) -> CompressResult:
        conversation = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        if existing_facts:
            facts_str = "\n".join(f"- {f['text']}" for f in existing_facts)
            existing_section = EXISTING_FACTS_SECTION.format(facts=facts_str)
        else:
            existing_section = ""

        prompt = COMPRESS_PROMPT.format(conversation=conversation, existing_section=existing_section)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        facts = [Fact(**f) for f in data.get("facts", [])]
        corrections = data.get("corrections", [])
        return CompressResult(facts=facts, corrections=corrections)
