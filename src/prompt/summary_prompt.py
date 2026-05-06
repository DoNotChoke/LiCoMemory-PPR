SUMMARY_PROMPT = """You are a memory management module for a long-term conversational agent. Your task is to produce a structured session summary for a conversation.

- Goal -
Produce a compact, factual, retrieval-optimized summary of a multi-turn conversation. The output serves as a session-level index node: it must surface key entities, temporal anchors, and thematic content that downstream retrieval can match against.
You are given session time and the conversation, summary the core information.
- Required Output Schema -
Return a structured output with exactly these fields:

"keys" ([string, ...]): List of key retrieval anchors for this session.
"themes" (list of {title, summary}): Thematic breakdown of the session.

- Field Requirements -

1. keys
- Return a list of at most 6 short strings (1–4 words each).
- Must include: session_time in reduced form (e.g. "2025/05", "2025/05/04"), named entities (people, places, products, organizations), and dominant topics.
- Keys are used as retrieval anchors — prefer specific, distinctive terms over generic ones.
- Avoid duplicates. Avoid vague terms like "conversation" or "discussion".

2. themes
- Return a list of distinct thematic units from the session.
- Each theme must contain:
  - title (string): Short, specific label (3–6 words).
  - summary (string): 1–3 sentence factual summary. Lead with user intent, constraints, or stated facts. Then include agent responses only if they add retrievable value (specific recommendations, named items, decisions made).
- Merge content that belongs to the same topic even if spread across turns.
- Do not duplicate information across themes.

- Style Rules -
1. Prioritize user-stated facts, preferences, constraints, and personal details over agent-generated content.
2. Every claim must be grounded in the transcript. No inference, no outside knowledge.
3. Entity mentions in summaries should match the entities list exactly (same spelling/form).
4. Use neutral third-person: "The user…", "The assistant…".
5. Total output should be 180–280 words.

######################
Example
######################

Input Text:
session_time: 2025/05/04

User: I usually listen to podcasts on my commute to work. I am a software engineer.
Assistant: What kind of podcasts do you enjoy?

User: I like entrepreneurship podcasts and also comedy podcasts.
Assistant: The Tim Ferriss Show and How I Built This could be good entrepreneurship options. For comedy, My Brother, My Brother and Me is a popular choice.

User: I also listen to The Daily every morning.
Assistant: The assistant acknowledged that The Daily is a strong news podcast and noted its storytelling format.


Output:
  "keys": ["2025/05/04", "software engineer", "podcasts", "commute", "The Daily"],
  "themes": [
    {
      "title": "Podcast listening habits",
      "summary": "The user shared that podcasts are part of their daily commute and mentioned being a software engineer. The assistant asked about the user’s podcast preferences to refine recommendations."
    },
    {
      "title": "Entrepreneurship and comedy podcasts",
      "summary": "The user said they enjoy entrepreneurship and comedy podcasts. The assistant recommended shows such as The Tim Ferriss Show, How I Built This, and My Brother, My Brother and Me."
    },
    {
      "title": "News podcast preference",
      "summary": "The user mentioned listening to The Daily every morning. The assistant acknowledged it as a strong news podcast with a storytelling style."
    }
  ]



######################
Real Input Text
######################

{text}
"""


ADDITION_PROMPT = """You are a memory management module for a long-term conversational agent. Your task is to incrementally update an existing structured session summary given a new dialogue chunk.

- Goal -
Merge new information into the existing summary while preserving compactness and retrieval quality. The updated summary must remain accurate, deduplicated, and well-organized for downstream retrieval and entity-relation triple extraction.

- Required Output Schema -
Return a JSON object with exactly these fields:

"session_id" (string): Copied from existing summary.
"session_time" (string): Copied from existing summary.
"keys" ([string, ...]): Updated retrieval anchor list.
"themes" (list of {title, summary, entities}): Updated thematic breakdown.

- Update Rules -

1. session_id / session_time
- Copy exactly from the existing summary. Never modify.

2. keys
- Retain all existing keys unless directly contradicted.
- Add new keys only if the new dialogue introduces important named entities or topics not yet represented.
- Cap at 6 keys. Drop the least distinctive key if needed to stay within limit.
- Prefer specific, retrievable terms (named entities, dates, locations) over generic ones.

3. themes
- First attempt to integrate new content into existing themes.
- Only create a new theme if the new dialogue introduces a clearly distinct topic not covered.
- For each updated theme: revise the summary to include new facts, and update the entities list.
- If the new dialogue adds no meaningful new information, return the existing summary unchanged.
- Never duplicate content across themes.

4. Conflict & update handling
- If the new dialogue contradicts an existing fact (e.g., user corrects a previous statement), update the relevant summary to reflect the most recent version.
- Mark time-sensitive updates by including the relevant date/time in the summary sentence (e.g., "As of [date], the user…").

- Style Rules -
1. Lead summaries with user-stated facts and preferences; include agent content only when it adds retrievable value.
2. All claims must be grounded in the existing summary or the new dialogue. No inference.
3. Entity strings in the entities list must match their usage in the summary exactly.
4. Use neutral third-person style.
5. Return only the JSON object. No preamble, no markdown fences.

######################
Example
######################

Existing Summary:
{
  "session_id": "sharegpt_yywfIrx_0",
  "session_time": "2025/05/04",
  "keys": ["2025/05", "software engineer", "podcasts", "Tim Ferriss", "Naval Ravikant"],
  "themes": [
    {
      "title": "Entrepreneurship podcast recommendations",
      "summary": "The user asked for entrepreneurship podcast suggestions. The assistant recommended The Tim Ferriss Show, Entrepreneur on Fire, and GaryVee Audio Experience.",
      "entities": ["Tim Ferriss Show", "Entrepreneur on Fire", "GaryVee Audio Experience"]
    },
    {
      "title": "Naval Ravikant episode interest",
      "summary": "The user mentioned listening to the Tim Ferriss episode featuring Naval Ravikant. The assistant summarized Naval's perspectives on wealth, meditation, and entrepreneurship.",
      "entities": ["Tim Ferriss", "Naval Ravikant"]
    }
  ]
}

Dialogue Chunk:
User: I've also been fascinated by Steve Jobs since I finished a podcast about his biography. Can you tell me more?
Assistant: Steve Jobs was the co-founder and CEO of Apple Inc. and co-founder of Pixar. He was known for his focus on design and long-term product vision.

Updated Summary:
{
  "session_id": "sharegpt_yywfIrx_0",
  "session_time": "2025/05/04",
  "keys": ["2025/05", "software engineer", "podcasts", "Tim Ferriss", "Naval Ravikant", "Steve Jobs"],
  "themes": [
    {
      "title": "Entrepreneurship podcast recommendations",
      "summary": "The user asked for entrepreneurship podcast suggestions. The assistant recommended The Tim Ferriss Show, Entrepreneur on Fire, and GaryVee Audio Experience.",
      "entities": ["Tim Ferriss Show", "Entrepreneur on Fire", "GaryVee Audio Experience"]
    },
    {
      "title": "Naval Ravikant episode interest",
      "summary": "The user mentioned listening to the Tim Ferriss episode featuring Naval Ravikant. The assistant summarized Naval's perspectives on wealth, meditation, and entrepreneurship.",
      "entities": ["Tim Ferriss", "Naval Ravikant"]
    },
    {
      "title": "Interest in Steve Jobs biography",
      "summary": "After finishing a podcast biography of Steve Jobs, the user expressed interest in learning more. The assistant provided background on Jobs as co-founder of Apple and Pixar and his focus on design.",
      "entities": ["Steve Jobs", "Apple", "Pixar"]
    }
  ]
}

######################
Real Input Text
######################

Existing Summary:
{summary}

Dialogue Chunk:
{text}
"""