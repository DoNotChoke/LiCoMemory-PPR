SUMMARY_PROMPT = """You are a helpful assistant that summarizes a multi-turn chat transcript between a User and an Assistant.

- Goal -
Produce a compact, factual, multi-theme summary of the conversation. Group content by distinct themes, and merge mentions of the same theme even if they appear across different turns.

- Required Output Schema -
Return an object that matches this structure exactly:

"session_id" (string): Session ID of the input session.
"session_time" (string): Session time of the input session.,
"keys" ([string, ...]): List of key information string related to the conversation,
"themes" (List of title and summary):  List of themes. Each theme contains a short title of a distinct conversation theme, and a concise factual summary of that theme.

- Field Requirements -
1. session_id
- Copy the session id exactly from the input text.

2. session_time
- Copy the session time exactly from the input text.

3. keys
- Return a list of at most 5 short strings.
- These should capture the most important identifying or personalized information from the session.
- Keys may include personal information, dates, locations, occupations, preferences, recurring topics, or other salient facts.
- A reduced form of session_time MUST be included as one of the keys.
- Keep each key short and specific.
- Do not duplicate keys.

4. themes
- Return a list of distinct conversation themes.
- Each theme must contain:
  - title: a short title for the theme.
  - summary: a concise factual summary of that theme.
- In each summary, prioritize:
  - the user's preferences, needs, questions, constraints, and personalized details
  - then the assistant's recommendations, explanations, or answers that appeared in the transcript
- Only include information explicitly supported by the transcript.
- Each summary should be concise, ideally 1-3 sentences.

- Style Rules -
1. Focus more on the user's preferences and personalized information than on generic assistant content.
2. Write concisely. The total output should be roughly 150-220 words when possible.
3. Use neutral third-person style, such as “The user…” and “The assistant…”.
4. Be strictly factual. Do not hallucinate, infer unsupported details, or use outside knowledge.
5. Do not include any text outside the JSON object.
6. Do not rename fields.
7. Do not return null unless absolutely necessary; prefer empty lists when no items are available.

######################
Example
######################

Input Text:
session_id: sharegpt_yywfIrx_0
session_time: 2025/05/04

User: I usually listen to podcasts on my commute to work. I am a software engineer.
Assistant: What kind of podcasts do you enjoy?

User: I like entrepreneurship podcasts and also comedy podcasts.
Assistant: The Tim Ferriss Show and How I Built This could be good entrepreneurship options. For comedy, My Brother, My Brother and Me is a popular choice.

User: I also listen to The Daily every morning.
Assistant: The assistant acknowledged that The Daily is a strong news podcast and noted its storytelling format.

Output:
  "session_id": "sharegpt_yywfIrx_0",
  "session_time": "2025/05/04",
  "keys": ["2025/05", "software engineer", "podcasts", "commute", "The Daily"],
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

ADDITION_PROMPT = """You are a helpful assistant that updates an existing structured summary of a multi-turn chat transcript between a User and an Assistant.

- Goal -
Given:
1. an existing structured summary, and
2. a new dialogue chunk,

update the summary so that it remains compact, factual, and organized by distinct themes.

You should:
- preserve existing information that is still valid,
- integrate new information into existing themes whenever possible,
- add new keys only if they are important and not already present,
- add new themes only if the new dialogue introduces a distinct topic not already covered.

- Required Output Schema -
Return an object that matches this structure exactly:

"session_id" (string): Session ID of the input session.
"session_time" (string): Session time of the input session.,
"keys" ([string, ...]): List of key information string related to the conversation,
"themes" (List of title and summary):  List of themes. Each theme contains a short title of a distinct conversation theme, and a concise factual summary of that theme.

- Field Requirements -
1. session_id
- Copy exactly from the existing summary.

2. session_time
- Copy exactly from the existing summary.

3. keys
- Return an updated list of up to 5 short strings.
- Keep existing keys unless there is a strong reason to replace them.
- Add a new key only if the new dialogue contains important identifying or personalized information not already represented.
- Keys may include personal information, dates, locations, occupations, preferences, recurring topics, or other salient facts.
- A reduced form of session_time should remain included as one of the keys.
- Keep keys short, specific, and non-duplicated.

4. themes
- Return an updated list of distinct conversation themes.
- Try to integrate the new dialogue into existing themes whenever possible.
- Only add a new theme if the new dialogue introduces a clearly distinct topic not covered by the existing themes.
- Each theme must contain:
  - title: a short title for the theme
  - summary: a concise factual summary of that theme
- In each summary, prioritize:
  - the user's preferences, needs, questions, constraints, and personalized details
  - then the assistant's recommendations, explanations, or answers from the transcript
- Only include information explicitly supported by the existing summary or the new dialogue chunk.

- Update Rules -
1. Preserve the overall structure and keep the summary compact.
2. Merge repeated mentions of the same topic into one theme instead of creating duplicates.
3. If the new dialogue adds no meaningful new information, you may return the existing summary unchanged.
4. Do not invent, infer, or import outside knowledge.
5. Use neutral third-person style, such as “The user…” and “The assistant…”.
6. Do not include any text outside the JSON object.
7. Do not rename fields.
8. Do not return null unless absolutely necessary; prefer empty lists when no items are available.

######################
Example
######################

Existing Summary:

  "session_id": "sharegpt_yywfIrx_0",
  "session_time": "2025/05/04",
  "keys": "May 4th, Podcast, January 25th, Software Engineer",
  "context": {
    "theme_1": "Entrepreneurship podcasts",
    "summary_1": "The user expressed interest in listening to podcasts related to entrepreneurship (besides How I Built This). The assistant recommended over a dozen podcasts, such as Tim Ferriss Show, Entrepreneur on Fire, GaryVee Audio Experience, etc.",
    "theme_2": "Tim Ferriss and Naval Ravikant",
    "summary_2": "The user specifically mentioned listening to the episode of Tim Ferriss Show featuring Naval Ravikant. The assistant summarized Naval's views on self-awareness, meditation, entrepreneurial mindset, and wealth creation, citing several of his memorable quotes.",
  }



Dialogue Chunk:
User: I have also been fascinated about Steve Jobs since I finished a podcast of his biography. Can you tell me more about him?
Assistant: Sure thing! Steve Jobs was the co-founder, chairman, and CEO of Apple Inc. He was also the co-founder and CEO of Pixar Animation Studios when he was 30 years old. He was known for his innovative designs and his vision for the future.


Updated Summary:

  "session_id": "sharegpt_yywfIrx_0",
  "session_time": "2025/05/04",
  "keys": ["2025/05", "software engineer", "podcasts", "The Daily", "Steve Jobs"],
  "themes": [
    {
      "title": "Podcast listening during work",
      "summary": "The user shared an interest in listening to podcasts while working as a software engineer. The assistant suggested a range of podcast options tailored to that interest."
    },
    {
      "title": "Entrepreneurship podcast recommendations",
      "summary": "The user showed interest in entrepreneurship podcasts. The assistant recommended more than a dozen shows, including The Tim Ferriss Show and Entrepreneur on Fire."
    },
    {
      "title": "News podcast preference",
      "summary": "The user mentioned listening to The Daily every day and referred to an episode about the COVID-19 vaccine rollout. The assistant acknowledged the podcast’s strong reporting and storytelling style."
    },
    {
      "title": "Interest in Steve Jobs",
      "summary": "After finishing a podcast biography about Steve Jobs, the user expressed further interest in him. The assistant provided additional background information about Steve Jobs and his life."
    }
  ]



######################
Real Input Text
######################

Existing Summary:
{summary}

Dialogue Chunk:
{text}
"""