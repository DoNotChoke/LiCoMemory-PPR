TRIPLE_FILTERING_PROMPT = """You are a critical component of a high-stakes question-answering system. Your task is to filter facts based on their relevance to a given question. 
The question may require direct lookup or multi-hop reasoning.
Your task is to choose a subset of facts that have a strong connection to the query.
It can be a task that support reasoning or answering the question.
Important rules: 
1. Use ONLY the provided facts. 
2. Do NOT generate new facts. 
3. Return AT MOST 10 indexes. 
4. Indexes MUST use the same 0-based numbering as the input facts.
5. If no facts are relevant, return an empty list. 
6. Output MUST be valid JSON only. 
7. Do NOT include any explanation, markdown, code fences, or extra text. 
Return exactly in this format: {"triples": [0, 2, 5]}. Just output the JSON answer, no need to explain.
Think carefully and decide which facts will be helpful for the query.
Question: {question} Facts: {facts}
"""
