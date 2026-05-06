RECOGNITION_MEMORY_PROMPT = """You are a critical component of a high-stakes question-answering system used by top researchers and decision-makers 
worldwide. Your task is to filter facts based on their relevance to a given query, ensuring that the most crucial information is presented to these stakeholders. 
The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information.
You must select up to 5 relevant facts from the provided candidate list that have a strong connection to the query, aiding in reasoning and providing an accurate answer.
Return the relevant facts exactly how they present.
The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. 
You must only use facts from the candidate list and not generate new facts. The future of critical decision making relies on your ability to accurately filter and present relevant information.

Example:
Question: Are Imperial River (Florida) and Amaradia(Dolj) both located in the same country? 
Fact Before Filter: 
("imperial river", "is located in", "florida")
("imperial river", "is a river in", "united states")
("imperial river", "may refer to", "south america")
("amaradia", "flows through", "roiade amaradia")
("imperial river", "may refer to", "united states")

Fact After Filter: 
("imperial river","is located in","florida")
("imperial river","is a river in","unitedstates")
("amaradia","flows through","roiade amaradia")

Now filter the facts and choose up to 5 relevant facts. Just about the relevant facts without any explanation.
Input:
Question: {question}
Facts: {facts}
"""