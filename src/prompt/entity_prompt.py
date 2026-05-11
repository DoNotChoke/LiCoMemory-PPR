QUERY_ENTITY_EXTRACTION_PROMPT = """You are a helpful assistant trying to extract entities from a given user query.

-Goal-
Given a user query and the query date, process identified temporal information and extract all entities of pre-defined entity types from the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized. This is a user query to the assistant so words like "I", "me", "my" refer to the user and "you", "your" refer to the assistant.
- entity_type: ONLY from one of the following types: [person, time, organization, location, event, concept, object]

Note: For entities that are related to the query time, process the temporal information and extract the result as a time entity.

2. Return output in English as a single list of all the entities identified in steps 1.


######################
-Examples-
######################
Example 1:

Query:
What coffee did you recommend me on January 15th?
Query time:
2023/02/01 (Wed) 10:20
################
Output:
Entity Name: User - Entity Type: person
Entity Name: Assistant - Entity Type: person
Entity Name: Coffee - Entity Type: object
Entity Name: January 15th - Entity Type: time

######################
Example 2:

Query:
What podcast did my sister recommend last Saturday?
Query time:
2024/07/12 (Mon) 8:20
################
Output:
Entity Name: User - Entity Type: person
Entity Name: Podcast - Entity Type: object
Entity Name: Sister - Entity Type: person
Entity Name: July 10th - Entity Type: time

######################
Example 3:

Query:
Who gave me a new stand mixer as a birthday gift last year?
Query time:
2022/05/12 (Fri) 12:20
################
Output:
Entity Name: User - Entity Type: person
Entity Name: Stand Mixer - Entity Type: object
Entity Name: Birthday Gift - Entity Type: object
Entity Name: 2021 - Entity Type: time

######################
-Real Text-
######################
Query:
{text}
Query time:
{session_time}
"""

DIALOGUE_EXTRACTION_PROMPT = """You are a helpful assistant trying to extract entities and relations from a given multi-turn chat transcript between a User and an Assistant.

-Goal-
Given a multi-turn chat transcript between a User and an Assistant in json format, identify all entities of pre-defined entity types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity in original format, capitalized. User and Assistant MUST be among the entities.
- entity_type: ONLY from one of the following types: [person, time, organization, location, event, concept, object]. Note that temporal information should be processed and extracted as a time entity, you must return a specific time if possible (e.g 2025/05/04, ...)
Format each entity as ("entity"|<entity_name>|<entity_type>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.

Prioritize the following relationship types (in order):
A. USER-SIDE relationships (highest priority):
   - User asked about / requested / mentioned X
   - User purchased / owns / uses X
   - User prefers / interested in / dislikes X
   - User plans to / needs to / wants to X

B. ASSISTANT-SIDE relationships (must not be skipped):
   - Assistant recommended / suggested / advised X
   - Assistant warned about / cautioned against X
   - X is used for / applied to / works with Y (when the assistant explains usage)
   - X is a type of / similar to / better than Y (when the assistant explains concepts)

C. ENTITY-ENTITY relationships:
   - X is used for Y (tool-purpose)
   - X is a component of Y (part-whole)
   - X is located in / belongs to Y

For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_name: how the source entity and the target entity are related to each other
- relationship_strength: a numeric score from 1 to 10.
  User-related relationships: 7-10.
  Assistant recommendations: 5-7.
  Entity-entity relationships: 4-6.
Format each relationship as ("relationship"|<source_entity>|<target_entity>|<relationship_name>|<relationship_strength>)

3. ORPHAN ENTITY RULE (mandatory check before output):
Every entity from step 1 MUST appear in at least one relationship in step 2.
If an entity has no relationship, you MUST either:
  (a) create a relationship for it based on the text, OR
  (b) remove it from the entity list.
Never output an entity that does not appear in any relationship — orphan entities
cannot be navigated in the knowledge graph and waste index space.

4. Return output in English as a single list of all the entities and relationships
identified in steps 1 and 2. Use **##** as the list delimiter.

5. When finished, output ##END##

######################
-Example-
######################
Text:
'text': 'User: I've been thinking about starting my own business as I graduated from business administration and I was wondering if you could recommend some inspiring podcasts about entrepreneurship, aside from \"How I Built This\" which I finished this Monday on my sister's recommendation.
Assistant: Congrats on starting your business! An inspiring podcast similar to "How I Built This" is **Side Hustle School** - Short tips to grow side gigs into full-time.
A single spark can launch bold goals. Good luck!',
'session_time': '2023/01/20 (Sat) 02:21',
################
("entity"|User|person)##
("entity"|Assistant|person)##
("entity"|How I Built This|concept)##
("entity"|Side Hustle School|concept)##
("entity"|Business Administration|concept)##
("entity"|Entrepreneurship|concept)##
("entity"|Sister|person)##
("entity"|January 15th|time)##
("entity"|Business|concept)##
("entity"|Podcast|object)##

("relationship"|User|Business Administration|has educational background in|10)##
("relationship"|User|Entrepreneurship|interested in|9)##
("relationship"|User|How I Built This|finished listening to|8)##
("relationship"|User|Sister|received recommendation from|9)##
("relationship"|User|Business|planning to start|8)##
("relationship"|User|Podcast|looking for recommendations for|7)##
("relationship"|How I Built This|January 15th|finished on|7)##
("relationship"|Sister|How I Built This|recommended|7)##
("relationship"|Assistant|Side Hustle School|recommended|6)##
("relationship"|How I Built This|Side Hustle School|similar to|6)##
("relationship"|Side Hustle School|Entrepreneurship|covers topic of|5)##
##END##

Note: Every entity (User, Assistant, How I Built This, Side Hustle School,
Business Administration, Entrepreneurship, Sister, January 15th, Business,
Podcast) appears in at least one relationship — orphan rule satisfied.

######################
-Real Text-
######################
Text:
{text}
################
Output:"""

LOCOMO_EXTRACTION_PROMPT = """You are a helpful assistant trying to extract entities and relations from a given multi-turn chat transcript.

-Goal-
Given a multi-turn chat transcript in json format, identify all entities of pre-defined entity types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity in original format, capitalized. Names of all speakers MUST be among the entities.
- entity_type: ONLY from one of the following types: [person, time, organization, location, event, concept, object].
  Note that temporal information should be processed and extracted as a time entity.
  You must return a specific time if possible based on the text and session_time
  (e.g., 2023/05/20, May 19th, yesterday, last Monday, etc.).
Format each entity as ("entity"|<entity_name>|<entity_type>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.

FOCUS MORE on relationships that clarify speakers' personal information, preferences, memories, actions, plans, and social connections.

Prioritize the following relationship types in order:

A. SPEAKER-SIDE relationships (highest priority):
   - Speaker owns / bought / borrowed / used / gave / received X
   - Speaker likes / dislikes / prefers / is interested in X
   - Speaker is from / lives in / works at / studies at X
   - Speaker plans to / wants to / needs to / decided to X
   - Speaker asked about / mentioned / talked about X
   - Speaker has a family / friendship / professional relationship with another person

B. PERSONAL MEMORY relationships:
   - Speaker did X at/on a specific time
   - Speaker experienced / attended / visited / celebrated X
   - Speaker associates an object, event, or place with a personal memory
   - Speaker has a habit, routine, preference, or recurring activity related to X

C. ENTITY-ENTITY relationships:
   - X is located in / belongs to Y
   - X is a part/component/type of Y
   - X is used for / related to / similar to Y
   - X happened at/on Y
   - X was given by / recommended by / borrowed from Y

For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_name: how the source entity and the target_entity are related to each other
- relationship_strength: a numeric score from 1 to 10.
  Speaker-related personal relationships: 7-10.
  Strong personal facts, preferences, ownership, family/social ties: 8-10.
  Time/action relationships: 6-9.
  Entity-entity relationships: 4-7.
Format each relationship as ("relationship"|<source_entity>|<target_entity>|<relationship_name>|<relationship_strength>)

3. ORPHAN ENTITY RULE (mandatory check before output):
Every entity from step 1 MUST appear in at least one relationship in step 2.
If an entity has no relationship, you MUST either:
  (a) create a relationship for it based on the text, OR
  (b) remove it from the entity list.
Never output an entity that does not appear in any relationship — orphan entities
cannot be navigated in the knowledge graph and waste index space.

4. Return output in English as a single list of all the entities and relationships
identified in steps 1 and 2. Use **##** as the list delimiter.

5. When finished, output ##END##

######################
-Example-
######################
Text:
'text': 'Caroline: Hey Melanie! Take a look at this. (attached is a photo of a woman holding a necklace with a cross and a heart)
Melanie: Hey, Caroline! Nice to hear from you! Love the necklace!
Caroline: Thanks! This necklace is super special to me - a gift from my grandma in my home country, Sweden.
Melanie: It\\'s so pretty! This reminds me, I will swing by and return you the earing I borrowed from you yesterday.',
'session_time': '2023/05/20',

################
("entity"|Caroline|person)##
("entity"|Melanie|person)##
("entity"|Grandma|person)##
("entity"|Earing|object)##
("entity"|Necklace|object)##
("entity"|Sweden|location)##
("entity"|May 19th|time)##
("entity"|Photo|object)##

("relationship"|Caroline|Necklace|owns|10)##
("relationship"|Caroline|Sweden|is from|9)##
("relationship"|Grandma|Caroline|gave the necklace to|9)##
("relationship"|Caroline|Melanie|sent a photo of a necklace to|9)##
("relationship"|Caroline|Photo|sent|7)##
("relationship"|Photo|Necklace|shows|6)##
("relationship"|Melanie|Necklace|likes|8)##
("relationship"|Melanie|Caroline|borrowed an earing from|9)##
("relationship"|Melanie|Earing|borrowed|9)##
("relationship"|Melanie|May 19th|borrowed an earing on|8)##
("relationship"|Melanie|Caroline|plans to return the earing to|8)##
##END##

Note: Every entity (Caroline, Melanie, Grandma, Earing, Necklace, Sweden,
May 19th, Photo) appears in at least one relationship — orphan rule satisfied.

######################
-Real Text-
######################
Text:
{text}
################
Output:"""