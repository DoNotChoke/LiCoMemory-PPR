from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Entity(BaseModel):
    entity_name: str = Field(description="Exact entity name from the query")
    entity_type: Literal["person", "time", "organization", "event", "concept", "object"] = Field(description="Type of the entity."
                                                                                                             "Only from of the following type: person, time, organization, location, event, concept or object")

class EntityExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)