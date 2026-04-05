from typing import List, Dict
from pydantic import BaseModel, Field

from .entity import Entity


class Relationship(BaseModel):
    source_entity: str = Field(description="Name of the source entity")
    target_entity: str = Field(description="Name of the target entity")
    relationship_name: str = Field(description="How the source entity and the target entity are related to each other")
    relationship_strength: int = Field(description="Strength of the relationship from 1 and 10.")


class EntityRelationshipExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)