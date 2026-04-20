from pydantic import BaseModel, Field
from typing import List


class RelevantTriples(BaseModel):
    triples: List[int] = Field(description="list of relevant fact's indexes", default_factory=list)