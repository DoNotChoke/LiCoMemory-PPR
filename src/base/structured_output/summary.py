from typing import List
from pydantic import BaseModel, Field

class ThemeSummary(BaseModel):
    title: str = Field(description="Short title of a distinct conversation theme")
    summary: str = Field(description="Concise factual summary of the theme")

class SessionSummaryResult(BaseModel):
    keys: List[str] = Field(default_factory=list, description="Up to 5 key information strings")
    themes: List[ThemeSummary] = Field(default_factory=list, description="Distinct themes discussed in the session")