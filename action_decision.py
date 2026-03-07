from typing import Literal
from pydantic import BaseModel, Field

class HeuristicAgentDecision(BaseModel):
    """Decision returned by the heuristic agent."""

    action: Literal["fold", "check_or_call", "raise_to"]
    raise_to: int | None = Field(default=None)
    rationale: str = Field(default="simple_heuristic")

class PokerAgentDecision(BaseModel):
    """Decision returned by the poker agent."""

    action: Literal["fold", "check_or_call", "raise_to"]
    raise_to: int | None = Field(default=None)
    reasoning_chain: str

