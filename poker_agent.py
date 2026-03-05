import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from agents import Agent, Runner
from pokerkit import State
from pydantic import BaseModel, Field


class PokerAgentDecision(BaseModel):
    """Structured action returned by the agent"""

    action: Literal["fold", "check_or_call", "raise_to"]
    raise_to: int | None = Field(default=None)
    rationale: str


@dataclass
class LlmPokerAgent:
    name: str = "poker_agent"
    model: str = "gpt-4.1-mini"
    previous_response_id: str | None = None

    def __post_init__(self) -> None:
        self._agent = Agent(
            name="Poker Agent",
            model=os.getenv(self.model),
            instructions=(
                "You are a poker decision policy for No-Limit Texas Hold'em. "
                "You receive only allowed game-state snapshots. "
                "Return exactly one legal action in the output schema. "
                "Prefer check_or_call when uncertain."
            ),
            output_type=PokerAgentDecision,
        )

    def reset_for_new_hand(self) -> None:
        """Reset per-hand memory chain."""
        # this is for if we ever want to run multiple hands
        self.previous_response_id = None

    def decide(self, llm_view: dict[str, Any]) -> PokerAgentDecision:
        """Ask the model for a decision from the allowed snapshot only."""
        result = Runner.run_sync(
            self._agent,
            json.dumps(llm_view, ensure_ascii=True),
            previous_response_id=self.previous_response_id,
        )
        """
        This is so that
        - decision 1 sees context from decision 0
        - decision 2 sees context from 0+1
        - decision 3 sees context from 0+1+2
        """
        self.previous_response_id = result.last_response_id
        return result.final_output_as(PokerAgentDecision)


def apply_poker_agent_decision(state: State, decision: PokerAgentDecision) -> str:
    """Apply poker-agent decision and raise on illegal actions."""
    if decision.action == "check_or_call" and state.can_check_or_call():
        state.check_or_call()
        return "check_or_call"

    if decision.action == "fold" and state.can_fold():
        state.fold()
        return "fold"

    if decision.action == "raise_to":
        target = decision.raise_to
        if target is not None and state.can_complete_bet_or_raise_to(target):
            state.complete_bet_or_raise_to(target)
            return f"raise_to_{target}"

    raise ValueError(f"Illegal poker-agent action: {decision.model_dump()}")
