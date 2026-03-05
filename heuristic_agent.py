"""Heuristic poker agent.

It always prefers check/call, then falls back to folding.
"""

from dataclasses import dataclass
from typing import Literal

from pokerkit import State
from pydantic import BaseModel, Field

from action_entry import ActionEntry


class HeuristicAgentDecision(BaseModel):
    """Decision returned by the heuristic agent."""

    action: Literal["fold", "check_or_call", "raise_to"]
    raise_to: int | None = Field(default=None)
    rationale: str = Field(default="simple_heuristic")


@dataclass
class HeuristicAgent:
    """Basic heuristic policy."""

    seat_index: int
    name: str = "heuristic"
    tracked_hole_cards: tuple[str, ...] = ()

    def reset_for_new_hand(self) -> None:
        """Clear per-hand memory."""
        self.tracked_hole_cards = ()

    def decide(self, state: State) -> HeuristicAgentDecision:
        """Choose an action from the heuristic policy."""
        if state.can_check_or_call():
            return HeuristicAgentDecision(action="check_or_call")

        if state.can_fold():
            return HeuristicAgentDecision(action="fold")

        raise ValueError("No legal heuristic action available.")


def apply_heuristic_agent_decision(
    state: State,
    decision: HeuristicAgentDecision,
    *,
    street: str,
) -> ActionEntry:
    """Apply heuristic decision and return an action entry."""
    player_index = state.actor_index
    if player_index is None:
        raise ValueError("No active actor when applying heuristic decision.")

    if decision.action == "check_or_call" and state.can_check_or_call():
        operation = state.check_or_call()
        amount = getattr(operation, "amount", 0)
        if amount == 0:
            return {
                "player": f"p{player_index}",
                "street": street,
                "action_taken": "check",
            }
        return {
            "player": f"p{player_index}",
            "street": street,
            "action_taken": "call",
            "amount": amount,
        }

    if decision.action == "fold" and state.can_fold():
        state.fold()
        return {
            "player": f"p{player_index}",
            "street": street,
            "action_taken": "fold",
        }

    raise ValueError(f"Illegal heuristic action: {decision.model_dump()}")
