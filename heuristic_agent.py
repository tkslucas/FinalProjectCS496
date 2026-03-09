"""Heuristic based poker agent with adjustable policies"""

import random
from dataclasses import dataclass, field
from typing import Any, cast

from pokerkit import (
    Deck,
    StandardHighHand,
    State,
    calculate_hand_strength,
)

from action_decision import HeuristicAgentDecision
from action_entry import ActionEntry


class RandomPolicy:
    """Pick randomly from the legal actions."""

    def decide(self, state, seat_index):
        actions = []

        if state.can_fold():
            actions.append(
                HeuristicAgentDecision(
                    action="fold",
                    rationale="random_policy",
                )
            )

        if state.can_check_or_call():
            actions.append(
                HeuristicAgentDecision(
                    action="check_or_call",
                    rationale="random_policy",
                )
            )

        if state.can_complete_bet_or_raise_to():
            min_raise_to = state.min_completion_betting_or_raising_to_amount
            max_raise_to = state.max_completion_betting_or_raising_to_amount
            if min_raise_to is not None and max_raise_to is not None:
                actions.append(
                    HeuristicAgentDecision(
                        action="raise_to",
                        raise_to=random.randint(min_raise_to, max_raise_to),
                        rationale="random_policy",
                    )
                )

        if not actions:
            raise ValueError("No legal heuristic action available.")

        return random.choice(actions)


class HandStrengthPolicy:
    """Choose actions from monte carlo hand strength estimates."""

    def __init__(
        self,
        sample_count=300,
        raise_threshold=0.7,
        call_threshold=0.3,
    ):
        self.sample_count = sample_count
        self.raise_threshold = raise_threshold
        self.call_threshold = call_threshold

    def decide(self, state, seat_index):
        strength = self._calculate_strength(state, seat_index)
        rationale = f"hand_strength={strength:.3f}"

        if state.can_complete_bet_or_raise_to() and strength >= self.raise_threshold:
            min_raise_to = state.min_completion_betting_or_raising_to_amount
            if min_raise_to is not None:
                return HeuristicAgentDecision(
                    action="raise_to",
                    raise_to=min_raise_to,
                    rationale=rationale,
                )

        if state.can_check_or_call() and strength >= self.call_threshold:
            return HeuristicAgentDecision(
                action="check_or_call",
                rationale=rationale,
            )

        if state.can_fold():
            return HeuristicAgentDecision(
                action="fold",
                rationale=rationale,
            )

        if state.can_check_or_call():
            return HeuristicAgentDecision(
                action="check_or_call",
                rationale=rationale,
            )

        raise ValueError("No legal heuristic action available.")

    def _calculate_strength(self, state, seat_index):
        hole_range = {frozenset(state.hole_cards[seat_index])}
        board_cards = tuple(
            card for board_dealing in state.board_cards for card in board_dealing
        )

        return calculate_hand_strength(
            sum(1 for is_active in state.statuses if is_active),
            hole_range,
            board_cards,
            2,
            5,
            cast(Deck, Deck.STANDARD),
            (StandardHighHand,),
            sample_count=self.sample_count,
        )


@dataclass
class HeuristicAgent:
    """Heuristic agent that decides based on its given policy"""

    seat_index: int
    name: str = "heuristic"
    tracked_hole_cards: tuple[str, ...] = ()
    policy: Any = field(default_factory=RandomPolicy)

    def reset_for_new_hand(self) -> None:
        """Clear per-hand memory."""
        self.tracked_hole_cards = ()

    def decide(self, state: State) -> HeuristicAgentDecision:
        """Choose an action from the configured policy."""
        return self.policy.decide(state, self.seat_index)


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

    if decision.action == "raise_to":
        target = decision.raise_to
        if target is not None and state.can_complete_bet_or_raise_to(target):
            operation = state.complete_bet_or_raise_to(target)
            amount = getattr(operation, "amount", target)
            return {
                "player": f"p{player_index}",
                "street": street,
                "action_taken": "raise_to",
                "amount": amount,
            }

    raise ValueError(f"Illegal heuristic action: {decision.model_dump()}")
