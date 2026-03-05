"""Heuristic poker agent.

It always prefers check/call, then falls back to folding.
"""

from dataclasses import dataclass

from pokerkit import State


@dataclass
class HeuristicAgent:
    """Basic heuristic policy."""

    seat_index: int
    name: str = "heuristic"
    tracked_hole_cards: tuple[str, ...] = ()

    def reset_for_new_hand(self) -> None:
        """Clear per-hand memory."""
        self.tracked_hole_cards = ()

    def observe_state(self, state: State) -> None:
        """Track this seat's current hole cards from simulator."""
        cards = state.hole_cards[self.seat_index]
        self.tracked_hole_cards = tuple(str(card) for card in cards)

    def act(
        self,
        state: State,
        recent_other_player_actions: list[dict[str, object]],
        street: str,
    ) -> str:
        """Apply one action and append a compact action record."""
        if state.can_check_or_call():
            operation = state.check_or_call()
            amount = getattr(operation, "amount", 0)
            item: dict[str, object] = {
                "player": f"p{self.seat_index}",
                "street": street,
                "action": "check" if amount == 0 else "call",
            }
            if amount != 0:
                item["amount"] = amount
            recent_other_player_actions.append(item)
            return "check_or_call"

        if state.can_fold():
            state.fold()
            recent_other_player_actions.append(
                {"player": f"p{self.seat_index}", "street": street, "action": "fold"}
            )
            return "fold"

        raise RuntimeError("No legal action found for heuristic agent.")
