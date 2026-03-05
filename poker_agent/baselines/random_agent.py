"""Random baseline agent: picks a legal action uniformly at random."""

from __future__ import annotations

import random
from ..game_state import GameState


class RandomAgent:
    """Selects a random legal action each decision."""

    def decide(self, game_state: GameState) -> tuple[str, int, str]:
        """Returns (action_type, amount, reasoning)."""
        actions = game_state.legal_actions
        if not actions:
            return "check", 0, "No legal actions, defaulting to check."

        chosen = random.choice(actions)
        action_type = chosen.action_type
        amount = 0

        if action_type == "raise":
            amount = random.randint(chosen.min_amount, chosen.max_amount)

        return action_type, amount, f"Random action: {action_type}"
