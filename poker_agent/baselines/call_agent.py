"""Call-station baseline agent: always checks or calls, never folds or raises."""

from __future__ import annotations

from ..game_state import GameState


class CallAgent:
    """Always checks or calls. Never folds, never raises."""

    def decide(self, game_state: GameState) -> tuple[str, int, str]:
        """Returns (action_type, amount, reasoning)."""
        for action in game_state.legal_actions:
            if action.action_type in ("check", "call"):
                return action.action_type, action.amount, "Call-station: always call."

        # Fallback: if somehow no check/call available, fold
        return "fold", 0, "Call-station: forced fold (no check/call available)."
