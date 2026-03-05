"""Equity-based heuristic agent: decides based on Monte Carlo equity thresholds."""

from __future__ import annotations

from ..game_state import GameState
from ..tools.equity import monte_carlo_equity


class HeuristicAgent:
    """
    Makes decisions based on equity thresholds:
    - equity >= raise_threshold -> raise
    - equity >= call_threshold  -> call
    - else                      -> fold
    """

    def __init__(
        self,
        call_threshold: float = 0.35,
        raise_threshold: float = 0.55,
        num_simulations: int = 500,
    ):
        self.call_threshold = call_threshold
        self.raise_threshold = raise_threshold
        self.num_simulations = num_simulations

    def decide(self, game_state: GameState) -> tuple[str, int, str]:
        """Returns (action_type, amount, reasoning)."""
        # Count active opponents
        num_opponents = sum(1 for i, a in enumerate(game_state.active_players)
                           if a and i != game_state.player_index)

        # Calculate equity
        equity_result = monte_carlo_equity(
            game_state.hole_cards,
            game_state.community_cards,
            num_opponents,
            self.num_simulations,
        )
        equity = equity_result["equity"]

        # Decision logic
        has_raise = any(a.action_type == "raise" for a in game_state.legal_actions)
        has_call = any(a.action_type in ("call", "check") for a in game_state.legal_actions)
        has_fold = any(a.action_type == "fold" for a in game_state.legal_actions)

        reasoning = f"Equity: {equity:.1%}"

        if equity >= self.raise_threshold and has_raise:
            raise_action = next(a for a in game_state.legal_actions if a.action_type == "raise")
            # Raise proportional to equity: min-raise for borderline, bigger for strong
            raise_frac = (equity - self.raise_threshold) / (1.0 - self.raise_threshold)
            amount = int(
                raise_action.min_amount +
                raise_frac * (raise_action.max_amount - raise_action.min_amount) * 0.5
            )
            amount = max(raise_action.min_amount, min(amount, raise_action.max_amount))
            return "raise", amount, f"{reasoning} -> raise to {amount}"

        if equity >= self.call_threshold and has_call:
            call_action = next(
                (a for a in game_state.legal_actions if a.action_type in ("call", "check")),
                None,
            )
            action = call_action.action_type if call_action else "call"
            return action, 0, f"{reasoning} -> {action}"

        # Check if we can check for free
        if has_call:
            check_action = next(
                (a for a in game_state.legal_actions if a.action_type == "check"),
                None,
            )
            if check_action:
                return "check", 0, f"{reasoning} -> check (free)"

        if has_fold:
            return "fold", 0, f"{reasoning} -> fold"

        return "check", 0, f"{reasoning} -> default check"
