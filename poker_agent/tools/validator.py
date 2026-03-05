"""Legal action validator for the poker agent."""

from __future__ import annotations

from ..game_state import LegalAction


def get_legal_actions_from_state(state) -> list[LegalAction]:
    """
    Query the PokerKit state for all legal actions.

    Args:
        state: A PokerKit State object

    Returns:
        List of LegalAction objects describing valid moves
    """
    actions = []

    if state.can_fold():
        actions.append(LegalAction(action_type="fold"))

    if state.can_check_or_call():
        max_bet = max(state.bets) if state.bets else 0
        actor = state.actor_index
        player_bet = state.bets[actor] if actor is not None and actor < len(state.bets) else 0
        call_amount = max_bet - player_bet

        if call_amount > 0:
            actions.append(LegalAction(action_type="call", amount=call_amount))
        else:
            actions.append(LegalAction(action_type="check"))

    if state.can_complete_bet_or_raise_to():
        actions.append(LegalAction(
            action_type="raise",
            min_amount=state.min_completion_betting_or_raising_to_amount,
            max_amount=state.max_completion_betting_or_raising_to_amount,
        ))

    return actions


def validate_action(
    state,
    action_type: str,
    amount: int = 0,
) -> dict:
    """
    Check whether a proposed action is legal.

    Args:
        state: PokerKit State object
        action_type: "fold", "check", "call", or "raise"
        amount: Raise amount (only relevant for raise actions)

    Returns:
        dict with keys:
            valid: bool
            reason: str (empty if valid)
            corrected_action: dict | None (suggested fix if invalid)
    """
    if action_type == "fold":
        if state.can_fold():
            return {"valid": True, "reason": "", "corrected_action": None}
        else:
            return {
                "valid": False,
                "reason": "Cannot fold (no bet to fold to, can check instead).",
                "corrected_action": {"action_type": "check"},
            }

    if action_type in ("check", "call"):
        if state.can_check_or_call():
            return {"valid": True, "reason": "", "corrected_action": None}
        else:
            return {
                "valid": False,
                "reason": "Cannot check or call in this state.",
                "corrected_action": {"action_type": "fold"},
            }

    if action_type == "raise":
        if not state.can_complete_bet_or_raise_to():
            return {
                "valid": False,
                "reason": "Cannot raise (may be all-in or max raises reached).",
                "corrected_action": {"action_type": "call"},
            }

        min_raise = state.min_completion_betting_or_raising_to_amount
        max_raise = state.max_completion_betting_or_raising_to_amount

        if amount < min_raise:
            return {
                "valid": False,
                "reason": f"Raise amount {amount} is below minimum {min_raise}.",
                "corrected_action": {"action_type": "raise", "amount": min_raise},
            }

        if amount > max_raise:
            return {
                "valid": False,
                "reason": f"Raise amount {amount} exceeds maximum {max_raise}.",
                "corrected_action": {"action_type": "raise", "amount": max_raise},
            }

        return {"valid": True, "reason": "", "corrected_action": None}

    return {
        "valid": False,
        "reason": f"Unknown action type: {action_type}",
        "corrected_action": {"action_type": "check"},
    }


def format_legal_actions(actions: list[LegalAction]) -> str:
    """Format legal actions as a readable string for the LLM agent."""
    lines = ["Available actions:"]
    for a in actions:
        if a.action_type == "raise":
            lines.append(f"  - raise (min: {a.min_amount}, max: {a.max_amount})")
        elif a.action_type == "call":
            lines.append(f"  - call (amount: {a.amount})")
        else:
            lines.append(f"  - {a.action_type}")
    return "\n".join(lines)
