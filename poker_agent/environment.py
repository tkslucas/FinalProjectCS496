"""PokerKit environment wrapper for No-Limit Texas Hold'em."""

from __future__ import annotations

import uuid
from typing import Any

from pokerkit import Automation, NoLimitTexasHoldem

from .game_state import GameState, LegalAction, STREET_NAMES, get_position_name


# Automate everything except player actions
AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_DEALING,
    Automation.BOARD_DEALING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


def card_to_str(card) -> str:
    """Convert a pokerkit Card to a short string like 'Ah', 'Kd'."""
    return repr(card)


class PokerEnvironment:
    """Wraps PokerKit to run No-Limit Texas Hold'em games."""

    def __init__(
        self,
        num_players: int = 6,
        starting_stack: int = 200,
        small_blind: int = 1,
        big_blind: int = 2,
    ):
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.state = None
        self.hand_id: str = ""
        self.dealer_index: int = 0
        self.betting_history: list[dict] = []
        self._stacks: list[int] = [starting_stack] * num_players

    def new_hand(self, stacks: list[int] | None = None) -> None:
        """Start a new hand. Optionally provide custom stacks."""
        if stacks is not None:
            self._stacks = list(stacks)

        # Cash game style: rebuy any busted players to starting stack
        for i in range(len(self._stacks)):
            if self._stacks[i] <= 0:
                self._stacks[i] = self.starting_stack

        self.hand_id = uuid.uuid4().hex[:8]
        self.betting_history = []

        self.state = NoLimitTexasHoldem.create_state(
            AUTOMATIONS,
            True,  # ante_trimming_status
            0,     # raw_antes
            (self.small_blind, self.big_blind),
            self.big_blind,  # min_bet
            tuple(self._stacks),
            self.num_players,
        )

    def get_game_state(self, player_index: int) -> GameState:
        """Build a structured GameState from the current PokerKit state."""
        s = self.state

        # Hole cards (only show this player's cards)
        hole_cards = []
        if player_index < len(s.hole_cards) and s.hole_cards[player_index]:
            hole_cards = [card_to_str(c) for c in s.hole_cards[player_index]]

        # Community cards (flatten nested list)
        community_cards = []
        for street_cards in s.board_cards:
            if hasattr(street_cards, '__iter__') and not isinstance(street_cards, str):
                for c in street_cards:
                    community_cards.append(card_to_str(c))
            else:
                community_cards.append(card_to_str(street_cards))

        # Street
        street_idx = s.street_index
        if street_idx is not None and street_idx < len(STREET_NAMES):
            street = STREET_NAMES[street_idx]
        else:
            street = "unknown"

        # Active players (not folded)
        active_players = list(s.statuses)

        # Legal actions
        legal_actions = self._get_legal_actions()

        # Position
        position = get_position_name(player_index, self.num_players, self.dealer_index)

        return GameState(
            hand_id=self.hand_id,
            player_index=player_index,
            num_players=self.num_players,
            hole_cards=hole_cards,
            community_cards=community_cards,
            pot_size=s.total_pot_amount,
            stacks=list(s.stacks),
            player_bets=list(s.bets),
            street=street,
            position=position,
            active_players=active_players,
            legal_actions=legal_actions,
            betting_history=list(self.betting_history),
            big_blind=self.big_blind,
        )

    def _get_legal_actions(self) -> list[LegalAction]:
        """Get legal actions for the current actor."""
        s = self.state
        actions = []

        if s.can_fold():
            actions.append(LegalAction(action_type="fold"))

        if s.can_check_or_call():
            # Determine if it's a check or call and the amount
            call_amount = 0
            if s.actor_index is not None:
                # Call amount = max bet at table - player's current bet
                max_bet = max(s.bets) if s.bets else 0
                player_bet = s.bets[s.actor_index] if s.actor_index < len(s.bets) else 0
                call_amount = max_bet - player_bet

            if call_amount > 0:
                actions.append(LegalAction(action_type="call", amount=call_amount))
            else:
                actions.append(LegalAction(action_type="check"))

        if s.can_complete_bet_or_raise_to():
            min_raise = s.min_completion_betting_or_raising_to_amount
            max_raise = s.max_completion_betting_or_raising_to_amount
            actions.append(LegalAction(
                action_type="raise",
                min_amount=min_raise,
                max_amount=max_raise,
            ))

        return actions

    def take_action(self, action_type: str, amount: int = 0) -> bool:
        """Apply an action. Returns True if successful."""
        s = self.state
        actor = s.actor_index

        # Record betting history
        position = get_position_name(actor, self.num_players, self.dealer_index)
        street_idx = s.street_index
        street = STREET_NAMES[street_idx] if street_idx is not None and street_idx < len(STREET_NAMES) else "unknown"

        try:
            if action_type == "fold":
                s.fold()
                self.betting_history.append({
                    "street": street, "position": position,
                    "player": actor, "action": "fold",
                })
            elif action_type in ("check", "call"):
                s.check_or_call()
                self.betting_history.append({
                    "street": street, "position": position,
                    "player": actor, "action": action_type,
                })
            elif action_type == "raise":
                # Clamp to legal range
                min_r = s.min_completion_betting_or_raising_to_amount
                max_r = s.max_completion_betting_or_raising_to_amount
                raise_amount = max(min_r, min(amount, max_r))
                s.complete_bet_or_raise_to(raise_amount)
                self.betting_history.append({
                    "street": street, "position": position,
                    "player": actor, "action": "raise",
                    "amount": raise_amount,
                })
            else:
                # Unknown action, default to check/call or fold
                if s.can_check_or_call():
                    s.check_or_call()
                elif s.can_fold():
                    s.fold()
                return False
            return True
        except Exception:
            # Fallback: check/call if possible, otherwise fold
            try:
                if s.can_check_or_call():
                    s.check_or_call()
                elif s.can_fold():
                    s.fold()
            except Exception:
                pass
            return False

    @property
    def current_player(self) -> int | None:
        """Index of the player whose turn it is, or None if hand is over."""
        return self.state.actor_index if self.state else None

    @property
    def is_hand_over(self) -> bool:
        return self.state is None or self.state.actor_index is None

    def get_results(self) -> dict:
        """Get hand results after completion."""
        if not self.is_hand_over:
            return {}

        s = self.state
        return {
            "hand_id": self.hand_id,
            "payoffs": list(s.payoffs),
            "final_stacks": list(s.stacks),
            "board": [card_to_str(c) for street in s.board_cards for c in (street if hasattr(street, '__iter__') and not isinstance(street, str) else [street])],
            "showdown_hands": {
                i: [card_to_str(c) for c in s.hole_cards[i]]
                for i in range(self.num_players)
                if s.hole_cards[i]  # Only show hands that went to showdown
            },
        }

    def advance_dealer(self) -> None:
        """Move the dealer button for the next hand."""
        self.dealer_index = (self.dealer_index + 1) % self.num_players

    def update_stacks(self) -> None:
        """Update internal stacks from the current state (call after hand ends)."""
        if self.state:
            self._stacks = list(self.state.stacks)
