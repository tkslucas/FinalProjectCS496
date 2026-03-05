"""Structured game state representation for the poker agent."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


POSITION_NAMES = {
    2: ["SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "UTG"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
}

STREET_NAMES = ["preflop", "flop", "turn", "river"]


@dataclass
class LegalAction:
    action_type: str  # "fold", "check", "call", "raise"
    amount: int = 0
    min_amount: int = 0
    max_amount: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GameState:
    hand_id: str
    player_index: int
    num_players: int
    hole_cards: list[str]
    community_cards: list[str]
    pot_size: int
    stacks: list[int]
    player_bets: list[int]
    street: str
    position: str
    active_players: list[bool]
    legal_actions: list[LegalAction]
    betting_history: list[dict] = field(default_factory=list)
    big_blind: int = 2

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_prompt(self) -> str:
        """Format game state as a prompt string for the LLM agent."""
        lines = [
            "=== CURRENT GAME STATE ===",
            f"Hand ID: {self.hand_id}",
            f"Street: {self.street}",
            f"Position: {self.position} (Player {self.player_index})",
            f"Your hole cards: {' '.join(self.hole_cards)}",
            f"Community cards: {' '.join(self.community_cards) if self.community_cards else 'None (preflop)'}",
            f"Pot size: {self.pot_size}",
            f"Big blind: {self.big_blind}",
            "",
            "--- Player Info ---",
        ]

        positions = POSITION_NAMES.get(self.num_players, [f"P{i}" for i in range(self.num_players)])
        for i in range(self.num_players):
            status = "active" if self.active_players[i] else "folded"
            marker = " (you)" if i == self.player_index else ""
            lines.append(
                f"  {positions[i]}{marker}: stack={self.stacks[i]}, "
                f"bet={self.player_bets[i]}, {status}"
            )

        lines.append("")
        lines.append("--- Legal Actions ---")
        for action in self.legal_actions:
            if action.action_type == "raise":
                lines.append(
                    f"  raise: min={action.min_amount}, max={action.max_amount}"
                )
            elif action.action_type == "call":
                lines.append(f"  call: amount={action.amount}")
            else:
                lines.append(f"  {action.action_type}")

        if self.betting_history:
            lines.append("")
            lines.append("--- Betting History (this hand) ---")
            for entry in self.betting_history:
                lines.append(
                    f"  [{entry['street']}] {entry['position']}: "
                    f"{entry['action']} {entry.get('amount', '')}"
                )

        return "\n".join(lines)


def get_position_name(player_index: int, num_players: int, dealer_index: int = 0) -> str:
    """Get position name for a player given dealer position."""
    positions = POSITION_NAMES.get(num_players, [f"P{i}" for i in range(num_players)])
    relative = (player_index - dealer_index) % num_players
    return positions[relative] if relative < len(positions) else f"P{relative}"
