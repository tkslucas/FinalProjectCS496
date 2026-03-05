"""Reasoning chain logger for post-game analysis."""

from __future__ import annotations

import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ToolCall:
    """Record of a single tool invocation during a decision."""
    tool_name: str
    inputs: dict
    output: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DecisionRecord:
    """Full record of one agent decision point."""
    hand_id: str
    decision_index: int
    player_index: int
    street: str
    position: str
    hole_cards: list[str]
    community_cards: list[str]
    pot_size: int
    stacks: list[int]
    legal_actions: list[dict]
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""
    action_taken: str = ""
    action_amount: int = 0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class HandLog:
    """All decisions and outcomes for one hand."""
    hand_id: str
    num_players: int
    big_blind: int
    decisions: list[DecisionRecord] = field(default_factory=list)
    result: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ReasoningLogger:
    """
    Captures and stores reasoning chains for every agent decision.
    Supports export to JSON for post-game analysis.
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hands: list[HandLog] = []
        self._current_hand: HandLog | None = None
        self._decision_counter: int = 0

    def start_hand(self, hand_id: str, num_players: int, big_blind: int) -> None:
        """Begin logging a new hand."""
        self._current_hand = HandLog(
            hand_id=hand_id,
            num_players=num_players,
            big_blind=big_blind,
        )
        self._decision_counter = 0

    def log_decision(
        self,
        player_index: int,
        game_state: dict,
        tool_calls: list[ToolCall],
        reasoning: str,
        action_taken: str,
        action_amount: int = 0,
    ) -> None:
        """Log a single decision with its full reasoning chain."""
        if self._current_hand is None:
            return

        record = DecisionRecord(
            hand_id=self._current_hand.hand_id,
            decision_index=self._decision_counter,
            player_index=player_index,
            street=game_state.get("street", ""),
            position=game_state.get("position", ""),
            hole_cards=game_state.get("hole_cards", []),
            community_cards=game_state.get("community_cards", []),
            pot_size=game_state.get("pot_size", 0),
            stacks=game_state.get("stacks", []),
            legal_actions=game_state.get("legal_actions", []),
            tool_calls=tool_calls,
            reasoning=reasoning,
            action_taken=action_taken,
            action_amount=action_amount,
        )

        self._current_hand.decisions.append(record)
        self._decision_counter += 1

    def end_hand(self, result: dict) -> None:
        """Finalize the current hand with results."""
        if self._current_hand is None:
            return

        self._current_hand.result = result
        self.hands.append(self._current_hand)
        self._current_hand = None

    def export_session(self, filename: str | None = None) -> str:
        """Export all hand logs to a JSON file. Returns the filepath."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"

        filepath = self.output_dir / filename
        data = {
            "session_info": {
                "num_hands": len(self.hands),
                "exported_at": datetime.now().isoformat(),
            },
            "hands": [h.to_dict() for h in self.hands],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def get_summary(self) -> dict:
        """Get a summary of the logged session."""
        total_decisions = sum(len(h.decisions) for h in self.hands)
        tools_used = {}
        for hand in self.hands:
            for decision in hand.decisions:
                for tc in decision.tool_calls:
                    tools_used[tc.tool_name] = tools_used.get(tc.tool_name, 0) + 1

        return {
            "total_hands": len(self.hands),
            "total_decisions": total_decisions,
            "tools_used": tools_used,
            "avg_decisions_per_hand": total_decisions / max(len(self.hands), 1),
        }

    def get_hand_analysis(self, hand_id: str) -> dict | None:
        """Get detailed analysis data for a specific hand."""
        for hand in self.hands:
            if hand.hand_id == hand_id:
                return hand.to_dict()
        return None
