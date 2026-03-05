"""RAG-based similar hand retriever for poker decision support."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Hand strength categories for feature encoding
HAND_CATEGORIES = [
    "high_card", "pair", "two_pair", "three_of_a_kind",
    "straight", "flush", "full_house", "four_of_a_kind",
    "straight_flush", "royal_flush",
]

STREET_TO_IDX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

POSITION_TO_IDX = {
    "BTN": 0, "SB": 1, "BB": 2, "UTG": 3, "HJ": 4, "CO": 5,
}

# Card rank values for feature extraction
RANK_VALUES = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


@dataclass
class HandRecord:
    """A single historical hand record."""
    hand_id: str
    hole_cards: list[str]
    community_cards: list[str]
    street: str
    position: str
    pot_size: float
    action_taken: str
    amount: float = 0
    result: float = 0  # net chips won/lost
    num_players: int = 6
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_features(
    hole_cards: list[str],
    community_cards: list[str],
    street: str,
    position: str,
    pot_size: float,
    num_players: int = 6,
) -> list[float]:
    """
    Extract a numeric feature vector from a hand state.
    Used for similarity comparison.
    """
    features = []

    # Hole card features
    if len(hole_cards) >= 2:
        r1 = RANK_VALUES.get(hole_cards[0][0], 0) / 14.0
        r2 = RANK_VALUES.get(hole_cards[1][0], 0) / 14.0
        suited = 1.0 if hole_cards[0][1] == hole_cards[1][1] else 0.0
        paired = 1.0 if hole_cards[0][0] == hole_cards[1][0] else 0.0
        connected = 1.0 if abs(RANK_VALUES.get(hole_cards[0][0], 0) - RANK_VALUES.get(hole_cards[1][0], 0)) <= 2 else 0.0
        features.extend([max(r1, r2), min(r1, r2), suited, paired, connected])
    else:
        features.extend([0, 0, 0, 0, 0])

    # Street (one-hot)
    street_vec = [0.0] * 4
    idx = STREET_TO_IDX.get(street, 0)
    street_vec[idx] = 1.0
    features.extend(street_vec)

    # Position (normalized)
    pos_idx = POSITION_TO_IDX.get(position, 3)
    features.append(pos_idx / 5.0)

    # Pot size (normalized by big blind, capped)
    features.append(min(pot_size / 200.0, 1.0))

    # Number of community cards (normalized)
    features.append(len(community_cards) / 5.0)

    # Number of players (normalized)
    features.append(num_players / 6.0)

    # Community card features
    if community_cards:
        board_ranks = [RANK_VALUES.get(c[0], 0) for c in community_cards]
        features.append(max(board_ranks) / 14.0)
        features.append(min(board_ranks) / 14.0)

        # Board suits (how many of same suit)
        suits = [c[1] for c in community_cards]
        max_suit_count = max(suits.count(s) for s in set(suits))
        features.append(max_suit_count / 5.0)

        # Board connectedness
        sorted_ranks = sorted(board_ranks)
        gaps = sum(1 for i in range(1, len(sorted_ranks)) if sorted_ranks[i] - sorted_ranks[i-1] <= 2)
        features.append(gaps / max(len(sorted_ranks) - 1, 1))
    else:
        features.extend([0, 0, 0, 0])

    return features


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two feature vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class HandHistoryDB:
    """
    In-memory database of poker hand histories for RAG retrieval.
    Uses feature-based similarity (no external embedding model needed).
    """

    def __init__(self):
        self.records: list[HandRecord] = []
        self._features: list[list[float]] = []

    def add_hand(self, record: HandRecord) -> None:
        """Add a hand record to the database."""
        self.records.append(record)
        features = _extract_features(
            record.hole_cards,
            record.community_cards,
            record.street,
            record.position,
            record.pot_size,
            record.num_players,
        )
        self._features.append(features)

    def retrieve_similar(
        self,
        hole_cards: list[str],
        community_cards: list[str],
        street: str,
        position: str,
        pot_size: float,
        num_players: int = 6,
        k: int = 5,
    ) -> list[dict]:
        """
        Find the k most similar hands in the database.

        Returns:
            List of dicts with hand info and similarity score
        """
        if not self.records:
            return []

        query_features = _extract_features(
            hole_cards, community_cards, street, position, pot_size, num_players
        )

        scored = []
        for i, feat in enumerate(self._features):
            sim = _cosine_similarity(query_features, feat)
            scored.append((sim, i))

        scored.sort(reverse=True)
        results = []
        for sim, idx in scored[:k]:
            record = self.records[idx]
            results.append({
                "similarity": round(sim, 4),
                "hand_id": record.hand_id,
                "hole_cards": record.hole_cards,
                "community_cards": record.community_cards,
                "street": record.street,
                "position": record.position,
                "pot_size": record.pot_size,
                "action_taken": record.action_taken,
                "amount": record.amount,
                "result": record.result,
                "notes": record.notes,
            })

        return results

    def load_from_json(self, filepath: str) -> int:
        """Load hand records from a JSON file. Returns number of records loaded."""
        path = Path(filepath)
        if not path.exists():
            return 0

        with open(path) as f:
            data = json.load(f)

        count = 0
        records = data if isinstance(data, list) else data.get("hands", [])
        for entry in records:
            record = HandRecord(**entry)
            self.add_hand(record)
            count += 1

        return count

    def save_to_json(self, filepath: str) -> None:
        """Save all records to a JSON file."""
        data = [r.to_dict() for r in self.records]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_phh(self, phh_dir: str) -> int:
        """
        Load hands from PHH (Poker Hand History) format files.
        PHH is a standardized format from https://github.com/uoftcprg/phh-std.

        This is a simplified parser that extracts key information.
        Returns the number of records loaded.
        """
        count = 0
        phh_path = Path(phh_dir)

        if not phh_path.exists():
            return 0

        for phh_file in phh_path.glob("*.phh"):
            try:
                records = self._parse_phh_file(phh_file)
                for record in records:
                    self.add_hand(record)
                    count += 1
            except Exception:
                continue

        return count

    def _parse_phh_file(self, filepath: Path) -> list[HandRecord]:
        """Parse a single PHH file into HandRecord objects."""
        records = []
        with open(filepath) as f:
            content = f.read()

        # PHH files use TOML-like format
        # Extract key fields: players, hole cards, board cards, actions
        hand_id = filepath.stem
        hole_cards_map = {}
        board_cards = []
        actions = []

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("hole_cards"):
                # Parse hole cards assignments
                parts = line.split("=", 1)
                if len(parts) == 2:
                    # Format varies; store raw for now
                    pass
            elif line.startswith("actions"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    actions.append(parts[1].strip().strip('"'))

        # For each action/decision point, create a record
        # This is a basic parser; full PHH parsing would need the phh library
        if actions:
            record = HandRecord(
                hand_id=hand_id,
                hole_cards=[],
                community_cards=board_cards,
                street="preflop",
                position="unknown",
                pot_size=0,
                action_taken=actions[0] if actions else "unknown",
            )
            records.append(record)

        return records

    def __len__(self) -> int:
        return len(self.records)


def format_similar_hands(results: list[dict]) -> str:
    """Format retrieval results as a readable string for the LLM agent."""
    if not results:
        return "No similar hands found in database."

    lines = [f"Found {len(results)} similar hands:"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n  Hand {i} (similarity: {r['similarity']:.1%}):")
        lines.append(f"    Cards: {' '.join(r['hole_cards'])}")
        if r["community_cards"]:
            lines.append(f"    Board: {' '.join(r['community_cards'])}")
        lines.append(f"    Street: {r['street']}, Position: {r['position']}")
        lines.append(f"    Pot: {r['pot_size']}")
        lines.append(f"    Action: {r['action_taken']}" + (f" {r['amount']}" if r['amount'] else ""))
        lines.append(f"    Result: {'+' if r['result'] >= 0 else ''}{r['result']}")
        if r["notes"]:
            lines.append(f"    Notes: {r['notes']}")

    return "\n".join(lines)
