"""Evaluation metrics: bb/100, win rate, session statistics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlayerStats:
    """Tracks statistics for a single player/agent across a session."""
    player_index: int
    agent_name: str = ""
    hands_played: int = 0
    total_profit: float = 0  # in chips
    big_blind: int = 2
    hands_won: int = 0
    hands_lost: int = 0
    voluntarily_put_in_pot: int = 0  # VPIP hands
    preflop_raises: int = 0

    @property
    def bb_per_100(self) -> float:
        """Big blinds won per 100 hands (primary metric)."""
        if self.hands_played == 0:
            return 0.0
        return (self.total_profit / self.big_blind) / self.hands_played * 100

    @property
    def win_rate(self) -> float:
        """Fraction of hands won."""
        if self.hands_played == 0:
            return 0.0
        return self.hands_won / self.hands_played

    @property
    def vpip(self) -> float:
        """Voluntarily Put money In Pot percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.voluntarily_put_in_pot / self.hands_played

    @property
    def pfr(self) -> float:
        """Pre-Flop Raise percentage."""
        if self.hands_played == 0:
            return 0.0
        return self.preflop_raises / self.hands_played

    def record_hand(self, payoff: float) -> None:
        """Record the result of a single hand."""
        self.hands_played += 1
        self.total_profit += payoff
        if payoff > 0:
            self.hands_won += 1
        elif payoff < 0:
            self.hands_lost += 1

    def summary(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "hands_played": self.hands_played,
            "total_profit": self.total_profit,
            "bb_per_100": round(self.bb_per_100, 2),
            "win_rate": round(self.win_rate, 4),
            "vpip": round(self.vpip, 4),
            "pfr": round(self.pfr, 4),
        }


class SessionTracker:
    """Tracks all player statistics across a multi-hand session."""

    def __init__(self, num_players: int, big_blind: int = 2, agent_names: list[str] | None = None):
        self.num_players = num_players
        self.big_blind = big_blind
        names = agent_names or [f"Player_{i}" for i in range(num_players)]
        self.players = [
            PlayerStats(player_index=i, agent_name=names[i], big_blind=big_blind)
            for i in range(num_players)
        ]
        self.hand_count = 0

    def record_hand(self, payoffs: list[float]) -> None:
        """Record results from one hand."""
        self.hand_count += 1
        for i, payoff in enumerate(payoffs):
            if i < len(self.players):
                self.players[i].record_hand(payoff)

    def get_leaderboard(self) -> list[dict]:
        """Get players sorted by bb/100."""
        return sorted(
            [p.summary() for p in self.players],
            key=lambda x: x["bb_per_100"],
            reverse=True,
        )

    def print_summary(self) -> str:
        """Format a readable summary table."""
        lines = [
            f"\n{'='*60}",
            f"Session Summary: {self.hand_count} hands played",
            f"{'='*60}",
            f"{'Agent':<20} {'bb/100':>8} {'Profit':>8} {'Win%':>6} {'Hands':>6}",
            f"{'-'*60}",
        ]
        for entry in self.get_leaderboard():
            lines.append(
                f"{entry['agent_name']:<20} "
                f"{entry['bb_per_100']:>8.2f} "
                f"{entry['total_profit']:>8.1f} "
                f"{entry['win_rate']*100:>5.1f}% "
                f"{entry['hands_played']:>6}"
            )
        lines.append(f"{'='*60}")
        return "\n".join(lines)
