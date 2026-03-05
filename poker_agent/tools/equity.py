"""Poker equity calculator using Monte Carlo simulation."""

from __future__ import annotations

import random
from itertools import combinations

from pokerkit import Card, Deck, StandardHighHand, calculate_equities, parse_range


# All 52 cards as strings
ALL_CARDS = [
    f"{r}{s}"
    for r in "23456789TJQKA"
    for s in "cdhs"
]


def monte_carlo_equity(
    hole_cards: list[str],
    community_cards: list[str],
    num_opponents: int = 1,
    num_simulations: int = 1000,
) -> dict:
    """
    Estimate win probability via Monte Carlo simulation.

    Args:
        hole_cards: Player's hole cards, e.g. ["Ah", "Kd"]
        community_cards: Current community cards, e.g. ["Jc", "Qs", "9h"]
        num_opponents: Number of opponents still in the hand
        num_simulations: Number of random simulations to run

    Returns:
        dict with keys: win_rate, tie_rate, lose_rate, equity
    """
    known_cards = set(hole_cards + community_cards)
    remaining_deck = [c for c in ALL_CARDS if c not in known_cards]
    cards_to_deal_board = 5 - len(community_cards)
    cards_per_opponent = 2

    wins = 0
    ties = 0
    total = 0

    for _ in range(num_simulations):
        # Shuffle remaining deck
        deck = list(remaining_deck)
        random.shuffle(deck)

        idx = 0
        # Deal remaining community cards
        sim_board = list(community_cards) + deck[idx:idx + cards_to_deal_board]
        idx += cards_to_deal_board

        # Deal opponent hole cards
        opponent_hands = []
        for _ in range(num_opponents):
            opp_hand = deck[idx:idx + cards_per_opponent]
            idx += cards_per_opponent
            opponent_hands.append(opp_hand)

        # Evaluate hands using pokerkit
        # from_game(hole_cards_str, board_cards_str)
        hero_hole_str = "".join(hole_cards)
        board_str = "".join(sim_board)
        hero_hand = StandardHighHand.from_game(hero_hole_str, board_str)

        best_opp_hand = None
        for opp in opponent_hands:
            opp_hole_str = "".join(opp)
            opp_hand = StandardHighHand.from_game(opp_hole_str, board_str)
            if best_opp_hand is None or opp_hand > best_opp_hand:
                best_opp_hand = opp_hand

        if best_opp_hand is None:
            wins += 1
        elif hero_hand > best_opp_hand:
            wins += 1
        elif hero_hand == best_opp_hand:
            ties += 1

        total += 1

    win_rate = wins / total if total > 0 else 0
    tie_rate = ties / total if total > 0 else 0
    lose_rate = 1 - win_rate - tie_rate
    equity = win_rate + tie_rate * 0.5

    return {
        "win_rate": round(win_rate, 4),
        "tie_rate": round(tie_rate, 4),
        "lose_rate": round(lose_rate, 4),
        "equity": round(equity, 4),
        "simulations": total,
    }


def pokerkit_equity(
    hole_cards: list[str],
    community_cards: list[str],
    num_opponents: int = 1,
    num_simulations: int = 1000,
) -> dict:
    """
    Calculate equity using PokerKit's built-in calculate_equities function.
    Uses random opponent ranges.

    Args:
        hole_cards: Player's hole cards, e.g. ["Ah", "Kd"]
        community_cards: Community cards
        num_opponents: Number of opponents
        num_simulations: Sample count for Monte Carlo

    Returns:
        dict with equity information
    """
    try:
        hero_range = parse_range("".join(hole_cards))
        # Use full range for opponents (all possible hands)
        opp_ranges = [parse_range("random")] * num_opponents

        board = Card.parse("".join(community_cards)) if community_cards else ()
        board_dealing_count = 5  # Texas Hold'em has 5 community cards total

        ranges = [hero_range] + opp_ranges
        equities = calculate_equities(
            ranges,
            board,
            2,  # hole_dealing_count (2 cards per player)
            board_dealing_count,
            Deck.STANDARD,
            (StandardHighHand,),
            sample_count=num_simulations,
        )

        return {
            "equity": round(equities[0], 4),
            "opponent_equities": [round(e, 4) for e in equities[1:]],
            "simulations": num_simulations,
        }
    except Exception as e:
        # Fallback to our Monte Carlo implementation
        return monte_carlo_equity(
            hole_cards, community_cards, num_opponents, num_simulations
        )


def format_equity_for_agent(result: dict) -> str:
    """Format equity result as a readable string for the LLM agent."""
    lines = [
        f"Equity: {result['equity']:.1%}",
    ]
    if "win_rate" in result:
        lines.extend([
            f"Win rate: {result['win_rate']:.1%}",
            f"Tie rate: {result['tie_rate']:.1%}",
            f"Lose rate: {result['lose_rate']:.1%}",
        ])
    lines.append(f"(based on {result['simulations']} simulations)")
    return "\n".join(lines)
