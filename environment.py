from pprint import pformat
from typing import cast

from pokerkit import Automation, NoLimitTexasHoldem, State

from action_entry import ActionEntry
from constants import (
    ANTE,
    BIG_BLIND,
    MIN_BET,
    PLAYER_COUNT,
    POKER_AGENT_SEAT,
    SMALL_BLIND,
    STARTING_STACK,
    USES_UNIFORM_ANTES,
)
from heuristic_agent import HandStrengthPolicy, HeuristicAgent
from performance_tracker import PerformanceTracker


########### HELPERS ###########
def _card_strings(cards) -> list[str]:
    return [str(card) for card in cards]


def _seat_label(seat_index: int | None) -> str | None:
    return None if seat_index is None else f"p{seat_index}"


def _street_name(street_index: int | None) -> str:
    if street_index is None:
        return "hand_over"
    names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
    return names.get(street_index, f"street_{street_index}")


########### ENVIRONMENT SETUP & INTERACTION ###########
def build_state():
    """Game configurations"""
    automations = cast(
        tuple[Automation, ...],
        (
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
        ),
    )

    return NoLimitTexasHoldem.create_state(
        automations,
        USES_UNIFORM_ANTES,
        ANTE,
        (SMALL_BLIND, BIG_BLIND),
        MIN_BET,
        (STARTING_STACK, STARTING_STACK, STARTING_STACK, STARTING_STACK),
        PLAYER_COUNT,
    )


def build_heuristic_table(player_count: int):
    """Create Heuristic agents for evaluation"""
    return {
        i: HeuristicAgent(seat_index=i, name=f"{policy.name}_p{i}", policy=policy)
        for i in range(player_count)
        if i != POKER_AGENT_SEAT
        # or RandomPolicy
        for policy in [HandStrengthPolicy()]
    }


def build_simulator_view(state: State) -> dict:
    """Everything the simulator knows right now."""
    actor = state.actor_index
    legal_actions = {
        "can_fold": state.can_fold() if actor is not None else False,
        "can_check_or_call": state.can_check_or_call() if actor is not None else False,
        "can_complete_bet_or_raise_to": (
            state.can_complete_bet_or_raise_to() if actor is not None else False
        ),
    }

    return {
        "status": state.status,
        "street_index": state.street_index,
        "actor_index": actor,
        "board_cards": _card_strings(state.board_cards),
        "hole_cards_by_player": {
            f"p{i}": _card_strings(cards) for i, cards in enumerate(state.hole_cards)
        },
        "statuses": list(state.statuses),
        "bets": list(state.bets),
        "stacks": list(state.stacks),
        "payoffs": list(state.payoffs),
        "pot_amounts": list(state.pot_amounts),
        "total_pot_amount": state.total_pot_amount,
        "checking_or_calling_amount": (
            state.checking_or_calling_amount if actor is not None else None
        ),
        "min_raise_to": (
            state.min_completion_betting_or_raising_to_amount
            if actor is not None
            else None
        ),
        "max_raise_to": (
            state.max_completion_betting_or_raising_to_amount
            if actor is not None
            else None
        ),
        "legal_actions_for_actor": legal_actions,
        "operation_count": len(state.operations),
        "last_operation": repr(state.operations[-1]) if state.operations else None,
    }


def build_llm_agent_allowed_view(
    state: State,
    hand_action_history: list[ActionEntry] | None = None,
) -> dict:
    """What the agent is allowed to see"""
    actor = state.actor_index
    is_poker_agent_turn = actor == POKER_AGENT_SEAT
    poker_agent_options = None
    if is_poker_agent_turn:
        poker_agent_options = {
            "to_call": state.checking_or_calling_amount,
            "min_raise_to": state.min_completion_betting_or_raising_to_amount,
            "max_raise_to": state.max_completion_betting_or_raising_to_amount,
            "can_fold": state.can_fold(),
            "can_check_or_call": state.can_check_or_call(),
            "can_complete_bet_or_raise_to": state.can_complete_bet_or_raise_to(),
        }

    return {
        "hand_active": state.status,
        "street": _street_name(state.street_index),
        "board_cards": _card_strings(state.board_cards),
        "whose_turn": _seat_label(actor),
        "is_poker_agent_turn": is_poker_agent_turn,
        "poker_agent_position": _seat_label(POKER_AGENT_SEAT),
        "active_players": [
            _seat_label(i) for i, is_active in enumerate(state.statuses) if is_active
        ],
        "stacks_by_player": {f"p{i}": stack for i, stack in enumerate(state.stacks)},
        "current_bets_by_player": {f"p{i}": bet for i, bet in enumerate(state.bets)},
        "pot_total": state.total_pot_amount,
        "pot_breakdown": list(state.pot_amounts),
        "hand_action_history": hand_action_history or [],
        "poker_agent_options_when_in_turn": poker_agent_options,
        "poker_agent_hole_cards": _card_strings(state.hole_cards[POKER_AGENT_SEAT]),
    }


########### PRINTING ###########
def print_state_views(state: State, hand_action_history: list[ActionEntry]) -> None:
    print("-----Simulator View-----")
    print(pformat(build_simulator_view(state), sort_dicts=False))
    if state.actor_index == POKER_AGENT_SEAT:
        print("\n------LLM Agent View-----\n")
        print(
            pformat(
                build_llm_agent_allowed_view(state, hand_action_history),
                sort_dicts=False,
            )
        )


def print_hand_summary(payoffs: list[float], tracker: PerformanceTracker):
    bb = tracker.big_blind

    hand_bb = tuple(
        f"{tracker.names[i]}: {p / bb:+.2f} BB" for i, p in enumerate(payoffs)
    )
    cum_bb = tuple(
        f"{tracker.names[i]}: {tracker.history[i][-1]:+.2f} BB"
        for i in range(PLAYER_COUNT)
    )

    print(f"Hand Payoffs (BB): {hand_bb}")
    print(f"Cumulative Total (BB): {cum_bb}")
