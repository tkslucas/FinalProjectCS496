import asyncio
import contextlib
import io
import os

from environment import (
    _street_name,
    build_heuristic_table,
    build_llm_agent_allowed_view,
    build_state,
    print_hand_summary,
)
from action_entry import ActionEntry
from heuristic_agent import apply_heuristic_agent_decision
from poker_agent import PokerAgent, apply_poker_agent_decision
from performance_tracker import PerformanceTracker
from logger import HandLogger
from constants import BIG_BLIND, LOG_DIR, NUM_HANDS, PLAYER_COUNT

async def main():

    poker_agent = PokerAgent()
    await poker_agent.initialize()

    player_names = {
        0: poker_agent.name,
        1: "heuristic_1",
        2: "heuristic_2",
        3: "heuristic_3",
    }

    tracker = PerformanceTracker(player_names, BIG_BLIND)
    logger = HandLogger(LOG_DIR)

    print(f"========== Starting Evaluation for {NUM_HANDS} hands ==========")

    os.makedirs("results", exist_ok=True)
    summary_file = open("results/hand_summaries.txt", "w")

    for hand_num in range(1, NUM_HANDS + 1):
        poker_agent_seat = (hand_num - 1) % PLAYER_COUNT
        heuristic_agents = build_heuristic_table(PLAYER_COUNT, poker_agent_seat)
        seat_to_identity = {poker_agent_seat: 0}
        heuristic_identity = 1
        for seat in range(PLAYER_COUNT):
            if seat == poker_agent_seat:
                continue
            seat_to_identity[seat] = heuristic_identity
            heuristic_identity += 1

        print(f"========== STARTING HAND {hand_num} ==========")
        print(f"Poker agent seat: {poker_agent_seat}")
        print(f"Heuristic seats: {sorted(heuristic_agents.keys())}")

        logger.start_new_hand()
        state = build_state()
        hand_action_history: list[ActionEntry] = []

        poker_agent.reset_for_new_hand()
        for agent in heuristic_agents.values():
            agent.reset_for_new_hand()

        action_num = 1
        while state.status:
            actor = state.actor_index
            if actor is None:
                break

            current_street = _street_name(state.street_index)

            if actor == poker_agent_seat:
                llm_view = build_llm_agent_allowed_view(
                    state,
                    poker_agent_seat,
                    hand_action_history,
                )
                decision = await poker_agent.decide(llm_view)
                action_entry = apply_poker_agent_decision(
                    state,
                    decision,
                    street=current_street,
                )
                hand_action_history.append(action_entry)
                logger.log_decision(llm_view, decision)
            else:
                heuristic_decision = heuristic_agents[actor].decide(state)
                action_entry = apply_heuristic_agent_decision(
                    state,
                    heuristic_decision,
                    street=current_street,
                )
                hand_action_history.append(action_entry)

            action_num += 1

        identity_payoffs = [0.0] * PLAYER_COUNT
        for seat, payoff in enumerate(state.payoffs):
            identity_payoffs[seat_to_identity[seat]] = payoff

        tracker.record_hand(hand_num, identity_payoffs)
        logger.log_final_result(state, seat_to_identity)
        print(f"========== ENDING HAND {hand_num} ==========")

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_hand_summary(identity_payoffs, tracker)
        summary_text = buffer.getvalue()
        print(summary_text, end="")
        summary_file.write(f"========== HAND {hand_num} SUMMARY ==========\n")
        summary_file.write(summary_text)
        summary_file.write("\n")

    summary_file.close()

    tracker.report()
    tracker.plot_results()

    await poker_agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
