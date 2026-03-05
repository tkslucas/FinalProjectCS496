"""Entry point for running poker agent simulations."""

from __future__ import annotations

import argparse
import asyncio
import sys

from .environment import PokerEnvironment
from .game_state import GameState
from .baselines.random_agent import RandomAgent
from .baselines.call_agent import CallAgent
from .baselines.heuristic_agent import HeuristicAgent
from .evaluation.metrics import SessionTracker
from .reasoning.logger import ReasoningLogger


AGENT_TYPES = {
    "random": lambda: RandomAgent(),
    "call": lambda: CallAgent(),
    "heuristic": lambda: HeuristicAgent(),
}


def create_agent(agent_type: str, **kwargs):
    """Create an agent by type name."""
    if agent_type == "llm":
        # Lazy import to avoid requiring openai-agents when not needed
        from .agent import PokerAgent
        return PokerAgent(**kwargs)

    if agent_type in AGENT_TYPES:
        return AGENT_TYPES[agent_type]()

    raise ValueError(f"Unknown agent type: {agent_type}. Options: {list(AGENT_TYPES.keys()) + ['llm']}")


async def run_hand(
    env: PokerEnvironment,
    agents: list,
    reasoning_logger: ReasoningLogger | None = None,
) -> dict:
    """Play a single hand and return results."""
    env.new_hand()

    if reasoning_logger:
        reasoning_logger.start_hand(env.hand_id, env.num_players, env.big_blind)

    while not env.is_hand_over:
        player_idx = env.current_player
        if player_idx is None:
            break

        agent = agents[player_idx]
        game_state = env.get_game_state(player_idx)

        # Get decision from agent
        if asyncio.iscoroutinefunction(getattr(agent, 'decide', None)):
            action, amount, reasoning = await agent.decide(game_state)
        else:
            action, amount, reasoning = agent.decide(game_state)

        # Apply action
        env.take_action(action, amount)

    results = env.get_results()

    if reasoning_logger:
        reasoning_logger.end_hand(results)

    return results


async def run_session(
    num_hands: int = 100,
    num_players: int = 6,
    starting_stack: int = 200,
    small_blind: int = 1,
    big_blind: int = 2,
    agent_configs: list[str] | None = None,
    verbose: bool = False,
) -> SessionTracker:
    """Run a full session of multiple hands."""
    # Default: one heuristic agent, rest are random
    if agent_configs is None:
        agent_configs = ["heuristic"] + ["random"] * (num_players - 1)

    if len(agent_configs) != num_players:
        raise ValueError(f"Need {num_players} agent configs, got {len(agent_configs)}")

    # Create agents
    agents = [create_agent(config) for config in agent_configs]
    agent_names = [f"{config}_{i}" for i, config in enumerate(agent_configs)]

    # Set up environment, tracker, logger
    env = PokerEnvironment(num_players, starting_stack, small_blind, big_blind)
    tracker = SessionTracker(num_players, big_blind, agent_names)
    logger = ReasoningLogger()

    for hand_num in range(num_hands):
        try:
            results = await run_hand(env, agents, logger)
            payoffs = results.get("payoffs", [0] * num_players)
            tracker.record_hand(payoffs)
            env.update_stacks()
            env.advance_dealer()

            if verbose and (hand_num + 1) % 10 == 0:
                print(f"Hand {hand_num + 1}/{num_hands} complete")

        except Exception as e:
            if verbose:
                print(f"Hand {hand_num + 1} error: {e}")
            # Reset stacks for next hand if error
            env._stacks = [starting_stack] * num_players
            continue

    # Export reasoning logs
    log_path = logger.export_session()
    if verbose:
        print(f"\nReasoning logs exported to: {log_path}")

    return tracker


def main():
    parser = argparse.ArgumentParser(description="Run poker agent simulations")
    parser.add_argument("-n", "--num-hands", type=int, default=100,
                        help="Number of hands to play (default: 100)")
    parser.add_argument("-p", "--num-players", type=int, default=6,
                        help="Number of players (2-6, default: 6)")
    parser.add_argument("-s", "--starting-stack", type=int, default=200,
                        help="Starting stack in chips (default: 200)")
    parser.add_argument("--small-blind", type=int, default=1,
                        help="Small blind (default: 1)")
    parser.add_argument("--big-blind", type=int, default=2,
                        help="Big blind (default: 2)")
    parser.add_argument("--agents", nargs="+", default=None,
                        help="Agent types for each seat (e.g., heuristic random random)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print progress updates")

    args = parser.parse_args()

    tracker = asyncio.run(run_session(
        num_hands=args.num_hands,
        num_players=args.num_players,
        starting_stack=args.starting_stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        agent_configs=args.agents,
        verbose=args.verbose,
    ))

    print(tracker.print_summary())


if __name__ == "__main__":
    main()
