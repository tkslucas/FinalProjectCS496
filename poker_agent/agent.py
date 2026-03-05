"""GPT-4.1 tool-augmented poker agent using the OpenAI Agents SDK."""

from __future__ import annotations

import json
import re
from typing import Any

from agents import Agent, Runner, function_tool, RunContextWrapper

from .game_state import GameState
from .tools.equity import monte_carlo_equity, format_equity_for_agent
from .tools.validator import validate_action, format_legal_actions
from .tools.rag import HandHistoryDB, format_similar_hands
from .reasoning.logger import ReasoningLogger, ToolCall


SYSTEM_PROMPT = """\
You are an expert Texas Hold'em poker player. You make strategic decisions \
based on your hole cards, community cards, position, pot odds, and opponent \
tendencies.

Before each decision, you MUST:
1. Call calculate_equity to estimate your win probability
2. Call get_legal_actions to see what moves are available
3. Optionally call retrieve_similar_hands for historical context

Then reason step-by-step about:
- Your hand strength and equity
- Pot odds and implied odds
- Your position at the table
- The betting action so far
- What similar historical hands suggest

Finally, output your decision as JSON:
{"action": "fold|check|call|raise", "amount": <number_if_raising>, "reasoning": "<your reasoning>"}

Always output ONLY the JSON decision as your final message.
"""


class AgentContext:
    """Shared context accessible by tool functions during a decision."""

    def __init__(self):
        self.game_state: GameState | None = None
        self.hand_db: HandHistoryDB | None = None
        self._tool_calls: list[ToolCall] = []

    def reset(self, game_state: GameState, hand_db: HandHistoryDB | None = None):
        self.game_state = game_state
        self.hand_db = hand_db
        self._tool_calls = []


# Module-level context shared with tool functions
_ctx = AgentContext()


@function_tool
def calculate_equity(
    hole_cards: str,
    community_cards: str,
    num_opponents: int,
    num_simulations: int = 1000,
) -> str:
    """
    Calculate your win probability using Monte Carlo simulation.

    Args:
        hole_cards: Your hole cards separated by space, e.g. "Ah Kd"
        community_cards: Community cards separated by space, e.g. "Jc Qs 9h". Empty string if preflop.
        num_opponents: Number of opponents still in the hand
        num_simulations: Number of simulations to run (default 1000)
    """
    hc = hole_cards.split() if hole_cards.strip() else []
    cc = community_cards.split() if community_cards.strip() else []

    result = monte_carlo_equity(hc, cc, num_opponents, num_simulations)
    output = format_equity_for_agent(result)

    _ctx._tool_calls.append(ToolCall(
        tool_name="calculate_equity",
        inputs={"hole_cards": hole_cards, "community_cards": community_cards,
                "num_opponents": num_opponents},
        output=output,
    ))

    return output


@function_tool
def get_legal_actions() -> str:
    """Get the list of legal actions you can take right now."""
    if _ctx.game_state is None:
        return "Error: No game state available."

    output = format_legal_actions(_ctx.game_state.legal_actions)

    _ctx._tool_calls.append(ToolCall(
        tool_name="get_legal_actions",
        inputs={},
        output=output,
    ))

    return output


@function_tool
def retrieve_similar_hands(
    hole_cards: str,
    community_cards: str,
    pot_size: float,
    position: str,
) -> str:
    """
    Retrieve similar historical hands from the database for strategic context.

    Args:
        hole_cards: Your hole cards, e.g. "Ah Kd"
        community_cards: Community cards, e.g. "Jc Qs 9h"
        pot_size: Current pot size
        position: Your position, e.g. "BTN", "BB", "UTG"
    """
    if _ctx.hand_db is None or len(_ctx.hand_db) == 0:
        output = "No historical hand database available."
        _ctx._tool_calls.append(ToolCall(
            tool_name="retrieve_similar_hands",
            inputs={"hole_cards": hole_cards, "community_cards": community_cards},
            output=output,
        ))
        return output

    hc = hole_cards.split() if hole_cards.strip() else []
    cc = community_cards.split() if community_cards.strip() else []
    street = "preflop"
    if len(cc) >= 5:
        street = "river"
    elif len(cc) >= 4:
        street = "turn"
    elif len(cc) >= 3:
        street = "flop"

    results = _ctx.hand_db.retrieve_similar(
        hc, cc, street, position, pot_size, k=3
    )
    output = format_similar_hands(results)

    _ctx._tool_calls.append(ToolCall(
        tool_name="retrieve_similar_hands",
        inputs={"hole_cards": hole_cards, "community_cards": community_cards,
                "pot_size": pot_size, "position": position},
        output=output,
    ))

    return output


def _build_agent(model: str = "gpt-4.1") -> Agent:
    """Create the OpenAI Agent with poker tools."""
    return Agent(
        name="PokerAgent",
        model=model,
        instructions=SYSTEM_PROMPT,
        tools=[calculate_equity, get_legal_actions, retrieve_similar_hands],
    )


def _parse_agent_response(text: str) -> dict:
    """Extract the JSON action from the agent's response."""
    # Try to find JSON in the response
    json_match = re.search(r'\{[^{}]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: parse natural language
    text_lower = text.lower()
    if "fold" in text_lower:
        return {"action": "fold", "amount": 0, "reasoning": text}
    elif "raise" in text_lower or "bet" in text_lower:
        # Try to extract amount
        amount_match = re.search(r'raise.*?(\d+)', text_lower)
        amount = int(amount_match.group(1)) if amount_match else 0
        return {"action": "raise", "amount": amount, "reasoning": text}
    elif "call" in text_lower:
        return {"action": "call", "amount": 0, "reasoning": text}
    else:
        return {"action": "check", "amount": 0, "reasoning": text}


class PokerAgent:
    """
    Tool-augmented poker agent powered by GPT-4.1.

    Uses the OpenAI Agents SDK with three tools:
    - Equity calculator (Monte Carlo simulation)
    - Legal action validator
    - RAG similar hand retriever
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        hand_db: HandHistoryDB | None = None,
        reasoning_logger: ReasoningLogger | None = None,
    ):
        self.model = model
        self.agent = _build_agent(model)
        self.hand_db = hand_db
        self.reasoning_logger = reasoning_logger

    async def decide(self, game_state: GameState) -> tuple[str, int, str]:
        """
        Make a poker decision given the current game state.

        Returns:
            (action_type, amount, reasoning)
        """
        # Set up shared context for tools
        _ctx.reset(game_state, self.hand_db)

        # Build the prompt with game state
        prompt = game_state.to_prompt()

        # Run the agent
        result = await Runner.run(self.agent, input=prompt)
        response_text = result.final_output

        # Parse the response
        parsed = _parse_agent_response(response_text)
        action = parsed.get("action", "check")
        amount = parsed.get("amount", 0)
        reasoning = parsed.get("reasoning", response_text)

        # Log the decision
        if self.reasoning_logger:
            self.reasoning_logger.log_decision(
                player_index=game_state.player_index,
                game_state=game_state.to_dict(),
                tool_calls=list(_ctx._tool_calls),
                reasoning=reasoning,
                action_taken=action,
                action_amount=amount,
            )

        return action, amount, reasoning
