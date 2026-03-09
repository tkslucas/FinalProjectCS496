import json
import os
import time
from typing import Any

from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from pokerkit import State
from dotenv import load_dotenv

from action_entry import ActionEntry
from action_decision import PokerAgentDecision
from constants import MCP_PATH, MODEL, SYSTEM_PROMPT, LOG_DIR
from rag_retrieval import retrieve_similar_hands


load_dotenv()

class PokerAgent:
    def __init__(self):
        self.model = MODEL
        self.mcp_path = os.path.abspath(MCP_PATH)
        self.previous_response_id: str | None = None    
        self.mcp_server: MCPServerStdio | None = None
        self._agent: Agent | None = None
        self.log_dir = LOG_DIR
        self.current_hand_id: int | None = None

    async def initialize(self):
        self.mcp_server = MCPServerStdio(
            name="equity_calculator",
            params={
                "command": "python",
                "args": [self.mcp_path]
            }
        )

        await self.mcp_server.connect()
        
        self._agent = Agent(
            name="Poker Agent",
            model=self.model,
            instructions=SYSTEM_PROMPT,
            output_type=PokerAgentDecision,
            mcp_servers=[self.mcp_server]
        )

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def reset_for_new_hand(self) -> None:
        """Reset per-hand memory chain."""
        # this is for if we ever want to run multiple hands
        self.previous_response_id = None
        self.current_hand_id = int(time.time() * 1000)

    async def decide(self, llm_view: dict[str, Any]) -> PokerAgentDecision:

        hand_history = llm_view.get("hand_action_history", [])
        similar_hands = retrieve_similar_hands(hand_history, llm_view)

        print("\n=== RAG RESULTS ===")
        print(similar_hands)
        print("===================\n")
        
        input_prompt = (
            f"Game State: {json.dumps(llm_view)}\n\n"
            f"Here are similar hands from a poker solver database for reference:\n"
            f"{similar_hands}\n\n"
            f"Use these as guidance but make your own decision based on the full game state."
        )

        result = await Runner.run(
            self._agent,
            input_prompt,
            previous_response_id=self.previous_response_id,
        )

        """
        This is so that
        - decision 1 sees context from decision 0
        - decision 2 sees context from 0+1
        - decision 3 sees context from 0+1+2
        """
        self.previous_response_id = result.last_response_id
        decision = result.final_output_as(PokerAgentDecision)

        self._log_hand(llm_view, decision)
        return decision
    
    def _log_hand(self, state: dict, decision: PokerAgentDecision):
        hand_dir = os.path.join(self.log_dir, f"hand_{self.current_hand_id}")
        if not os.path.exists(hand_dir):
            os.makedirs(hand_dir)
        street = state.get('street', 'unknown')
        filename = f"{street}.json"
        filepath = os.path.join(hand_dir, filename)
        
        log_entry = {
            "decision": {
                "action": decision.action,
                "amount": decision.raise_to,
                "reasoning_chain": decision.reasoning_chain
            },
            "raw_state": state
        }
        
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=4)

    def _log_final_result(self, state: State):
        hand_dir = os.path.join(self.log_dir, f"hand_{self.current_hand_id}")
        filepath = os.path.join(hand_dir, "result.json")
        
        result_data = {
            "final_stacks": list(state.stacks),
            "payoffs": list(state.payoffs),
        }

        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=4)

    async def cleanup(self):
        await self.mcp_server.cleanup()
        self.mcp_server = None

def apply_poker_agent_decision(
    state: State,
    decision: PokerAgentDecision,
    *,
    street: str,
) -> ActionEntry:
    """Apply poker-agent decision and return an action entry."""
    player_index = state.actor_index
    if player_index is None:
        raise ValueError("No active actor when applying poker-agent decision.")

    if decision.action == "check_or_call" and state.can_check_or_call():
        operation = state.check_or_call()
        amount = getattr(operation, "amount", 0)
        if amount == 0:
            return {
                "player": f"p{player_index}",
                "street": street,
                "action_taken": "check",
            }
        return {
            "player": f"p{player_index}",
            "street": street,
            "action_taken": "call",
            "amount": amount,
        }

    if decision.action == "fold" and state.can_fold():
        state.fold()
        return {
            "player": f"p{player_index}",
            "street": street,
            "action_taken": "fold",
        }

    if decision.action == "raise_to":
        target = decision.raise_to
        if target is not None and state.can_complete_bet_or_raise_to(target):
            operation = state.complete_bet_or_raise_to(target)
            amount = getattr(operation, "amount", target)
            return {
                "player": f"p{player_index}",
                "street": street,
                "action_taken": "raise_to",
                "amount": amount,
            }

    raise ValueError(f"Illegal poker-agent action: {decision.model_dump()}")