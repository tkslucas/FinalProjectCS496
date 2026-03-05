import json
import os
from typing import Any

from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from pokerkit import State
from dotenv import load_dotenv

from action_entry import ActionEntry
from action_decision import PokerAgentDecision
from constants import MCP_PATH, MODEL, SYSTEM_PROMPT

load_dotenv()

class PokerAgent:
    def __init__(self):
        self.model = MODEL
        self.mcp_path = os.path.abspath(MCP_PATH)
        self.previous_response_id: str | None = None    
        self.mcp_server: MCPServerStdio | None = None
        self._agent: Agent | None = None

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
        
        print("DEBUG: Agent and MCP Provider ready.")

    def reset_for_new_hand(self) -> None:
        """Reset per-hand memory chain."""
        # this is for if we ever want to run multiple hands
        self.previous_response_id = None

    async def decide(self, llm_view: dict[str, Any]) -> PokerAgentDecision:
        input_prompt = f"Game State: {json.dumps(llm_view)}\n"

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
        return result.final_output_as(PokerAgentDecision)
    
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
