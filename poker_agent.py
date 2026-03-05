import json
import os
from dotenv import load_dotenv

from pokerkit import State
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from action_entry import ActionEntry
from action_decision import PokerAgentDecision
from constants import MCP_PATH, MODEL, SYSTEM_PROMPT

load_dotenv() ## OPEN_API_KEY

class PokerMCPToolkit:
    def __init__(self):
        self.server_path = os.path.abspath(MCP_PATH)
        self.client = None

    async def get_tools(self):
        print(f"DEBUG: Connecting to MCP Server at {self.server_path}...")
        configs = {
            "poker_engine": {
                "command": "python",
                "args": [self.server_path],
                "transport": "stdio"
            }
        }
        self.client = MultiServerMCPClient(configs)
        return await self.client.get_tools()

class PokerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL, temperature=0)
        self.toolkit = PokerMCPToolkit()
        self.agent_executor = None

    async def initialize(self):
        mcp_tools = await self.toolkit.get_tools()
        print(f"DEBUG: Loaded tools: {[t.name for t in mcp_tools]}")

        self.agent_executor = create_agent(
            model=self.llm,
            tools=mcp_tools,
            system_prompt=SYSTEM_PROMPT,
            response_format=PokerAgentDecision
        )
        print("DEBUG: Agent Executor created and ready.")

    async def decide(self, llm_view: dict) -> PokerAgentDecision:
        
        input_prompt = f"Game State: {json.dumps(llm_view)}\n"
        
        print(f"\n--- AGENT START ({llm_view['street'].upper()}) ---")

        result = await self.agent_executor.ainvoke({"messages": [("user", input_prompt)]})

        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"DEBUG: [Calling MCP Tool] -> {tc['name']}({tc['args']})")
            
            elif msg.type == "tool":
                print(f"DEBUG: [MCP Output Received] -> {msg.content[:200]}...")

            elif msg.type == "ai" and msg.content:
                print(f"DEBUG: [AI Reasoning] -> {msg.content}")

        print("--- AGENT FINISHED ---")
        return result["structured_response"]

    async def cleanup(self):
        self.toolkit.client = None
        self.toolkit = None

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