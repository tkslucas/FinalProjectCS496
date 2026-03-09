import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp.server.fastmcp import FastMCP
from rag_retrieval import retrieve_similar_hands

mcp = FastMCP("rag_retriever")

@mcp.tool()
def get_similar_hands(hand_history: list[dict], llm_view: dict) -> str:
    """
    Search a database of 10,000 solver-optimal poker hands for situations 
    similar to the current one. Returns the 3 most similar scenarios with 
    the solver's recommended action for each.
    
    Use this tool when:
    - You are unsure about the optimal action in the current situation
    - You want to validate your decision against solver recommendations
    - The pot is large and the decision is high stakes
    
    Args:
        hand_history: The list of actions taken so far this hand
        llm_view: The current game state dict
    
    Returns:
        3 similar scenarios with solver-optimal actions
    """
    return retrieve_similar_hands(hand_history, llm_view)

if __name__ == "__main__":
    mcp.run()