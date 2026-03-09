import os
import json
import time
from typing import Any
from pokerkit import State
from action_decision import PokerAgentDecision

class HandLogger:
    """Logs hands"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.current_hand_id: int | None = None
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def start_new_hand(self):
        """Generates a new hand ID based on timestamp."""
        self.current_hand_id = int(time.time() * 1000)
        hand_path = self._get_hand_path()
        if not os.path.exists(hand_path):
            os.makedirs(hand_path)

    def _get_hand_path(self) -> str:
        return os.path.join(self.log_dir, f"hand_{self.current_hand_id}")

    def log_decision(self, llm_view: dict[str, Any], decision: PokerAgentDecision):
        """Logs the agent's decision and the state it saw."""
        if self.current_hand_id is None:
            return

        street = llm_view.get('street', 'unknown')
        filepath = os.path.join(self._get_hand_path(), f"{street}.json")
        
        log_entry = {
            "decision": {
                "action": decision.action,
                "amount": decision.raise_to,
                "reasoning_chain": decision.reasoning_chain
            },
            "llm_view": llm_view
        }
        
        with open(filepath, "w") as f:
            json.dump(log_entry, f, indent=4)

    def log_final_result(self, state: State):
        """Logs the final payoffs and stacks for the hand."""
        if self.current_hand_id is None:
            return

        filepath = os.path.join(self._get_hand_path(), "result.json")
        result_data = {
            "final_stacks": list(state.stacks),
            "payoffs": list(state.payoffs),
        }

        with open(filepath, "w") as f:
            json.dump(result_data, f, indent=4)