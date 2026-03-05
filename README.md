# FinalProjectCS496

The Pokerkit state object internally updates the turn, order, phase. state.actor_index is who the current player is.
For example if we call `state.check_or_call()`, then the current player made that move and the state.actor_index
is updated to who should act next.

Player 0 will be the LLM Poker agent. Players 1, 2, 3 are heuristic-based agents (right now it's super basic)

## Environment & API key setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The OpenAI Agents SDK automatically reads `OPENAI_API_KEY` from environment variables.
