from dotenv import load_dotenv
load_dotenv()

from rag_retrieval import retrieve_similar_hands

# Fake a hand history like what your agent would produce
fake_hand_history = [
    {"player": "p2", "street": "preflop", "action_taken": "raise", "amount": 2},
    {"player": "p0", "street": "preflop", "action_taken": "call"},
    {"player": "p0", "street": "flop", "action_taken": "check"},
    {"player": "p2", "street": "flop", "action_taken": "bet", "amount": 3},
    {"player": "p0", "street": "flop", "action_taken": "call"},
]

# Fake a game state like what llm_view looks like
fake_llm_view = {
    "position": "BB",
    "hole_cards": ["King of Diamond", "Jack of Spade"],
    "board": ["King of Spade", "Seven of Heart", "Two of Diamond", "Jack of Club"],
    "street": "turn",
    "pot": 24,
    "available_actions": ["check", "bet", "fold"]
}

result = retrieve_similar_hands(fake_hand_history, fake_llm_view)
print(result)