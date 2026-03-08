import os
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

_collection = None
_openai_client = None

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="./chroma_db")
        ef = OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-3-small"
        )
        _collection = client.get_collection("poker_hands", embedding_function=ef)
    return _collection

def _get_openai():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client

def hand_history_to_natural_language(hand_history: list[dict], llm_view: dict) -> str:
    """Ask the LLM to describe the current hand in PokerBench-style natural language."""
    client = _get_openai()

    prompt = f"""Convert this poker hand history into a natural language description 
    in this exact style:

    "You are a specialist in playing 6-handed No Limit Texas Holdem. The following will be 
    a game scenario and you need to make the optimal decision. Here is a game summary: 
    The small blind is 0.5 chips and the big blind is 1 chips. Everyone started with 100 chips. 
    The player positions involved in this game are UTG, HJ, CO, BTN, SB, BB. In this hand, 
    your position is [POSITION], and your holding is [CARDS]. Before the flop, [ACTIONS]. 
    The flop comes [CARDS], then [ACTIONS]. Now it is your turn to make a move. 
    To remind you, the current pot size is [POT] chips, and your holding is [CARDS]."

    Hand history: {hand_history}
    Current game state: {llm_view}

    Return only the natural language description, nothing else."""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def retrieve_similar_hands(hand_history: list[dict], llm_view: dict, n: int = 3) -> str:
    """Convert hand to natural language, query RAG DB, return similar hands."""
    
    # Step 1: LLM describes the hand in natural language
    nl_description = hand_history_to_natural_language(hand_history, llm_view)
    
    # Step 2: Use that description to search the DB
    collection = _get_collection()
    results = collection.query(query_texts=[nl_description], n_results=n)

    output = [f"Your current hand described:\n{nl_description}\n"]
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        output.append(
            f"Similar scenario:\n{doc}\nSolver's optimal action: {meta['optimal_action']}"
        )
    return "\n\n---\n\n".join(output)