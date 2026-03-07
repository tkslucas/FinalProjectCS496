import os
from datasets import load_dataset
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
load_dotenv()

dataset = load_dataset("RZ412/PokerBench", split="train[:10000]")

client = chromadb.PersistentClient(path="./chroma_db")
ef = OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small")
collection = client.get_or_create_collection("poker_hands", embedding_function=ef)

if collection.count() == 10000:
    print("DB already exists, skipping ingestion.")
else:
    batch_size = 500
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        collection.upsert(
            documents=batch["instruction"],
            metadatas=[{"optimal_action": a} for a in batch["output"]],
            ids=[f"hand_{i+j}" for j in range(len(batch["instruction"]))]
        )
        print(f"Ingested {min(i+batch_size, len(dataset))}/10000")

# Check how many hands are in the DB
print(f"Total hands in DB: {collection.count()}")

# Do a test query
results = collection.query(
    query_texts=["I'm in BTN with pocket aces, flop is K72 rainbow, pot is 10 chips, opponent checks"],
    n_results=3
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print("---")
    print(doc[:200])  # first 200 chars of the scenario
    print(f"Optimal action: {meta['optimal_action']}")