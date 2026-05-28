# import basics
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# initialize pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# set the pinecone index
index_name = "sampleindex"
index = pc.Index(index_name)

# initialize embeddings model + vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# ---- METHOD 1: Similarity Search with Score ----
print("=== SIMILARITY SEARCH WITH SCORE ===")

results = vector_store.similarity_search_with_score(
    "what did you have for breakfast?",
    k=2,
    filter={"source": "tweet"},
)

for res, score in results:
    print(f"* {res.page_content} [{res.metadata}] -- Score: {score}")


# ---- METHOD 2: Retriever with Score Threshold ----
print("\n=== RETRIEVER WITH SCORE THRESHOLD ===")

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.6,
        "filter": {"source": "tweet"}
    },
)

results = retriever.invoke("what did you have for breakfast?")

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")