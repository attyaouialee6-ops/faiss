
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

corpus = [
    "Artificial Intelligence is transforming industries.",
    "FAISS enables efficient vector similarity search.",
    "Machine learning models require high-quality data.",
    "Python is widely used for data science tasks.",
    "AI helps factories automate production and reduce costs.",
    "Deep learning is used in image and speech recognition.",
    "Data science helps businesses make better decisions.",
    "Robots use artificial intelligence to perform industrial tasks.",
    "Natural language processing allows computers to understand text.",
    "Cloud computing provides scalable computing resources.",
    "AI can improve healthcare diagnosis and patient care.",
    "Cybersecurity protects computer systems from digital attacks.",
    "Neural networks learn patterns from large datasets.",
    "Computer vision allows machines to analyze images.",
    "Predictive maintenance uses AI to detect equipment failures."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(corpus, convert_to_numpy=True)

embeddings = np.array(embeddings, dtype=np.float32)

d = embeddings.shape[1]

index = faiss.IndexFlatL2(d)

index.add(embeddings)

print("FAISS index created successfully!")
print("Number of documents:", index.ntotal)
print("Embedding dimension:", d)

queries = [
    "How can I use AI in industry?",
    "How does FAISS search vectors?",
    "How can Python help with data science?"
]

for query in queries:
    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = np.array(query_vector, dtype=np.float32)

    distances, indices = index.search(query_vector, k=2)

    print("\nQuery:", query)
    print("Top results:")

    for rank, idx in enumerate(indices[0], start=1):
        print(f"{rank}. {corpus[idx]}")
        print(f"   Distance: {distances[0][rank - 1]:.4f}")
