import faiss
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# STEP 1: Load data
# ----------------------------
with open("emaar_properties.txt", "r", encoding="utf-8") as f:
    data = f.read()

# ----------------------------
# STEP 2: Split into chunks
# ----------------------------
def split_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

chunks = split_text(data)

# ----------------------------
# STEP 3: Load embedding model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# STEP 4: Convert to vectors
# ----------------------------
embeddings = model.encode(chunks)

# ----------------------------
# STEP 5: Store in FAISS
# ----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ----------------------------
# STEP 6: Ask question
# ----------------------------
while True:
    question = input("\nAsk your question: ")

    if question.lower() == "exit":
        break

    # Convert question to vector
    q_embedding = model.encode([question])

    # Search top 3 similar chunks
    distances, indices = index.search(np.array(q_embedding), k=3)

    # Get context
    context = "\n".join([chunks[i] for i in indices[0]])

    # ----------------------------
    # STEP 7: Send to LLM
    # ----------------------------
    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.
If answer is not present, say: "Not found in document".

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="tinyllama",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:", response["message"]["content"])