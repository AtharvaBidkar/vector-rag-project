import faiss
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer


with open("emaar_properties.txt", "r", encoding="utf-8") as f:
    data = f.read()


def split_text(text, chunk_size=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks

chunks = split_text(data)


model = SentenceTransformer("all-MiniLM-L6-v2")


embeddings = model.encode(chunks)


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


while True:
    question = input("\nAsk your question: ")

    if question.lower() == "exit":
        break


    q_embedding = model.encode([question])

  
    distances, indices = index.search(np.array(q_embedding), k=3)

    context = "\n".join([chunks[i] for i in indices[0]])

    
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