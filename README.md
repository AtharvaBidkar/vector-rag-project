import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

with open("sobha_properties.txt", "r", encoding="utf-8") as f:
    text = f.read()

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = split_text(text)

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

documents = chunks

query = input("Ask your question: ")

print("="*50)
print("STEP 1: Query received")
print("="*50)
print("Query:", query)

query_embedding = model.encode([query])

D, I = index.search(query_embedding.astype("float32"), k=3)

results = [documents[i] for i in I[0]]

print("="*50)
print("STEP 2: Retrieved context")
print("="*50)
print(results)

context = "\n".join(results)

prompt = f"""
Answer ONLY from the context below.

Context:
{context}

Question:
{query}
"""

response = ollama.chat(
    model='llama3',
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("="*50)
print("STEP 3: Sending to LLM")
print("="*50)

response = ollama.chat(
    model='llama3',
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("="*50)
print("FINAL ANSWER:")
print("="*50)
print(response['message']['content'])   
