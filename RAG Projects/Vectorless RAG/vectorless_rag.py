import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.split("\n\n")  # chunk by paragraphs


class VectorlessRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        
        top_indices = scores.argsort()[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]


class VectorlessRAG:
    def __init__(self, retriever, client):
        self.retriever = retriever
        self.client = client

    def generate(self, query):
        context_docs = self.retriever.retrieve(query)
        context = "\n\n".join(context_docs)

        prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content


if __name__ == "__main__":


    def generate_answer(prompt):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {e}"

    print(generate_answer("Hello"))

    docs = load_data("emaar_properties.txt")

    retriever = VectorlessRetriever(docs)

    rag = VectorlessRAG(retriever, client)

    while True:
        query = input("\nAsk your question: ")
        if query.lower() in ["exit", "quit"]:
            break

        answer = rag.generate(query)
        print("\nAnswer:\n", answer)