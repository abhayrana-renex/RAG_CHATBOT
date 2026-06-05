import os
from dotenv import load_dotenv

from src.vectorstore import FaissVectorStore
from langchain_ollama import ChatOllama

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "mistral:latest"
    ):

        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        if not (
            os.path.exists(faiss_path)
            and os.path.exists(meta_path)
        ):

            print("[INFO] FAISS index not found. Building vector store...")

            from src.data_loader import load_all_documents

            docs = load_all_documents(
                "data/Agmlo.pdf"
            )

            self.vectorstore.build_from_documents(docs)

        else:

            print("[INFO] Loading existing FAISS index...")

            self.vectorstore.load()

        self.llm = ChatOllama(
            model=llm_model,
            temperature=0.2
        )

        print(f"[INFO] Ollama model initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 8
    ) -> str:

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        texts = []

        for result in results:

            metadata = result.get(
                "metadata",
                {}
            )

            text = metadata.get(
                "text",
                ""
            )

            if text:
                texts.append(text)

        context = "\n\n".join(texts)

        if not context:
            return "No relevant information found in the document."

#         prompt = f"""
# You are a precise legal document assistant for eBay's User Agreement.

# Instructions:
# 1. Answer only using the provided context.
# 2. Answer should be summarzied.
# 2. Quote relevant sections when useful.
# 3. If information is missing, say:
#    "I could not find that information in the document."
# 4. Do not make assumptions.
# 5. Answer in 3-5 sentences MAX
# 6. Use bullet points only if
        prompt = f"""
You are a document question-answering assistant.

Answer the user's question directly using only the context.

Do NOT summarize the entire context.

Extract only the information that answers the question.

If the question is broad, provide a concise answer.

If the answer is not found, say:
"I could not find that information in the document."

Question:
{query}

Context:
{context}

Answer:
"""
        
        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":

    rag_search = RAGSearch(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="mistral:latest"
    )

    while True:

        query = input(
            "\nAsk a question (or type 'exit'): "
        )

        if query.lower() == "exit":
            break

        answer = rag_search.search_and_summarize(
            query=query,
            top_k=3
        )

        print("\nAnswer:")
        print(answer)
