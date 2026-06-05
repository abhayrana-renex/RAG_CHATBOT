from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

if __name__ == "__main__":

    # CHANGE THIS TO YOUR PDF FILE
    PDF_PATH = "data/Agmlo.pdf"

    # Create vector store
    store = FaissVectorStore(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2"
    )

    # Build index only if it doesn't exist
    if not os.path.exists("faiss_store/faiss.index"):

        print("[INFO] Building vector database...")

        docs = load_all_documents(PDF_PATH)

        store.build_from_documents(docs)

    else:

        print("[INFO] Loading existing vector database...")

        store.load()

    # Create RAG system
    rag_search = RAGSearch(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="mistral:latest"
    )

    print("\nRAG System Ready!")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Question: ")

        if query.lower() == "exit":
            break

        answer = rag_search.search_and_summarize(
            query=query,
            top_k=8
        )

        print("\nAnswer:")
        print(answer)
        print("\n" + "=" * 80 + "\n")