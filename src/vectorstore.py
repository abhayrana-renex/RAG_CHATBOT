import os
import faiss
import pickle
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):

        print(f"[INFO] Building vector store from {len(documents)} raw documents...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = emb_pipe.chunk_documents(documents)

        embeddings = emb_pipe.embed_chunks(chunks)

        metadatas = []

        for chunk in chunks:
            metadatas.append(
                {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )

        self.add_embeddings(
            np.array(embeddings).astype("float32"),
            metadatas
        )

        self.save()

        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any] = None
    ):

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index.")

    def save(self):

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 8
    ):

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for idx, dist in zip(indices[0], distances[0]):

            if idx >= len(self.metadata):
                continue

            results.append(
                {
                    "index": int(idx),
                    "distance": float(dist),
                    "metadata": self.metadata[idx]
                }
            )

        return results

    def query(
        self,
        query: str,
        top_k: int = 8
    ):

        print(f"[INFO] Querying vector store for: '{query}'")

        query_embedding = self.model.encode(
            [query]
        ).astype("float32")

        return self.search(
            query_embedding,
            top_k=top_k
        )