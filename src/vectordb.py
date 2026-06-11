import os
import faiss
import pickle
import numpy as np


class VectorDB:

    def __init__(self):

        os.makedirs(
            "vectordb",
            exist_ok=True
        )

        self.index = None
        self.metadata = []

    def create_index(self, dimension):

        self.index = faiss.IndexFlatL2(
            dimension
        )

    def add(self, embeddings, metadata):

        self.index.add(
            np.array(
                embeddings
            ).astype("float32")
        )

        self.metadata.extend(
            metadata
        )

    def save(self):

        faiss.write_index(
            self.index,
            "vectordb/faiss.index"
        )

        with open(
            "vectordb/metadata.pkl",
            "wb"
        ) as f:

            pickle.dump(
                self.metadata,
                f
            )

    def load(self):

        self.index = faiss.read_index(
            "vectordb/faiss.index"
        )

        with open(
            "vectordb/metadata.pkl",
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

    def search(
        self,
        query_embedding,
        k=3
    ):

        D, I = self.index.search(
            query_embedding,
            k
        )

        return I