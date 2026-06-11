from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class Embedder:

    def __init__(self):
        self.model = SentenceTransformer( "all-MiniLM-L6-v2" )

    def chunk_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap=100)
        return splitter.split_documents(documents)

    def create_embeddings(self, texts):
        return self.model.encode(texts)
