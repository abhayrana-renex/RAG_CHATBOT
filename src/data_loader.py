from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader

def load_all_documents(pdf_path: str) -> List[Any]:
    """
    Load a single PDF document.
    """
    
    pdf_file = Path(pdf_path)

    print(f"Processing: {pdf_file.name}")

    try:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["file_type"] = "pdf"

        print(f"✓ Loaded {len(documents)} pages")

        return documents

    except Exception as e:
        print(f"✗ Error: {e}")
        return []


# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data/Agmlo.pdf")
    print(f"Loaded {len(docs)} documents.")
    if docs:
        print("Example document:")
        print(docs[0])
