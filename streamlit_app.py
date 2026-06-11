import streamlit as st
from src.search import RAGSearch

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# Load RAG
# -------------------------

@st.cache_resource
def load_rag():
    return RAGSearch(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="mistral:latest"
    )

rag = load_rag()

# -------------------------
# Session State
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Sidebar
# -------------------------

with st.sidebar:

    st.title("⚙️ System Info")

    st.write("**Model:**")
    st.success("mistral:latest")

    st.write("**Embedding Model:**")
    st.info("all-MiniLM-L6-v2")

    try:
        chunk_count = len(rag.vectorstore.metadata)
    except:
        chunk_count = 0

    st.write("**Indexed Chunks:**")
    st.success(chunk_count)

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# -------------------------
# Header
# -------------------------

st.title("📚 RAG Document Assistant")
st.caption("Ask questions about your PDF document")

# -------------------------
# Show Chat History
# -------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat Input
# -------------------------

query = st.chat_input(
    "Ask a question..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        answer = rag.search_and_summarize(
            query=query,
            top_k=8
        )

        full_response = ""

        for word in answer.split():

            full_response += word + " "

            response_placeholder.markdown(
                full_response + "▌"
            )

        response_placeholder.markdown(
            full_response
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )

    # -------------------------
    # Sources
    # -------------------------

    results = rag.vectorstore.query(
        query,
        top_k=8
    )

    with st.expander("📄 Source Chunks"):

        for i, result in enumerate(results, start=1):

            metadata = result.get(
                "metadata",
                {}
            )

            text = metadata.get(
                "text",
                ""
            )

            st.markdown(
                f"### Chunk {i}"
            )

            st.write(text[:1000])

            st.divider()

# -------------------------
# Footer
# -------------------------

st.markdown("---")

st.caption(
    "Powered by FAISS + SentenceTransformers + Ollama + Streamlit"
)
