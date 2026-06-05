import streamlit as st
from src.search import RAGSearch

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# ------------------------------------------------
# LOAD RAG SYSTEM ONCE
# ------------------------------------------------

@st.cache_resource
def load_rag():
    return RAGSearch(
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="mistral:latest"
    )

rag = load_rag()

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.title("⚙️ System Information")

    st.markdown("### Current Model")
    st.success("mistral:latest")

    st.markdown("### Indexed Chunks")

    try:
        chunk_count = len(rag.vectorstore.metadata)
    except:
        chunk_count = 0

    st.info(f"{chunk_count} Chunks")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("📚 Document Chat Assistant")

st.markdown(
    """
Ask questions about your PDF document.

Features:
- Semantic Search (FAISS)
- Local LLM (Ollama)
- Source References
- Chat History
"""
)

# ------------------------------------------------
# SHOW CHAT HISTORY
# ------------------------------------------------

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# ------------------------------------------------
# USER INPUT
# ------------------------------------------------

query = st.chat_input(
    "Ask a question about the document..."
)

# ------------------------------------------------
# PROCESS QUESTION
# ------------------------------------------------

if query:

    # Show user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # Assistant response

    with st.chat_message("assistant"):

        # -----------------------------
        # Retrieve source chunks
        # -----------------------------

        results = rag.vectorstore.query(
            query=query,
            top_k=8
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

        # -----------------------------
        # Same prompt as your backend
        # -----------------------------

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

        # -----------------------------
        # STREAMING RESPONSE
        # -----------------------------

        answer_box = st.empty()

        full_response = ""

        try:

            for chunk in rag.llm.stream(prompt):

                token = chunk.content

                if token:

                    full_response += token

                    answer_box.markdown(
                        full_response + "▌"
                    )

            answer_box.markdown(
                full_response
            )

        except Exception:

            response = rag.llm.invoke(prompt)

            full_response = response.content

            answer_box.markdown(
                full_response
            )

        # -----------------------------
        # Save chat
        # -----------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response
            }
        )

        # -----------------------------
        # Show Sources
        # -----------------------------

        with st.expander(
            "📄 Source Passages Used"
        ):

            for i, text in enumerate(
                texts,
                start=1
            ):

                st.markdown(
                    f"### Source Chunk {i}"
                )

                st.write(text)

                st.divider()

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")

st.caption(
    "Powered by FAISS + SentenceTransformers + Ollama + Streamlit"
)