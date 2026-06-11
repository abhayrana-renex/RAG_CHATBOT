from langchain_ollama import ChatOllama


class RAG:

    def __init__(self):

        self.llm = ChatOllama(
            model="mistral:latest",
            temperature=0.2
        )

    def generate_answer(
        self,
        question,
        context
    ):

        prompt = f"""
Answer only from the context.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content