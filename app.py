from __future__ import annotations

import streamlit as st

from rag_pipeline import RAGEngine


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource
def load_rag_engine() -> RAGEngine:
    """Load the RAG engine once and reuse it across interactions."""
    return RAGEngine()


def initialize_session_state() -> None:
    """Initialize Streamlit session state."""
    if "rag" not in st.session_state:
        st.session_state.rag = load_rag_engine()

    if "document_name" not in st.session_state:
        st.session_state.document_name = None

    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    initialize_session_state()

    st.title("📄 AI Document Assistant")
    st.caption(
        "Ask questions about your PDF using retrieval-augmented generation."
    )

    with st.sidebar:
        st.header("Document")

        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help="Upload a text-based PDF to build the searchable knowledge base.",
        )

        if uploaded_file is not None:

            if uploaded_file.name != st.session_state.document_name:
                with st.spinner("Processing document..."):
                    try:
                        pdf_bytes = uploaded_file.getvalue()

                        chunk_count = st.session_state.rag.build_index(
                            pdf_bytes
                        )

                        st.session_state.document_name = uploaded_file.name
                        st.session_state.chunk_count = chunk_count
                        st.session_state.messages = []

                        st.success(
                            f"Processed {chunk_count} text chunks."
                        )

                    except ValueError as exc:
                        st.error(str(exc))

                    except Exception as exc:
                        st.error(
                            f"Document processing failed: {exc}"
                        )

        if st.session_state.document_name:
            st.divider()

            st.write(
                f"**File:** {st.session_state.document_name}"
            )

            st.write(
                f"**Indexed chunks:** {st.session_state.chunk_count}"
            )

            st.info(
                "Ask questions about the uploaded document."
            )

    if not st.session_state.document_name:
        st.info(
            "Upload a PDF from the sidebar to start chatting."
        )

        st.markdown(
            """
### How it works

1. Upload a PDF.
2. The document is split into searchable chunks.
3. Semantic embeddings are generated.
4. FAISS retrieves the most relevant chunks.
5. GPT-OSS 20B generates an answer using the retrieved context.
            """
        )

        return

    # Display previous messages.
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("View sources"):
                        for chunk, score in sources:
                            st.markdown(
                                f"**Page {chunk.page}** "
                                f"(similarity: {score:.4f})"
                            )
                            st.caption(chunk.text)

    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching document and generating answer..."):
                try:
                    answer, sources = (
                        st.session_state.rag.generate_answer(
                            question
                        )
                    )

                    st.markdown(answer)

                    if sources:
                        with st.expander("View sources"):
                            for chunk, score in sources:
                                st.markdown(
                                    f"**Page {chunk.page}** "
                                    f"(similarity: {score:.4f})"
                                )
                                st.caption(chunk.text)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as exc:
                    error_message = (
                        f"Sorry, something went wrong: {exc}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )


if __name__ == "__main__":
    main()