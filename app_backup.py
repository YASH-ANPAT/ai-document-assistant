from __future__ import annotations

import html

import streamlit as st

from rag_pipeline import RAGEngine


st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="AD",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        /* ---------- GLOBAL ---------- */

        .stApp {
            background: #f5f3ee;
            color: #171717;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: #151515;
            border-right: 1px solid #292929;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 0;
        }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background: #1d1d1d;
            border: 1px solid #373737;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p {
            color: #d6d1c7;
        }

        /* ---------- SIDEBAR ---------- */

        .side-wrap {
            padding: 30px 22px 20px 22px;
            min-height: 93vh;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 44px;
        }

        .brand-mark {
            width: 40px;
            height: 40px;
            border: 1px solid #888278;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #f5f3ee;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.14em;
        }

        .brand-name {
            color: #f5f3ee;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.12em;
            line-height: 1.2;
        }

        .brand-note {
            color: #8f8a81;
            font-size: 10px;
            margin-top: 4px;
        }

        .side-label {
            color: #8f8a81;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 700;
            margin: 22px 0 10px 0;
        }

        .doc-card {
            background: #1d1d1d;
            border: 1px solid #343434;
            padding: 15px;
            margin-top: 14px;
        }

        .doc-file {
            color: #f5f3ee;
            font-size: 13px;
            font-weight: 600;
            line-height: 1.4;
            word-break: break-word;
        }

        .doc-caption {
            color: #858078;
            font-size: 10px;
            margin-top: 5px;
        }

        .side-stat {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #2c2c2c;
            font-size: 11px;
        }

        .side-stat-name {
            color: #888278;
        }

        .side-stat-value {
            color: #ebe7df;
            font-weight: 700;
        }

        .pipeline-line {
            color: #aaa59d;
            font-size: 11px;
            line-height: 1.9;
        }

        .pipeline-line strong {
            color: #e8e3da;
            font-weight: 600;
        }

        .side-footer {
            border-top: 1px solid #2c2c2c;
            margin-top: 42px;
            padding-top: 12px;
            color: #6f6a63;
            font-size: 9px;
            letter-spacing: 0.08em;
        }

        /* ---------- MAIN ---------- */

        .main-shell {
            max-width: 1120px;
            margin: 0 auto;
            padding: 48px 46px 30px 46px;
        }

        .top-line {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 34px;
        }

        .eyebrow {
            color: #8a847b;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 10px;
            font-weight: 700;
        }

        .ready-badge {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            border: 1px solid #d8d4cc;
            background: #faf9f6;
            padding: 7px 11px;
            color: #5e5a54;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .ready-dot {
            width: 7px;
            height: 7px;
            background: #5c8d68;
            border-radius: 50%;
        }

        .hero-title {
            font-size: 44px;
            line-height: 1.02;
            letter-spacing: -0.045em;
            font-weight: 700;
            color: #161616;
            margin: 0;
        }

        .hero-copy {
            max-width: 650px;
            color: #77716a;
            font-size: 15px;
            line-height: 1.65;
            margin-top: 14px;
        }

        /* ---------- EMPTY STATE ---------- */

        .empty-card {
            margin-top: 38px;
            border: 1px solid #ddd9d1;
            background: #fbfaf7;
            padding: 68px 30px;
            text-align: center;
        }

        .empty-kicker {
            color: #918b82;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-weight: 700;
        }

        .empty-title {
            margin-top: 12px;
            color: #242321;
            font-size: 22px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .empty-copy {
            max-width: 530px;
            margin: 9px auto 0 auto;
            color: #827c74;
            font-size: 13px;
            line-height: 1.6;
        }

        .empty-flow {
            margin-top: 30px;
            color: #918b82;
            font-size: 11px;
            letter-spacing: 0.04em;
        }

        /* ---------- ACTIVE DOCUMENT ---------- */

        .active-header {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            align-items: flex-end;
            margin-bottom: 27px;
        }

        .active-file {
            margin-top: 9px;
            font-size: 25px;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: #1c1b19;
            word-break: break-word;
        }

        .active-meta {
            margin-top: 5px;
            font-size: 12px;
            color: #817b73;
        }

        .active-status {
            border: 1px solid #d4d0c8;
            padding: 7px 11px;
            background: #fbfaf7;
            color: #5d5953;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            white-space: nowrap;
        }

        /* ---------- SUGGESTIONS ---------- */

        .suggest-label {
            color: #8c867d;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            margin-bottom: 10px;
        }

        .stButton > button {
            width: 100%;
            min-height: 42px;
            border-radius: 0;
            border: 1px solid #d6d1c8;
            background: #fbfaf7;
            color: #48443f;
            font-size: 12px;
            text-align: left;
            transition: all 0.15s ease;
        }

        .stButton > button:hover {
            border-color: #a9a39a;
            background: #f0ede7;
            color: #1e1d1b;
        }

        /* ---------- CHAT ---------- */

        [data-testid="stChatMessage"] {
            background: transparent;
            padding: 8px 0;
        }

        [data-testid="stChatMessageContent"] {
            max-width: 820px;
        }

        [data-testid="stChatMessage"] p {
            font-size: 14px;
            line-height: 1.7;
        }

        /* ---------- SOURCES ---------- */

        .source-box {
            border: 1px solid #ded9d1;
            background: #fbfaf7;
            padding: 13px 15px;
            margin-top: 8px;
        }

        .source-meta {
            color: #8d877f;
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            margin-bottom: 7px;
        }

        .source-text {
            color: #59544e;
            font-size: 11px;
            line-height: 1.6;
        }

        /* ---------- INPUT ---------- */

        [data-testid="stChatInput"] {
            border-top: 1px solid #d8d3cb;
            padding-top: 16px;
        }

        /* ---------- MISC ---------- */

        .rule {
            border-top: 1px solid #ddd8d0;
            margin: 34px 0 22px 0;
        }

        .footer-note {
            color: #99938b;
            font-size: 9px;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        div[data-testid="stAlert"] {
            border-radius: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag_engine() -> RAGEngine:
    return RAGEngine()


def initialize_session_state() -> None:
    if "rag" not in st.session_state:
        st.session_state.rag = load_rag_engine()

    if "document_name" not in st.session_state:
        st.session_state.document_name = None

    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_sources(sources) -> None:
    if not sources:
        return

    unique_pages = sorted(
        {chunk.page for chunk, _ in sources}
    )

    with st.expander(
        f"Sources  |  {len(sources)} retrieved chunks"
    ):
        st.markdown(
            "**Source pages:** "
            + ", ".join(
                f"Page {page}" for page in unique_pages
            )
        )

        for i, (chunk, score) in enumerate(
            sources,
            start=1,
        ):
            safe_text = html.escape(chunk.text)

            st.markdown(
                f"""
                <div class="source-box">
                    <div class="source-meta">
                        Retrieved chunk {i} | Page {chunk.page} |
                        Similarity {score:.4f}
                    </div>
                    <div class="source-text">
                        {safe_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def process_upload(uploaded_file) -> None:
    if uploaded_file is None:
        return

    if uploaded_file.name == st.session_state.document_name:
        return

    with st.spinner("Indexing document..."):
        try:
            chunk_count = (
                st.session_state.rag.build_index(
                    uploaded_file.getvalue()
                )
            )

            st.session_state.document_name = uploaded_file.name
            st.session_state.chunk_count = chunk_count
            st.session_state.messages = []

        except ValueError as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(
                f"Document processing failed: {exc}"
            )


def main() -> None:
    initialize_session_state()

    # ---------------- SIDEBAR ----------------

    with st.sidebar:

        st.markdown(
            "<div class='brand-row'>"
            "<div class='brand-mark'>AD</div>"
            "<div>"
            "<div class='brand-name'>AI DOCUMENT<br>ASSISTANT</div>"
            "<div class='brand-note'>Grounded document intelligence</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
        )

        process_upload(uploaded_file)

        if st.session_state.document_name:

            st.markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-file">
                        {html.escape(st.session_state.document_name)}
                    </div>
                    <div class="doc-caption">
                        Active document
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="side-label">Index</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="side-stat">
                    <span class="side-stat-name">Indexed chunks</span>
                    <span class="side-stat-value">{st.session_state.chunk_count}</span>
                </div>
                <div class="side-stat">
                    <span class="side-stat-name">Status</span>
                    <span class="side-stat-value">Ready</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="side-label">Pipeline</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="pipeline-line">
                    <strong>01</strong> Extract<br>
                    <strong>02</strong> Embed<br>
                    <strong>03</strong> Retrieve<br>
                    <strong>04</strong> Generate
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="side-footer">
                FAISS / SENTENCE TRANSFORMERS / GPT-OSS 20B
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- MAIN ----------------

    st.markdown('<div class="main-shell">', unsafe_allow_html=True)

    if not st.session_state.document_name:

        st.markdown(
            """
            <div class="top-line">
                <div class="eyebrow">Document workspace</div>
                <div class="ready-badge">
                    <span class="ready-dot"></span>
                    System ready
                </div>
            </div>

            <div class="hero-title">
                Ask your document.
            </div>

            <div class="hero-copy">
                Upload a document and explore it through grounded
                question answering. Responses are generated from
                retrieved document context rather than outside knowledge.
            </div>

            <div class="empty-card">
                <div class="empty-kicker">Start here</div>
                <div class="empty-title">
                    Bring one document into the workspace.
                </div>
                <div class="empty-copy">
                    Use the upload panel on the left. The document will be
                    extracted, embedded, indexed and made searchable automatically.
                </div>
                <div class="empty-flow">
                    EXTRACT  /  EMBED  /  RETRIEVE  /  GENERATE
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="rule"></div>
            <div class="footer-note">
                Private document workspace  grounded retrieval
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(
        """
        <div class="top-line">
            <div class="eyebrow">Active document</div>
            <div class="ready-badge">
                <span class="ready-dot"></span>
                Indexed and ready
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="active-header">
            <div>
                <div class="active-file">
                    {html.escape(st.session_state.document_name)}
                </div>
                <div class="active-meta">
                    {st.session_state.chunk_count} searchable chunks
                </div>
            </div>
            <div class="active-status">
                RAG workspace
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":
                display_sources(
                    message.get("sources", [])
                )

    question = st.chat_input(
        "Ask anything about this document..."
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

            with st.spinner(
                "Retrieving context and generating answer..."
            ):
                try:
                    answer, sources = (
                        st.session_state.rag.generate_answer(
                            question
                        )
                    )

                    st.markdown(answer)

                    display_sources(sources)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as exc:

                    error_message = (
                        "Something went wrong while processing "
                        f"your question: {exc}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )

    st.markdown(
        """
        <div class="rule"></div>
        <div class="footer-note">
            AI DOCUMENT ASSISTANT  GROUNDED RAG WORKSPACE
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
