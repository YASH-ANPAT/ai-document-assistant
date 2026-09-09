from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO
from typing import List

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


load_dotenv()


@dataclass
class DocumentChunk:
    """A piece of document text with its source page."""
    
    text: str
    page: int


class RAGEngine:
    """
    Complete Retrieval-Augmented Generation pipeline.

    Pipeline:
        PDF → text extraction → chunking → embeddings → FAISS
        → similarity retrieval → grounded LLM response
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 4,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedding_model = SentenceTransformer(
            embedding_model_name
        )

        self.index: faiss.Index | None = None
        self.chunks: List[DocumentChunk] = []

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. "
                "Add it to your .env file."
            )

        self.llm = Groq(api_key=api_key)

        self.model_name = "openai/gpt-oss-20b"

    # ------------------------------------------------------------------
    # PDF PROCESSING
    # ------------------------------------------------------------------

    def extract_text_from_pdf(
        self,
        pdf_bytes: bytes,
    ) -> List[tuple[str, int]]:
        """Extract text page-by-page from a PDF."""

        try:
            reader = PdfReader(BytesIO(pdf_bytes))
        except Exception as exc:
            raise ValueError(
                "The uploaded file could not be read as a valid PDF."
            ) from exc

        pages: List[tuple[str, int]] = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text() or ""

            if text.strip():
                pages.append(
                    (
                        text.strip(),
                        page_number,
                    )
                )

        return pages

    # ------------------------------------------------------------------
    # CHUNKING
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> List[str]:
        """Create overlapping character-based chunks."""

        text = " ".join(text.split())

        if not text:
            return []

        chunks: List[str] = []
        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = max(
                end - self.chunk_overlap,
                start + 1,
            )

        return chunks

    # ------------------------------------------------------------------
    # INDEX BUILDING
    # ------------------------------------------------------------------

    def build_index(
        self,
        pdf_bytes: bytes,
    ) -> int:
        """
        Extract, chunk, embed, and index the PDF.
        
        Returns:
            Number of indexed chunks.
        """

        pages = self.extract_text_from_pdf(
            pdf_bytes
        )

        if not pages:
            raise ValueError(
                "No extractable text was found in this PDF. "
                "Scanned/image-only PDFs are not supported yet."
            )

        self.chunks = []

        for page_text, page_number in pages:

            page_chunks = self.split_text(
                page_text
            )

            for chunk in page_chunks:

                self.chunks.append(
                    DocumentChunk(
                        text=chunk,
                        page=page_number,
                    )
                )

        if not self.chunks:
            raise ValueError(
                "The document did not produce usable text chunks."
            )

        texts = [
            chunk.text
            for chunk in self.chunks
        ]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        return len(self.chunks)

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
    ) -> List[tuple[DocumentChunk, float]]:
        """Retrieve the most relevant document chunks."""

        if self.index is None:
            raise RuntimeError(
                "No document has been indexed yet."
            )

        query = query.strip()

        if not query:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        k = min(
            self.top_k,
            len(self.chunks),
        )

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results: List[
            tuple[DocumentChunk, float]
        ] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            results.append(
                (
                    self.chunks[index],
                    float(score),
                )
            )

        return results

    # ------------------------------------------------------------------
    # PROMPT BUILDING
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        question: str,
        retrieved_chunks: List[
            tuple[DocumentChunk, float]
        ],
    ) -> str:
        """Build a grounded RAG prompt."""

        context_parts: List[str] = []

        for i, (chunk, score) in enumerate(
            retrieved_chunks,
            start=1,
        ):
            context_parts.append(
                f"[Source {i} | Page {chunk.page}]\n"
                f"{chunk.text}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

Rules:
1. Use ONLY the provided document context.
2. Do not invent, infer, or assume facts that are not supported
   by the document.
3. If the answer is not present in the context, clearly say:
   "The information is not available in the uploaded document."
4. Always provide a complete, non-empty answer.
5. For questions asking for multiple items, provide all relevant
   items found in the context.
6. Keep the response concise but complete.
7. Mention page numbers when useful..

DOCUMENT CONTEXT:
----------------
{context}
----------------

USER QUESTION:
{question}

ANSWER:
""".strip()

        return prompt

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
    ) -> tuple[str, List[tuple[DocumentChunk, float]]]:
        """
        Retrieve relevant context and generate a grounded answer.

        Returns:
            answer, retrieved sources
        """

        retrieved_chunks = self.retrieve(
            question
        )

        if not retrieved_chunks:
            return (
                "I couldn't find relevant information "
                "in the uploaded document.",
                [],
            )

        prompt = self.build_prompt(
            question,
            retrieved_chunks,
        )

        try:
            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise document "
                            "question-answering assistant."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=800,
            )

        except Exception as exc:
            raise RuntimeError(
                f"LLM generation failed: {exc}"
            ) from exc

        answer = (
            response.choices[0]
            .message
            .content
            .strip()
        )

        return (
            answer,
            retrieved_chunks,
        )