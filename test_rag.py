from pathlib import Path

from rag_pipeline import RAGEngine


PDF_PATH = Path("test.pdf")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            "Put a PDF named 'test.pdf' in the project folder."
        )

    pdf_bytes = PDF_PATH.read_bytes()

    print("Loading RAG engine...")

    rag = RAGEngine()

    print("Building FAISS index...")

    chunk_count = rag.build_index(pdf_bytes)

    print(f"Indexed chunks: {chunk_count}")

    while True:
        question = input(
            "\nAsk a question (type 'exit' to stop): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        print("\nGenerating answer...\n")

        answer, sources = rag.generate_answer(question)

        print("ANSWER")
        print("------")
        print(answer)

        print("\nSOURCES")
        print("-------")

        for chunk, score in sources:
            print(
                f"Page {chunk.page} "
                f"(similarity: {score:.4f})"
            )


if __name__ == "__main__":
    main()