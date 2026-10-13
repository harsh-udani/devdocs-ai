
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documentation(page: dict) -> list[dict]:
    """Split extracted documentation text into chunks with source metadata."""

    text = page.get("text", "").strip()

    if not text:
        raise ValueError("Documentation text cannot be empty.")

    metadata = {
        "title": page["title"],
        "source_url": page["url"],
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = splitter.create_documents(
        texts=[text],
        metadatas=[metadata],
    )

    chunks = []

    for index, document in enumerate(documents):
        chunks.append({
            "chunk_id": index,
            "text": document.page_content,
            "title": document.metadata["title"],
            "source_url": document.metadata["source_url"],
            "start_index": document.metadata.get("start_index", 0),
        })

    return chunks


if __name__ == "__main__":
    from ingestion import extract_documentation

    page = extract_documentation(
        "https://docs.python.org/3/tutorial/"
    )

    chunks = chunk_documentation(page)

    print("Document title:", page["title"])
    print("Total chunks:", len(chunks))

    for chunk in chunks[:3]:
        print("\n--- CHUNK", chunk["chunk_id"], "---")
        print("Source:", chunk["source_url"])
        print("Start index:", chunk["start_index"])
        print("Text:", chunk["text"][:300])
