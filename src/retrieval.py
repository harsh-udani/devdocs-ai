
import faiss
import numpy as np

from src.ingestion import extract_documentation
from src.chunking import chunk_documentation
from src.embeddings import create_embedding_model, embed_texts


def build_search_index(chunks, model):
    """Build a FAISS index from documentation chunks."""

    if not chunks:
        raise ValueError("No chunks available to index.")

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts, model)

    # FAISS expects a contiguous float32 matrix.
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    # Vectors are normalized, so inner product = cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, vectors



def search_documentation(
    query,
    chunks,
    model,
    index,
    top_k=3,
    min_similarity=0.30,
):
    """Retrieve the most relevant chunks above a configurable threshold."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks or index.ntotal == 0:
        return []

    query_vector = embed_texts([query], model)
    query_vector = np.ascontiguousarray(
        query_vector, dtype=np.float32
    )

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        similarity = float(score)

        # Reject results below the configured relevance threshold.
        if similarity < min_similarity:
            continue

        result = chunks[int(idx)].copy()
        result["similarity"] = similarity
        results.append(result)

    return results



if __name__ == "__main__":
    page = extract_documentation(
        "https://docs.python.org/3/tutorial/"
    )
    chunks = chunk_documentation(page)

    model = create_embedding_model()
    index, _ = build_search_index(chunks, model)

    query = "How do I define and call a function in Python?"

    results = search_documentation(
        query=query,
        chunks=chunks,
        model=model,
        index=index,
        top_k=3,
    )

    print("\nQUERY:", query)

    for rank, result in enumerate(results, start=1):
        print(f"\n--- RESULT {rank} ---")
        print("Similarity:", round(result["similarity"], 4))
        print("Title:", result["title"])
        print("Source:", result["source_url"])
        print("Text:", result["text"][:500])

