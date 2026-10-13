
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embedding_model():
    """Load the sentence embedding model."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], model):
    """Convert a list of texts into numerical vectors."""

    if not texts:
        raise ValueError("At least one text is required.")

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return vectors


if __name__ == "__main__":
    model = create_embedding_model()

    texts = [
        "How do I authenticate a user with JWT?",
        "Configure JSON Web Token authentication for your API.",
        "Python lists support indexing and slicing.",
    ]

    vectors = embed_texts(texts, model)

    print("Number of texts:", len(texts))
    print("Embedding matrix shape:", vectors.shape)
    print("First embedding dimensions:", vectors[0][:10])
