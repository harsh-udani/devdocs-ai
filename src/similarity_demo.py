
import numpy as np
from embeddings import create_embedding_model, embed_texts


texts = [
    "How do I authenticate a user with JWT?",
    "Configure JSON Web Token authentication for your API.",
    "Python lists support indexing and slicing.",
]

model = create_embedding_model()
vectors = embed_texts(texts, model)

query_vector = vectors[0]

# Compare the first text with every vector.
scores = vectors @ query_vector

for text, score in zip(texts, scores):
    print(f"{score:.4f} | {text}")
