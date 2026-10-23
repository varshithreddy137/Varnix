
import numpy as np
import streamlit as st



# Loaded once and reused across Streamlit reruns.
@__import__("streamlit").cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")




def chunk_document(
    text: str,
    max_words: int = 180,
    overlap: int = 35,
) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks




def retrieve_relevant_chunks(
    question: str,
    chunks: list[dict],
    top_k: int = 4,
    min_score: float = 0.15,
) -> list[dict]:
    """Retrieve semantically relevant passages above a minimum score."""
    if not question.strip() or not chunks:
        return []

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    document_embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = np.dot(
        document_embeddings,
        question_embedding[0],
    )

    
    relevant_indices = [
        index
        for index, score in enumerate(scores)
        if float(score) >= min_score
    ]

    if not relevant_indices:
        return []

    
    ranked_indices = sorted(
        relevant_indices,
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            **chunks[int(index)],
            "score": float(scores[index]),
        })

    return results

