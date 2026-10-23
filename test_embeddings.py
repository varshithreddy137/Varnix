
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The internship requires Python programming skills.",
    "The company provides summer internship opportunities.",
    "The document discusses an unrelated topic about cooking.",
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
)

question = "What programming skills are required?"
question_embedding = model.encode(
    [question],
    normalize_embeddings=True,
)

scores = embeddings @ question_embedding[0]

for sentence, score in sorted(
    zip(sentences, scores),
    key=lambda item: item[1],
    reverse=True,
):
    print(f"{score:.3f} | {sentence}")

print("Embedding test completed.")
