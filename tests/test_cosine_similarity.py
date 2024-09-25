from aleph_alpha_client import Client
import os

API_KEY = os.getenv("AA_TOKEN")
client = Client(token=API_KEY)

from typing import Sequence
from aleph_alpha_client import Prompt, SemanticEmbeddingRequest, SemanticRepresentation
import math


# helper function to embed text using the symmetric or asymmetric model
def embed(text: str, representation: SemanticRepresentation):
    request = SemanticEmbeddingRequest(
        prompt=Prompt.from_text(text), representation=representation
    )
    result = client.semantic_embed(request, model="luminous-base")
    return result.embedding


# helper function to calculate the cosine similarity between two vectors
def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


# helper function to print the similarity between the query and text embeddings
def print_result(texts, query, query_embedding, text_embeddings):
    for i, text in enumerate(texts):
        print(
            f"Similarity between '{query}' and '{text[:25]}...': {cosine_similarity(query_embedding, text_embeddings[i])}"
        )


texts = [
    "Agent?team=0_0",
    "Agent?team=0_1",
    "Agent?team=0_2",
]

# query = "Agent 'Agent?team=0_0' go to bottom center"
# query = "Agent 0_0 go to bottom center"
# query = "fire 0 go to bottom center"
query = "Fire 0 go"

symmetric_query = embed(query, SemanticRepresentation.Symmetric)
asymmetric_query = embed(query, SemanticRepresentation.Query)
symmetric_embeddings = [embed(text, SemanticRepresentation.Symmetric) for text in texts]
asymmetric_embeddings = [embed(text, SemanticRepresentation.Document) for text in texts]

print("Symmetric: ")
print_result(texts, query, symmetric_query, symmetric_embeddings)
print("\nAsymmetric: ")
print_result(texts, query, asymmetric_query, asymmetric_embeddings)
