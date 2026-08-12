"""
Validation script to check ONNX EmbeddingGemma server outputs against OpenAI API standards.
Tests single string embeddings, batch embeddings, normalization (unit vector length), and cosine similarity semantics.
"""
import sys
import math
import requests
import argparse


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def test_health_endpoint(server_url: str):
    print("\n--- Test 0: Health / Ready Endpoint ---")
    url = f"{server_url.rstrip('/')}/health"
    resp = requests.get(url, timeout=5)
    assert resp.status_code == 200, f"Health check failed with HTTP {resp.status_code}: {resp.text}"
    info = resp.json()
    print(f"✓ Server Status: {info.get('status')}")
    print(f"✓ Model Type: {info.get('model_type')}")
    print(f"✓ Embedding Loaded: {info.get('embedding_loaded')}")


def test_single_embedding(server_url: str):
    print("\n--- Test 1: Single Input Embedding ---")
    url = f"{server_url.rstrip('/')}/v1/embeddings"
    payload = {
        "input": "EmbeddingGemma produces high quality vector embeddings for text retrieval.",
        "model": "google/embeddinggemma-300m",
    }
    resp = requests.post(url, json=payload, timeout=10)
    assert resp.status_code == 200, f"HTTP error {resp.status_code}: {resp.text}"
    data = resp.json()

    assert data.get("object") == "list", "Expected object='list'"
    assert len(data.get("data", [])) == 1, "Expected 1 embedding item"

    item = data["data"][0]
    embedding = item.get("embedding")
    assert isinstance(embedding, list), "Embedding field must be a list"

    dim = len(embedding)
    assert dim > 1, f"Embedding vector should be multi-dimensional (got {dim})"

    norm = math.sqrt(sum(x * x for x in embedding))
    print(f"✓ Received embedding vector of dimension: {dim}")
    print(f"✓ L2 norm of embedding vector: {norm:.4f} (expected ~1.0 for normalized embeddings)")
    assert abs(norm - 1.0) < 0.05, f"Vector norm {norm} deviates significantly from unit length"
    print("✓ Single input embedding test PASSED.")


def test_batch_embeddings_and_similarity(server_url: str):
    print("\n--- Test 2: Batch Input & Semantic Cosine Similarity ---")
    url = f"{server_url.rstrip('/')}/v1/embeddings"

    doc_anchor = "Artificial intelligence and machine learning are revolutionizing software."
    doc_similar = "Deep learning and AI models are changing how computer programs are built."
    doc_dissimilar = "The recipe requires three ripe tomatoes, olive oil, and fresh garlic."

    payload = {
        "input": [doc_anchor, doc_similar, doc_dissimilar],
        "model": "google/embeddinggemma-300m",
    }

    resp = requests.post(url, json=payload, timeout=10)
    assert resp.status_code == 200, f"HTTP error {resp.status_code}: {resp.text}"
    data = resp.json()

    data_list = data.get("data", [])
    assert len(data_list) == 3, f"Expected 3 embeddings, got {len(data_list)}"

    e_anchor = data_list[0]["embedding"]
    e_similar = data_list[1]["embedding"]
    e_dissimilar = data_list[2]["embedding"]

    sim_related = cosine_similarity(e_anchor, e_similar)
    sim_unrelated = cosine_similarity(e_anchor, e_dissimilar)

    print(f"  Anchor vs Similar Doc Cosine Similarity:    {sim_related:.4f}")
    print(f"  Anchor vs Unrelated Doc Cosine Similarity:  {sim_unrelated:.4f}")

    assert sim_related > sim_unrelated, (
        f"Semantic test failed: Related similarity ({sim_related:.4f}) "
        f"should be higher than unrelated similarity ({sim_unrelated:.4f})"
    )
    print("✓ Semantic similarity test verified (related documents score higher).")
    print("✓ Batch embeddings test PASSED.")


def main():
    parser = argparse.ArgumentParser(description="Verification script for ONNX Embedding Server")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="Server base URL")
    args = parser.parse_args()

    print(f"Testing ONNX Embedding Server at: {args.server_url}")
    try:
        test_health_endpoint(args.server_url)
        test_single_embedding(args.server_url)
        test_batch_embeddings_and_similarity(args.server_url)
        print("\n==================================================")
        print(" ALL EMBEDDING VERIFICATION TESTS PASSED!")
        print("==================================================")
    except AssertionError as ae:
        print(f"\n❌ VERIFICATION TEST FAILED: {ae}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR OCCURRED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
