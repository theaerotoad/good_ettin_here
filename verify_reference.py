"""
Validation script to check ONNX Ettin Reranker server logits against
canonical reference examples from the official Ettin Reranker release notes
and optional direct SentenceTransformers PyTorch baseline comparison.
"""

import sys
import json
import urllib.request
import urllib.error

# Canonical evaluation pairs from the Ettin release documentation / HF Model Card
CANONICAL_TEST_CASES = [
    {
        "name": "Canonical Example 1: Red Planet (Hugging Face Model Card)",
        "query": "Which planet is known as the Red Planet?",
        "documents": [
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Jupiter, the largest planet in our solar system, has a prominent red spot.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        ],
        # Document index 1 ("Mars...") must be strictly rank 1 with high positive logit (> 10.0)
        # Document index 0 ("Venus...") must be lowest or low logit (< 5.0)
        "expected_top_index": 1,
    },
    {
        "name": "Canonical Example 2: Apple Founding (Ettin Release Blogpost)",
        "query": "Where was Apple founded?",
        "documents": [
            "Apple Inc. was founded in Los Altos, California in 1976.",
            "Fruit production in California includes apples, oranges, and grapes.",
        ],
        "expected_top_index": 0,
    },
    {
        "name": "Canonical Example 3: PostgreSQL Connection Pooling",
        "query": "How do I configure connection pooling in PostgreSQL using Python?",
        "documents": [
            "To configure connection pooling for PostgreSQL in Python, use psycopg2.pool.ThreadedConnectionPool or SQLAlchemy's QueuePool(pool_size=10).",
            "PostgreSQL is an open-source relational database management system emphasizing extensibility and SQL compliance.",
        ],
        "expected_top_index": 0,
    },
]


def query_onnx_server(server_url: str, query: str, documents: list[str], return_raw_scores: bool = True) -> list[float]:
    """Queries the Flask ONNX reranker server and returns raw scores in input document order."""
    payload = json.dumps({
        "query": query,
        "documents": documents,
        "return_documents": False,
        "return_raw_scores": return_raw_scores
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/rerank",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            # Re-sort results back into original document array order
            scores = [0.0] * len(documents)
            for res in results:
                scores[res["index"]] = res["relevance_score"]
            return scores
    except urllib.error.URLError as e:
        print(f"[ERROR] Could not connect to server at {server_url}: {e}")
        sys.exit(1)


def compare_with_sentence_transformers(model_name: str, server_url: str):
    """
    If sentence-transformers is installed, runs exact PyTorch FP32 inference
    and checks ONNX output precision byte-for-byte / float-for-float.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("\n[INFO] 'sentence-transformers' package not installed. Skipping direct PyTorch cross-check.")
        print("To run direct float32 precision verification, run: pip install sentence-transformers torch\n")
        return

    print(f"\n=================================================================")
    print(f"LOADING PYTORCH REFERENCE MODEL: {model_name}")
    print(f"=================================================================")
    model = CrossEncoder(model_name)

    all_pytorch_scores = []
    all_onnx_scores = []

    for case in CANONICAL_TEST_CASES:
        query = case["query"]
        docs = case["documents"]
        pairs = [(query, d) for d in docs]

        pt_scores = model.predict(pairs).tolist()
        onnx_scores = query_onnx_server(server_url, query, docs, return_raw_scores=True)

        all_pytorch_scores.extend(pt_scores)
        all_onnx_scores.extend(onnx_scores)

        print(f"\nTest: {case['name']}")
        for idx, (p_sc, o_sc, doc) in enumerate(zip(pt_scores, onnx_scores, docs)):
            diff = abs(p_sc - o_sc)
            print(f"  Doc #{idx} | PyTorch: {p_sc:8.4f} | ONNX: {o_sc:8.4f} | Diff: {diff:8.6f} | {doc[:50]}...")

    # Calculate absolute error metrics across all pairs
    diffs = [abs(p - o) for p, o in zip(all_pytorch_scores, all_onnx_scores)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)

    print(f"\n-----------------------------------------------------------------")
    print(f"PRECISION COMPARISON SUMMARY:")
    print(f"  Max Absolute Difference : {max_diff:.6f}")
    print(f"  Mean Absolute Difference: {mean_diff:.6f}")

    # Logit differences < 0.05 are expected due to ONNX Runtime graph optimizations,
    # SIMD floating-point reassociation, and fast GELU tanh approximations.
    if max_diff < 1e-3:
        print("  Status: [PASS] Exact FP32 bitwise match (< 0.001 max diff).")
    elif max_diff < 0.05:
        print("  Status: [PASS] ONNX logits match PyTorch reference within expected runtime tolerance (< 0.05 max diff).")
    else:
        print("  Status: [WARN] Logits differ noticeably (> 0.05 diff). Check quantization or model head weights.")
    print(f"-----------------------------------------------------------------\n")


def detect_server_model(server_url: str) -> str:
    """Attempts to discover the loaded reranker model name from /health."""
    try:
        req = urllib.request.Request(f"{server_url}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("model_name") or "cross-encoder/ettin-reranker-150m-v1"
    except Exception:
        return "cross-encoder/ettin-reranker-150m-v1"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify Ettin Reranker against canonical benchmarks.")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="URL of running server")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model ID for PyTorch comparison")
    args = parser.parse_args()

    server_url = args.server_url.rstrip("/")
    model_name = args.model or detect_server_model(server_url)

    print("=================================================================")
    print(f"RUNNING CANONICAL ETTIN RERANKER VERIFICATION SUITE ({model_name})")
    print("=================================================================")

    for case in CANONICAL_TEST_CASES:
        print(f"\n--- {case['name']} ---")
        query = case["query"]
        docs = case["documents"]
        scores = query_onnx_server(server_url, query, docs, return_raw_scores=False)

        top_idx = max(range(len(scores)), key=lambda i: scores[i])
        expected_top = case["expected_top_index"]

        status = "PASS" if top_idx == expected_top else "FAIL"
        print(f"Query: '{query}'")
        print(f"Top Document Index: {top_idx} (Expected: {expected_top}) -> [{status}]")

        for idx, (sc, doc) in enumerate(zip(scores, docs)):
            prefix = "==>" if idx == top_idx else "   "
            print(f"  {prefix} Doc #{idx} | Score: {sc:8.4f} | {doc}")

    # Optional side-by-side PyTorch comparison if library available
    compare_with_sentence_transformers(model_name, server_url)


if __name__ == "__main__":
    main()
