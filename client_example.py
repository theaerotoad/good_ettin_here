"""
Comprehensive client test suite for Ettin ONNX Reranker server.
Tests multi-domain queries, technical docs, and distractor handling.
Uses standard Python library (urllib) - zero external client dependencies!
"""

import json
import urllib.request

BASE_URL = "http://localhost:8000"


def send_rerank_request(query: str, documents: list[str], top_n: int = None):
    payload = {
        "model": "cross-encoder/ettin-reranker-150m-v1",
        "query": query,
        "documents": documents,
        "top_n": top_n or len(documents),
        "return_documents": True,
    }

    req = urllib.request.Request(
        f"{BASE_URL}/v1/rerank",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_suite_1_astronomy():
    print("\n" + "=" * 65)
    print("TEST SUITE 1: Astronomy & Planetary Science Trivia")
    print("=" * 65)

    query = "Which planet is known as the Red Planet?"
    docs = [
        "Venus is often called Earth's twin because of its similar size.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    ]

    res = send_rerank_request(query, docs)
    print(f"Query: '{query}'\n")
    for rank, item in enumerate(res["results"], start=1):
        score = item["relevance_score"]
        doc = item["document"]
        orig_idx = item["index"]
        print(f"Rank {rank} (Doc #{orig_idx}) | Score: {score:.4f} | {doc}")


def test_suite_2_technical_docs():
    print("\n" + "=" * 65)
    print("TEST SUITE 2: Technical Documentation & Code Query")
    print("=" * 65)

    query = "How do I configure connection pooling in PostgreSQL using Python?"
    docs = [
        "PostgreSQL is an open-source relational database management system emphasizing extensibility and SQL compliance.",
        "To configure connection pooling for PostgreSQL in Python, use psycopg2.pool.ThreadedConnectionPool or SQLAlchemy's QueuePool(pool_size=10).",
        "In Python, you can connect to MySQL using mysql-connector-python or PyMySQL with standard cursor execution.",
        "Connection pooling in Redis can be configured using redis.ConnectionPool(host='localhost', port=6379, db=0).",
        "To fix a 'Too many connections' error in PostgreSQL, adjust max_connections in postgresql.conf on the server.",
    ]

    res = send_rerank_request(query, docs)
    print(f"Query: '{query}'\n")
    for rank, item in enumerate(res["results"], start=1):
        score = item["relevance_score"]
        doc = item["document"]
        orig_idx = item["index"]
        print(f"Rank {rank} (Doc #{orig_idx}) | Score: {score:.4f} | {doc}")


def test_suite_3_distractors_and_fine_grained_nuance():
    print("\n" + "=" * 65)
    print("TEST SUITE 3: Distractors (Austria vs Australia, Sydney vs Canberra)")
    print("=" * 65)

    query = "What is the official capital city of Australia?"
    docs = [
        "Canberra was officially named the capital of Australia in 1913 as a compromise between rival cities Sydney and Melbourne.",
        "Sydney is the most populous city in Australia and the capital city of the state of New South Wales.",
        "Melbourne is the capital of Victoria and served as Australia's temporary seat of government from 1901 to 1927.",
        "Vienna is the capital and largest city of Austria, located in Central Europe along the Danube River.",
    ]

    res = send_rerank_request(query, docs)
    print(f"Query: '{query}'\n")
    for rank, item in enumerate(res["results"], start=1):
        score = item["relevance_score"]
        doc = item["document"]
        orig_idx = item["index"]
        print(f"Rank {rank} (Doc #{orig_idx}) | Score: {score:.4f} | {doc}")


if __name__ == "__main__":
    test_suite_1_astronomy()
    test_suite_2_technical_docs()
    test_suite_3_distractors_and_fine_grained_nuance()
