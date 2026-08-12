import os
import sys
import uuid
import time
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory (project root) to sys.path so the script can be executed directly by path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify

from app import config
from app.model import EttinONNXReranker

logger = logging.getLogger("ettin-reranker-server")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
reranker_model: Optional[EttinONNXReranker] = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
@app.route("/ready", methods=["GET"])
def health():
    if reranker_model is None:
        return jsonify({
            "status": "unhealthy",
            "error": "Model not initialized"
        }), 503

    return jsonify({
        "status": "ok",
        "model_dir": config.MODEL_DIR,
        "model_name": config.MODEL_NAME,
        "engine": "onnxruntime",
        "gpu_available": config.USE_GPU,
    }), 200


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": config.MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "cross-encoder",
            }
        ],
    })


def extract_pairs(payload: dict) -> List[tuple[str, str]]:
    pairs = []
    query = payload.get("query")
    documents = payload.get("documents")
    inp = payload.get("input")

    # Format 1: query + documents explicitly passed
    if query is not None and documents is not None:
        for doc in documents:
            doc_str = doc.get("text", str(doc)) if isinstance(doc, dict) else str(doc)
            pairs.append((query, doc_str))
        return pairs

    # Format 2: query + input passed as list of documents
    if query is not None and inp is not None:
        docs = inp if isinstance(inp, list) else [inp]
        for doc in docs:
            pairs.append((query, str(doc)))
        return pairs

    # Format 3: input passed as strings, pairs, or dicts
    if inp is not None:
        if isinstance(inp, str):
            parts = inp.split("\n", 1)
            q = parts[0]
            d = parts[1] if len(parts) > 1 else parts[0]
            pairs.append((q, d))
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pairs.append((str(item[0]), str(item[1])))
                elif isinstance(item, dict) and "query" in item and "document" in item:
                    pairs.append((str(item["query"]), str(item["document"])))
                elif isinstance(item, str):
                    parts = item.split("\n", 1)
                    q = parts[0]
                    d = parts[1] if len(parts) > 1 else parts[0]
                    pairs.append((q, d))
                else:
                    pairs.append(("", str(item)))

    return pairs


@app.route("/v1/embeddings", methods=["POST", "OPTIONS"])
def create_embeddings():
    if request.method == "OPTIONS":
        return "", 200

    if reranker_model is None:
        return jsonify({"error": {"message": "Model server not initialized", "type": "server_error"}}), 503

    payload = request.get_json(force=True, silent=True) or {}
    pairs = extract_pairs(payload)

    if not pairs:
        return jsonify({
            "error": {
                "message": "Unable to extract valid (query, document) pairs from request payload.",
                "type": "invalid_request_error",
            }
        }), 400

    scores, total_tokens = reranker_model.predict(pairs, batch_size=config.BATCH_SIZE)

    data_items = [
        {
            "object": "embedding",
            "embedding": [score],
            "score": score,
            "index": idx,
        }
        for idx, score in enumerate(scores)
    ]

    model_id = payload.get("model") or config.MODEL_NAME

    return jsonify({
        "object": "list",
        "data": data_items,
        "model": model_id,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    })


@app.route("/v1/rerank", methods=["POST", "OPTIONS"])
@app.route("/rerank", methods=["POST", "OPTIONS"])
def rerank():
    if request.method == "OPTIONS":
        return "", 200

    if reranker_model is None:
        return jsonify({"error": {"message": "Model server not initialized", "type": "server_error"}}), 503

    payload = request.get_json(force=True, silent=True) or {}
    query = payload.get("query")
    raw_docs = payload.get("documents", [])
    top_n = payload.get("top_n")
    return_documents = payload.get("return_documents", True)
    rank_fields = payload.get("rank_fields")

    if not query or not isinstance(raw_docs, list):
        return jsonify({
            "error": {
                "message": "'query' string and 'documents' list are required fields.",
                "type": "invalid_request_error",
            }
        }), 400

    doc_pairs = []
    for doc in raw_docs:
        if isinstance(doc, dict):
            text_val = ""
            if rank_fields:
                text_val = " ".join([str(doc.get(f, "")) for f in rank_fields if f in doc])
            if not text_val:
                text_val = doc.get("text", doc.get("content", str(doc)))
            doc_pairs.append((query, text_val))
        else:
            doc_pairs.append((query, str(doc)))

    scores, total_tokens = reranker_model.predict(doc_pairs, batch_size=config.BATCH_SIZE)

    results = []
    for idx, (score, doc_orig) in enumerate(zip(scores, raw_docs)):
        results.append({
            "index": idx,
            "relevance_score": score,
            "document": doc_orig if return_documents else None,
        })

    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    if top_n is not None and top_n > 0:
        results = results[:top_n]

    model_id = payload.get("model") or config.MODEL_NAME

    return jsonify({
        "id": f"rerank-{uuid.uuid4().hex[:12]}",
        "model": model_id,
        "results": results,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    })


def main():
    parser = argparse.ArgumentParser(description="Ettin 150M ONNX Reranker API Server (Flask)")
    parser.add_argument("--model-dir", type=str, default=config.MODEL_DIR, help="Path to local directory with model files")
    parser.add_argument("--onnx-path", type=str, default=config.ONNX_PATH, help="Direct path to model.onnx file")
    parser.add_argument("--host", type=str, default=config.HOST, help="Host address to bind to")
    parser.add_argument("--port", type=int, default=config.PORT, help="Port to bind to")
    parser.add_argument("--max-length", type=int, default=config.MAX_LENGTH, help="Maximum token sequence length")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for inference")
    parser.add_argument("--use-gpu", action="store_true", default=config.USE_GPU, help="Use CUDA GPU if available")
    parser.add_argument("--normalize-scores", action="store_true", default=config.NORMALIZE_SCORES, help="Apply sigmoid score normalization")
    parser.add_argument("--no-normalize-scores", action="store_false", dest="normalize_scores", help="Disable score normalization")

    args = parser.parse_args()

    # Update global config
    config.MODEL_DIR = args.model_dir
    config.ONNX_PATH = args.onnx_path
    config.HOST = args.host
    config.PORT = args.port
    config.MAX_LENGTH = args.max_length
    config.BATCH_SIZE = args.batch_size
    config.USE_GPU = args.use_gpu
    config.NORMALIZE_SCORES = args.normalize_scores

    global reranker_model
    logger.info(f"Initializing model from local directory: {config.MODEL_DIR}")
    reranker_model = EttinONNXReranker(
        model_dir=config.MODEL_DIR,
        onnx_path=config.ONNX_PATH,
        max_length=config.MAX_LENGTH,
        use_gpu=config.USE_GPU,
        normalize_scores=config.NORMALIZE_SCORES,
    )

    logger.info(f"Starting Flask server on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, threaded=True)


if __name__ == "__main__":
    main()
