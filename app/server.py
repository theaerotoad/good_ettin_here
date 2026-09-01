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

from flask import Flask, request, jsonify, render_template

from app import config
from app.model import EttinONNXReranker, DocLayNetONNX

logger = logging.getLogger("ettin-reranker-server")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
reranker_model: Optional[EttinONNXReranker] = None
embedding_model: Optional[object] = None
doclaynet_model: Optional[DocLayNetONNX] = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
@app.route("/ready", methods=["GET"])
def health():
    if reranker_model is None and embedding_model is None and doclaynet_model is None:
        return jsonify({
            "status": "unhealthy",
            "error": "No model initialized"
        }), 503

    return jsonify({
        "status": "ok",
        "model_type": config.MODEL_TYPE,
        "reranker_loaded": reranker_model is not None,
        "embedding_loaded": embedding_model is not None,
        "doclaynet_loaded": doclaynet_model is not None,
        "model_dir": config.MODEL_DIR,
        "model_name": config.MODEL_NAME,
        "embedding_model_name": config.EMBEDDING_MODEL_NAME,
        "doclaynet_model_name": config.DOCLAYNET_MODEL_NAME,
        "engine": "onnxruntime",
        "gpu_available": config.USE_GPU,
    }), 200


@app.route("/v1/models", methods=["GET"])
def list_models():
    models = []
    if reranker_model is not None:
        models.append({
            "id": config.MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "cross-encoder",
        })
    if embedding_model is not None:
        models.append({
            "id": config.EMBEDDING_MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google",
        })
    if doclaynet_model is not None:
        models.append({
            "id": config.DOCLAYNET_MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "doclaynet",
        })
    return jsonify({
        "object": "list",
        "data": models,
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

    if embedding_model is None:
        return jsonify({
            "error": {
                "message": "Embedding model is not loaded or failed initialization. Verify embedding model files in startup directory.",
                "type": "server_error",
            }
        }), 503

    payload = request.get_json(force=True, silent=True) or {}
    raw_input = payload.get("input")
    texts = []
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list):
        texts = [str(item) for item in raw_input]
    elif "documents" in payload and isinstance(payload["documents"], list):
        texts = [str(doc) for doc in payload["documents"]]
    elif "query" in payload:
        texts = [str(payload["query"])]

    if not texts:
        return jsonify({
            "error": {
                "message": "Missing 'input' field (string or list of strings) for embeddings endpoint.",
                "type": "invalid_request_error",
            }
        }), 400

    embeddings, total_tokens = embedding_model.embed(texts, batch_size=config.BATCH_SIZE)

    data_items = [
        {
            "object": "embedding",
            "embedding": vec,
            "index": idx,
        }
        for idx, vec in enumerate(embeddings)
    ]

    model_id = payload.get("model") or config.EMBEDDING_MODEL_NAME

    return jsonify({
        "object": "list",
        "data": data_items,
        "model": model_id,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    })


@app.route("/v1/vision/layout", methods=["POST", "OPTIONS"])
@app.route("/v1/layout/detect", methods=["POST", "OPTIONS"])
@app.route("/v1/doclaynet", methods=["POST", "OPTIONS"])
def detect_layout():
    if request.method == "OPTIONS":
        return "", 200

    if doclaynet_model is None:
        return jsonify({"error": {"message": "DocLayNet model server not initialized", "type": "server_error"}}), 503

    conf_thresh = None
    iou_thresh = None
    extract_tables = config.ENABLE_TABLE_RECOGNITION
    images_to_process = []

    # Check for multipart/form-data upload or JSON payload
    if request.content_type and "multipart/form-data" in request.content_type:
        if "confidence_threshold" in request.form:
            conf_thresh = float(request.form.get("confidence_threshold"))
        if "iou_threshold" in request.form:
            iou_thresh = float(request.form.get("iou_threshold"))
        if "extract_tables" in request.form:
            extract_tables = request.form.get("extract_tables").lower() in ("true", "1", "yes")

        for key in ("file", "image"):
            if key in request.files:
                images_to_process.append(request.files[key].read())

        if not images_to_process and request.files:
            for file_storage in request.files.values():
                images_to_process.append(file_storage.read())
    else:
        payload = request.get_json(force=True, silent=True) or {}
        if "confidence_threshold" in payload:
            conf_thresh = float(payload["confidence_threshold"])
        if "iou_threshold" in payload:
            iou_thresh = float(payload["iou_threshold"])
        if "extract_tables" in payload:
            extract_tables = bool(payload["extract_tables"])

        if "image" in payload:
            images_to_process.append(payload["image"])
        elif "images" in payload and isinstance(payload["images"], list):
            images_to_process.extend(payload["images"])
        elif "input" in payload:
            inp = payload["input"]
            if isinstance(inp, list):
                images_to_process.extend(inp)
            else:
                images_to_process.append(inp)

    if not images_to_process:
        return jsonify({
            "error": {
                "message": "No image provided. Pass 'image' (base64 or URL), 'images', or upload a file via multipart form.",
                "type": "invalid_request_error",
            }
        }), 400

    try:
        results = []
        for idx, img_input in enumerate(images_to_process):
            detections, (w, h) = doclaynet_model.predict(
                img_input,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                extract_tables=extract_tables,
            )
            results.append({
                "image_index": idx,
                "width": w,
                "height": h,
                "detections": detections,
            })

        response_payload = {
            "model": config.DOCLAYNET_MODEL_NAME,
            "results": results,
            "usage": {
                "total_images": len(images_to_process),
            },
        }

        # For single image queries, provide top-level detections directly for convenience
        if len(results) == 1:
            response_payload["detections"] = results[0]["detections"]
            response_payload["image_size"] = {
                "width": results[0]["width"],
                "height": results[0]["height"],
            }

        return jsonify(response_payload)
    except Exception as e:
        logger.exception("Error running DocLayNet layout detection")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "processing_error",
            }
        }), 500


@app.route("/v1/vision/table", methods=["POST", "OPTIONS"])
@app.route("/v1/table", methods=["POST", "OPTIONS"])
def extract_table():
    if request.method == "OPTIONS":
        return "", 200

    if doclaynet_model is None or doclaynet_model.table_recognizer is None:
        return jsonify({
            "error": {
                "message": "Table structure recognition model is not initialized or dependencies missing.",
                "type": "server_error",
            }
        }), 503

    images_to_process = []
    if request.content_type and "multipart/form-data" in request.content_type:
        for key in ("file", "image"):
            if key in request.files:
                images_to_process.append(request.files[key].read())
        if not images_to_process and request.files:
            for file_storage in request.files.values():
                images_to_process.append(file_storage.read())
    else:
        payload = request.get_json(force=True, silent=True) or {}
        if "image" in payload:
            images_to_process.append(payload["image"])
        elif "input" in payload:
            inp = payload["input"]
            if isinstance(inp, list):
                images_to_process.extend(inp)
            else:
                images_to_process.append(inp)

    if not images_to_process:
        return jsonify({
            "error": {
                "message": "No table image provided. Pass 'image' or upload a file via multipart form.",
                "type": "invalid_request_error",
            }
        }), 400

    try:
        results = []
        for img_input in images_to_process:
            loaded_img = doclaynet_model._load_image(img_input)
            table_data = doclaynet_model.table_recognizer.extract(loaded_img)
            results.append(table_data)

        if len(results) == 1:
            return jsonify({
                "model": "slanet-onnx",
                "html": results[0]["html"],
                "markdown": results[0]["markdown"],
            })
        return jsonify({
            "model": "slanet-onnx",
            "results": results,
        })
    except Exception as e:
        logger.exception("Error running table extraction")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "processing_error",
            }
        }), 500


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
    # 1. Pre-parse configuration file argument
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML configuration file")
    pre_args, _ = pre_parser.parse_known_args()

    # Load YAML configuration if specified or found at default location
    if pre_args.config or Path("config.yaml").exists() or Path("config.yml").exists():
        try:
            config.load_config(pre_args.config)
            if pre_args.config:
                logger.info(f"Loaded configuration from {pre_args.config}")
            elif Path("config.yaml").exists():
                logger.info("Loaded default configuration from config.yaml")
            elif Path("config.yml").exists():
                logger.info("Loaded default configuration from config.yml")
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Ettin ONNX Reranker, Embedding, & Vision Server (Flask)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration File
    parser.add_argument("-c", "--config", type=str, default=pre_args.config, help="Path to YAML configuration file")

    # Server Configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument("--host", type=str, default=config.HOST, help="Host address to bind to")
    server_group.add_argument("--port", type=int, default=config.PORT, help="Port to bind to")
    server_group.add_argument("--use-gpu", action="store_true", dest="use_gpu", default=config.USE_GPU, help="Use CUDA GPU if available")
    server_group.add_argument("--no-gpu", action="store_false", dest="use_gpu", help="Force CPU execution")

    # Model Loading
    model_group = parser.add_argument_group("Model Loading")
    model_group.add_argument(
        "--model-type",
        type=str,
        default=config.MODEL_TYPE,
        choices=["ettin", "embeddinggemma", "gemma", "doclaynet", "vision", "both", "all", "auto"],
        help="Model type to host",
    )
    model_group.add_argument("--model-dir", type=str, default=config.MODEL_DIR, help="Path to local directory with default model files")
    model_group.add_argument("--onnx-path", type=str, default=config.ONNX_PATH, help="Explicit path to ONNX model file (e.g. onnx/model_O4.onnx or model_qint8_arm64.onnx)")
    model_group.add_argument("--model-name", type=str, default=config.MODEL_NAME, help="Reranker model identifier/name")
    model_group.add_argument("--embedding-model-dir", type=str, default=config.EMBEDDING_MODEL_DIR, help="Path to EmbeddingGemma directory (if different from model-dir)")
    model_group.add_argument("--doclaynet-model-dir", type=str, default=config.DOCLAYNET_MODEL_DIR, help="Path to DocLayNet directory (if different from model-dir)")

    # Inference Settings
    inference_group = parser.add_argument_group("Text Inference Settings")
    inference_group.add_argument("--max-length", type=int, default=config.MAX_LENGTH, help="Maximum token sequence length")
    inference_group.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for inference")

    # Vision & Layout Settings
    vision_group = parser.add_argument_group("Vision & Layout Settings (DocLayNet)")
    vision_group.add_argument("--conf-threshold", type=float, default=config.DOCLAYNET_CONF_THRESHOLD, help="Confidence threshold for object detection")
    vision_group.add_argument("--iou-threshold", type=float, default=config.DOCLAYNET_IOU_THRESHOLD, help="IoU NMS threshold")
    vision_group.add_argument("--image-size", type=int, default=config.DOCLAYNET_IMAGE_SIZE, help="Input image dimension for YOLOv8")
    vision_group.add_argument("--table-model-path", type=str, default=config.TABLE_MODEL_PATH, help="Path to RapidTable / SLANet ONNX model file")
    vision_group.add_argument("--disable-table-rec", action="store_false", dest="enable_table_rec", default=config.ENABLE_TABLE_RECOGNITION, help="Disable table structure HTML/Markdown extraction")
    vision_group.add_argument("--enable-table-rec", action="store_true", dest="enable_table_rec", help="Enable table structure HTML/Markdown extraction")

    args = parser.parse_args()

    # Update global config with CLI overrides
    config.MODEL_TYPE = args.model_type
    config.MODEL_DIR = args.model_dir
    config.ONNX_PATH = args.onnx_path
    config.MODEL_NAME = args.model_name
    config.EMBEDDING_MODEL_DIR = args.embedding_model_dir or config.EMBEDDING_MODEL_DIR or args.model_dir
    config.DOCLAYNET_MODEL_DIR = args.doclaynet_model_dir or config.DOCLAYNET_MODEL_DIR or args.model_dir
    
    config.DOCLAYNET_CONF_THRESHOLD = args.conf_threshold
    config.DOCLAYNET_IOU_THRESHOLD = args.iou_threshold
    config.DOCLAYNET_IMAGE_SIZE = args.image_size
    config.ENABLE_TABLE_RECOGNITION = args.enable_table_rec
    config.TABLE_MODEL_PATH = args.table_model_path
    
    config.HOST = args.host
    config.PORT = args.port
    config.MAX_LENGTH = args.max_length
    config.BATCH_SIZE = args.batch_size
    config.USE_GPU = args.use_gpu

    global reranker_model, embedding_model, doclaynet_model
    from app.model import EttinONNXReranker, EmbeddingGemmaONNX, DocLayNetONNX

    model_type = config.MODEL_TYPE.lower()

    # 1. Initialize Ettin Reranker if requested
    if model_type in ("ettin", "both", "auto"):
        try:
            logger.info(f"Initializing Ettin Reranker model from directory: {config.MODEL_DIR}")
            reranker_model = EttinONNXReranker(
                model_dir=config.MODEL_DIR,
                onnx_path=config.ONNX_PATH,
                max_length=config.MAX_LENGTH,
                use_gpu=config.USE_GPU,
            )
            if reranker_model.model_name and (args.model_name == parser.get_default("model_name")):
                config.MODEL_NAME = reranker_model.model_name
        except Exception as e:
            if model_type == "ettin":
                raise e
            logger.warning(f"Could not load Ettin Reranker: {e}")

    # 2. Initialize EmbeddingGemma if requested
    if model_type in ("embeddinggemma", "gemma", "both", "auto"):
        try:
            emb_dir = config.EMBEDDING_MODEL_DIR
            logger.info(f"Initializing EmbeddingGemma model from directory: {emb_dir}")
            embedding_model = EmbeddingGemmaONNX(
                model_dir=emb_dir,
                max_length=config.MAX_LENGTH,
                use_gpu=config.USE_GPU,
                normalize_embeddings=True,
            )
        except Exception as e:
            if model_type in ("embeddinggemma", "gemma"):
                raise e
            logger.warning(f"Could not load EmbeddingGemma model: {e}")

    # 3. Initialize DocLayNet if requested
    if model_type in ("doclaynet", "vision", "all", "auto"):
        try:
            doc_dir = config.DOCLAYNET_MODEL_DIR
            logger.info(f"Initializing DocLayNet YOLOv8 model from directory: {doc_dir}")
            doclaynet_model = DocLayNetONNX(
                model_dir=doc_dir,
                conf_threshold=config.DOCLAYNET_CONF_THRESHOLD,
                iou_threshold=config.DOCLAYNET_IOU_THRESHOLD,
                image_size=config.DOCLAYNET_IMAGE_SIZE,
                use_gpu=config.USE_GPU,
                enable_table_rec=config.ENABLE_TABLE_RECOGNITION,
                table_model_path=config.TABLE_MODEL_PATH,
            )
        except Exception as e:
            if model_type in ("doclaynet", "vision"):
                raise e
            logger.warning(f"Could not load DocLayNet model: {e}")

    if reranker_model is None and embedding_model is None and doclaynet_model is None:
        logger.error("Failed to load any model!")
        sys.exit(1)

    logger.info(f"Starting Flask server on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, threaded=True)


if __name__ == "__main__":
    main()
