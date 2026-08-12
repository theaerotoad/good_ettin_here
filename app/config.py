import os

MODEL_TYPE = os.getenv("MODEL_TYPE", "ettin").lower()
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
ONNX_PATH = os.getenv("ONNX_PATH", None)
MODEL_NAME = os.getenv("MODEL_NAME", "cross-encoder/ettin-reranker-150m-v1")

EMBEDDING_MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR", None)
EMBEDDING_ONNX_PATH = os.getenv("EMBEDDING_ONNX_PATH", None)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "8192"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
USE_GPU = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
NORMALIZE_SCORES = os.getenv("NORMALIZE_SCORES", "false").lower() in ("true", "1", "yes")
