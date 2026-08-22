import os

MODEL_TYPE = os.getenv("MODEL_TYPE", "ettin").lower()
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MODEL_NAME = os.getenv("MODEL_NAME", "cross-encoder/ettin-reranker-150m-v1")
ONNX_PATH = os.getenv("ONNX_PATH", None)

EMBEDDING_MODEL_DIR = os.getenv("EMBEDDING_MODEL_DIR", None)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "google/embeddinggemma-300m")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "8192"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
USE_GPU = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")

DOCLAYNET_MODEL_DIR = os.getenv("DOCLAYNET_MODEL_DIR", None)
DOCLAYNET_MODEL_NAME = os.getenv("DOCLAYNET_MODEL_NAME", "yolov8n-doclaynet")
DOCLAYNET_CONF_THRESHOLD = float(os.getenv("DOCLAYNET_CONF_THRESHOLD", "0.25"))
DOCLAYNET_IOU_THRESHOLD = float(os.getenv("DOCLAYNET_IOU_THRESHOLD", "0.45"))
DOCLAYNET_IMAGE_SIZE = int(os.getenv("DOCLAYNET_IMAGE_SIZE", "640"))
ENABLE_TABLE_RECOGNITION = os.getenv("ENABLE_TABLE_RECOGNITION", "true").lower() in ("true", "1", "yes")
TABLE_MODEL_PATH = os.getenv("TABLE_MODEL_PATH", None)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
