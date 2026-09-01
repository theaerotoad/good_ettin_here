import os
from pathlib import Path
from typing import Optional, Any, Dict
import yaml

# ------------------------------------------------------------------------------
# Default settings initialized from environment variables
# ------------------------------------------------------------------------------
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


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file if specified or found at default locations,
    updating the module-level configuration variables.
    """
    target_path = None
    if config_path:
        target_path = Path(config_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    else:
        for candidate in ["config.yaml", "config.yml"]:
            cand_p = Path(candidate)
            if cand_p.exists():
                target_path = cand_p
                break

    if not target_path:
        return {}

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    global MODEL_TYPE, MODEL_DIR, MODEL_NAME, ONNX_PATH
    global EMBEDDING_MODEL_DIR, EMBEDDING_MODEL_NAME, MAX_LENGTH, BATCH_SIZE, USE_GPU
    global DOCLAYNET_MODEL_DIR, DOCLAYNET_MODEL_NAME, DOCLAYNET_CONF_THRESHOLD
    global DOCLAYNET_IOU_THRESHOLD, DOCLAYNET_IMAGE_SIZE, ENABLE_TABLE_RECOGNITION, TABLE_MODEL_PATH
    global HOST, PORT

    # Server settings
    server_cfg = data.get("server", {})
    if isinstance(server_cfg, dict):
        if "host" in server_cfg:
            HOST = str(server_cfg["host"])
        if "port" in server_cfg:
            PORT = int(server_cfg["port"])
        if "use_gpu" in server_cfg:
            USE_GPU = bool(server_cfg["use_gpu"])

    if "host" in data:
        HOST = str(data["host"])
    if "port" in data:
        PORT = int(data["port"])
    if "use_gpu" in data:
        USE_GPU = bool(data["use_gpu"])

    # Global / Top-level model settings
    if "model_type" in data:
        MODEL_TYPE = str(data["model_type"]).lower()
    if "model_dir" in data:
        MODEL_DIR = str(data["model_dir"])
    if "onnx_path" in data:
        ONNX_PATH = data["onnx_path"]
    if "model_name" in data:
        MODEL_NAME = str(data["model_name"])
    if "max_length" in data:
        MAX_LENGTH = int(data["max_length"])
    if "batch_size" in data:
        BATCH_SIZE = int(data["batch_size"])

    # Reranker settings
    reranker_cfg = data.get("reranker", {})
    if isinstance(reranker_cfg, dict):
        if "model_dir" in reranker_cfg:
            MODEL_DIR = str(reranker_cfg["model_dir"])
        if "onnx_path" in reranker_cfg:
            ONNX_PATH = reranker_cfg["onnx_path"]
        if "model_name" in reranker_cfg:
            MODEL_NAME = str(reranker_cfg["model_name"])
        if "max_length" in reranker_cfg:
            MAX_LENGTH = int(reranker_cfg["max_length"])
        if "batch_size" in reranker_cfg:
            BATCH_SIZE = int(reranker_cfg["batch_size"])

    # Embedding settings
    embedding_cfg = data.get("embedding", {})
    if isinstance(embedding_cfg, dict):
        if "model_dir" in embedding_cfg:
            EMBEDDING_MODEL_DIR = str(embedding_cfg["model_dir"])
        if "model_name" in embedding_cfg:
            EMBEDDING_MODEL_NAME = str(embedding_cfg["model_name"])
        if "max_length" in embedding_cfg and "max_length" not in reranker_cfg:
            MAX_LENGTH = int(embedding_cfg["max_length"])
        if "batch_size" in embedding_cfg and "batch_size" not in reranker_cfg:
            BATCH_SIZE = int(embedding_cfg["batch_size"])

    if "embedding_model_dir" in data:
        EMBEDDING_MODEL_DIR = str(data["embedding_model_dir"])
    if "embedding_model_name" in data:
        EMBEDDING_MODEL_NAME = str(data["embedding_model_name"])

    # Vision & Layout settings
    vision_cfg = data.get("vision", {})
    if isinstance(vision_cfg, dict):
        layout_cfg = vision_cfg.get("layout", {})
        if isinstance(layout_cfg, dict):
            if "model_dir" in layout_cfg:
                DOCLAYNET_MODEL_DIR = str(layout_cfg["model_dir"])
            if "model_name" in layout_cfg:
                DOCLAYNET_MODEL_NAME = str(layout_cfg["model_name"])
            if "conf_threshold" in layout_cfg:
                DOCLAYNET_CONF_THRESHOLD = float(layout_cfg["conf_threshold"])
            if "iou_threshold" in layout_cfg:
                DOCLAYNET_IOU_THRESHOLD = float(layout_cfg["iou_threshold"])
            if "image_size" in layout_cfg:
                DOCLAYNET_IMAGE_SIZE = int(layout_cfg["image_size"])
            if "extract_tables" in layout_cfg:
                ENABLE_TABLE_RECOGNITION = bool(layout_cfg["extract_tables"])

        table_cfg = vision_cfg.get("table", {})
        if isinstance(table_cfg, dict):
            if "model_path" in table_cfg:
                TABLE_MODEL_PATH = table_cfg["model_path"]
            if "enable" in table_cfg:
                ENABLE_TABLE_RECOGNITION = bool(table_cfg["enable"])

        if "model_dir" in vision_cfg:
            DOCLAYNET_MODEL_DIR = str(vision_cfg["model_dir"])
        if "model_name" in vision_cfg:
            DOCLAYNET_MODEL_NAME = str(vision_cfg["model_name"])
        if "conf_threshold" in vision_cfg:
            DOCLAYNET_CONF_THRESHOLD = float(vision_cfg["conf_threshold"])
        if "iou_threshold" in vision_cfg:
            DOCLAYNET_IOU_THRESHOLD = float(vision_cfg["iou_threshold"])
        if "image_size" in vision_cfg:
            DOCLAYNET_IMAGE_SIZE = int(vision_cfg["image_size"])
        if "enable_table_recognition" in vision_cfg:
            ENABLE_TABLE_RECOGNITION = bool(vision_cfg["enable_table_recognition"])
        if "table_model_path" in vision_cfg:
            TABLE_MODEL_PATH = vision_cfg["table_model_path"]

    if "doclaynet_model_dir" in data:
        DOCLAYNET_MODEL_DIR = str(data["doclaynet_model_dir"])
    if "doclaynet_model_name" in data:
        DOCLAYNET_MODEL_NAME = str(data["doclaynet_model_name"])
    if "conf_threshold" in data:
        DOCLAYNET_CONF_THRESHOLD = float(data["conf_threshold"])
    if "iou_threshold" in data:
        DOCLAYNET_IOU_THRESHOLD = float(data["iou_threshold"])
    if "image_size" in data:
        DOCLAYNET_IMAGE_SIZE = int(data["image_size"])
    if "enable_table_recognition" in data:
        ENABLE_TABLE_RECOGNITION = bool(data["enable_table_recognition"])
    if "table_model_path" in data:
        TABLE_MODEL_PATH = data["table_model_path"]

    return data