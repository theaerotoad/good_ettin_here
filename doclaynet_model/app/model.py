<<<<<<< SEARCH
import os
import io
import math
import base64
import logging
import requests
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer
from safetensors.numpy import load_file
=======
import os
import io
import json
import math
import base64
import logging
import requests
import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer
from safetensors.numpy import load_file
>>>>>>> REPLACE

```

```python
<<<<<<< SEARCH
        # 1. Locate ONNX model file
        if onnx_path and os.path.exists(onnx_path):
            actual_onnx_path = onnx_path
        else:
            candidates = [
                os.path.join(model_dir, "onnx", "yolov8n-doclaynet.onnx"),
                os.path.join(model_dir, "yolov8n-doclaynet.onnx"),
                os.path.join(model_dir, "model.onnx"),
                os.path.join(model_dir, "doclaynet.onnx"),
                os.path.join(model_dir, "model_quantized.onnx"),
            ]
            actual_onnx_path = None
            for cand in candidates:
                if os.path.exists(cand):
                    actual_onnx_path = cand
                    break

            if not actual_onnx_path:
                raise FileNotFoundError(
                    f"No ONNX DocLayNet model found in '{model_dir}'. Checked: {candidates}"
                )

        logger.info(f"Loading YOLOv8 DocLayNet ONNX model from {actual_onnx_path}...")
=======
        # 1. Load labels dynamically from config.json if present
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "id2label" in cfg and isinstance(cfg["id2label"], dict):
                    loaded_labels = [
                        cfg["id2label"][str(i)]
                        for i in range(len(cfg["id2label"]))
                        if str(i) in cfg["id2label"]
                    ]
                    if loaded_labels:
                        self.labels = loaded_labels
                        logger.info(f"Loaded {len(self.labels)} class labels from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load id2label from {config_path}: {e}")

        # 2. Locate ONNX model file
        if onnx_path and os.path.exists(onnx_path):
            actual_onnx_path = onnx_path
        else:
            candidates = [
                os.path.join(model_dir, "onnx", "model.onnx"),
                os.path.join(model_dir, "onnx", "model_quantized.onnx"),
                os.path.join(model_dir, "model.onnx"),
                os.path.join(model_dir, "model_quantized.onnx"),
                os.path.join(model_dir, "onnx", "yolov8x-doclaynet.onnx"),
                os.path.join(model_dir, "onnx", "yolov8n-doclaynet.onnx"),
                os.path.join(model_dir, "yolov8x-doclaynet.onnx"),
                os.path.join(model_dir, "yolov8n-doclaynet.onnx"),
                os.path.join(model_dir, "doclaynet.onnx"),
            ]
            actual_onnx_path = None
            for cand in candidates:
                if os.path.exists(cand):
                    actual_onnx_path = cand
                    break

            # Fallback scan for any .onnx file in model_dir or model_dir/onnx
            if not actual_onnx_path and os.path.isdir(model_dir):
                for root, _, files in os.walk(model_dir):
                    for file in files:
                        if file.endswith(".onnx"):
                            actual_onnx_path = os.path.join(root, file)
                            break
                    if actual_onnx_path:
                        break

            if not actual_onnx_path:
                raise FileNotFoundError(
                    f"No ONNX DocLayNet model found in '{model_dir}'. Checked: {candidates}"
                )

        logger.info(f"Loading YOLOv8 DocLayNet ONNX model from {actual_onnx_path}...")
>>>>>>> REPLACE

```

```python
<<<<<<< SEARCH
        self.session = ort.InferenceSession(actual_onnx_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"DocLayNet ONNX inputs expected: {self.input_names}")
        logger.info(f"DocLayNet ONNX outputs expected: {self.output_names}")
=======
        self.session = ort.InferenceSession(actual_onnx_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"DocLayNet ONNX inputs expected: {self.input_names}")
        logger.info(f"DocLayNet ONNX outputs expected: {self.output_names}")

        # If model expects a specific fixed input size, adopt it automatically
        try:
            inp_shape = self.session.get_inputs()[0].shape
            if len(inp_shape) == 4 and isinstance(inp_shape[2], int) and inp_shape[2] > 0:
                self.image_size = inp_shape[2]
                logger.info(f"Using ONNX model input dimensions: {self.image_size}x{self.image_size}")
        except Exception:
            pass
>>>>>>> REPLACE
