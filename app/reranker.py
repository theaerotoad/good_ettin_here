import os
import json
import math
import logging
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from safetensors.numpy import load_file

from .utils import _gelu_numpy, _get_tensor

logger = logging.getLogger("ettin-reranker")


class EttinONNXReranker:
    """
    Pure ONNX-based Cross-Encoder reranker supporting all Ettin model sizes
    (17M, 32M, 68M, 150M, 400M, 1B) and quantization formats.
    """

    def __init__(
        self,
        model_dir: str = "./model",
        onnx_path: str = None,
        max_length: int = 8192,
        use_gpu: bool = False,
        normalize_scores: bool = True,
    ):
        self.model_dir = model_dir
        self.max_length = max_length
        self.normalize_scores = normalize_scores
        self.model_name = self._detect_model_name()

        # 1. Load Tokenizer from local directory
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_file}")

        logger.info(f"Loading tokenizer from {tokenizer_file}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_file)

        # Configure pad token dynamically
        pad_id = self.tokenizer.token_to_id("[PAD]")
        pad_token = "[PAD]"
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("<pad>")
            pad_token = "<pad>"
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("<|endoftext|>")
            pad_token = "<|endoftext|>"
        if pad_id is None:
            pad_id = 0
            pad_token = "[PAD]"

        self.tokenizer.enable_truncation(max_length=self.max_length, strategy="longest_first")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

        # 2. Locate and load local ONNX model
        actual_onnx_path = self._find_onnx_model(model_dir, onnx_path)
        logger.info(f"Loading ONNX model ({self.model_name}) from {actual_onnx_path}...")

        # Select ONNX execution providers
        available_providers = ort.get_available_providers()
        providers = []
        if use_gpu and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info(f"Initializing ONNX InferenceSession with providers: {providers}")
        self.session = ort.InferenceSession(actual_onnx_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"ONNX Model inputs expected: {self.input_names}")
        logger.info(f"ONNX Model outputs available: {self.output_names}")

        # 3. Load classification head weights if required for backbone-only ONNX graphs
        self.head_weights = self._load_classification_head()

    def _detect_model_name(self) -> str:
        """Attempts to infer the Ettin model variant from config.json or directory path."""
        config_file = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "_name_or_path" in data and data["_name_or_path"]:
                        return str(data["_name_or_path"])
            except Exception:
                pass
        dir_name = os.path.basename(os.path.abspath(self.model_dir))
        if "ettin" in dir_name.lower():
            return f"cross-encoder/{dir_name}"
        return "cross-encoder/ettin-reranker"

    @staticmethod
    def _is_valid_onnx_file(path: str) -> bool:
        if not path or not os.path.isfile(path):
            return False
        try:
            size = os.path.getsize(path)
            if size < 1024:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(100)
                    if "git-lfs" in head or "oid sha256" in head:
                        logger.warning(f"File {path} is an unresolved Git LFS pointer, skipping.")
                        return False
                return False
            return True
        except Exception:
            return False

    def _find_onnx_model(self, model_dir: str, onnx_path: str = None) -> str:
        """Locates ONNX model across standard Hugging Face Optimum and quantized conventions."""
        if onnx_path and self._is_valid_onnx_file(onnx_path):
            return onnx_path

        if onnx_path:
            for parent in [model_dir, os.path.join(model_dir, "onnx")]:
                cand = os.path.join(parent, onnx_path)
                if self._is_valid_onnx_file(cand):
                    return cand

        candidates = [
            os.path.join(model_dir, "onnx", "model.onnx"),
            os.path.join(model_dir, "model.onnx"),
            os.path.join(model_dir, "onnx", "model_O4.onnx"),
            os.path.join(model_dir, "model_O4.onnx"),
            os.path.join(model_dir, "onnx", "model_O3.onnx"),
            os.path.join(model_dir, "model_O3.onnx"),
            os.path.join(model_dir, "onnx", "model_O2.onnx"),
            os.path.join(model_dir, "model_O2.onnx"),
            os.path.join(model_dir, "onnx", "model_O1.onnx"),
            os.path.join(model_dir, "model_O1.onnx"),
            os.path.join(model_dir, "onnx", "model_quint8_avx2.onnx"),
            os.path.join(model_dir, "model_quint8_avx2.onnx"),
            os.path.join(model_dir, "onnx", "model_qint8_avx512.onnx"),
            os.path.join(model_dir, "model_qint8_avx512.onnx"),
            os.path.join(model_dir, "onnx", "model_qint8_avx512_vnni.onnx"),
            os.path.join(model_dir, "model_qint8_avx512_vnni.onnx"),
            os.path.join(model_dir, "onnx", "model_qint8_arm64.onnx"),
            os.path.join(model_dir, "model_qint8_arm64.onnx"),
            os.path.join(model_dir, "model_quantized.onnx"),
            os.path.join(model_dir, "onnx", "model_quantized.onnx"),
        ]

        for cand in candidates:
            if self._is_valid_onnx_file(cand):
                return cand

        for search_dir in [os.path.join(model_dir, "onnx"), model_dir]:
            if os.path.exists(search_dir):
                for fname in sorted(os.listdir(search_dir)):
                    if fname.endswith(".onnx"):
                        cand = os.path.join(search_dir, fname)
                        if self._is_valid_onnx_file(cand):
                            return cand

        raise FileNotFoundError(
            f"No valid ONNX model file found in '{model_dir}'. Checked standard paths and subdirectories."
        )

    def _load_classification_head(self) -> dict:
        """
        Loads classification head weights (2_Dense, 3_LayerNorm, 4_Dense) from local safetensors files.
        """
        weights = {}

        def find_safetensor(module_name: str):
            candidates = [
                os.path.join(self.model_dir, module_name, "model.safetensors"),
                os.path.join(self.model_dir, f"{module_name}.safetensors"),
                os.path.join(self.model_dir, module_name, "dense.safetensors"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
            return None

        d2_path = find_safetensor("2_Dense")
        ln3_path = find_safetensor("3_LayerNorm")
        d4_path = find_safetensor("4_Dense")

        if d2_path and ln3_path and d4_path:
            logger.info("Loading classification head weights from safetensors...")
            weights["d2"] = load_file(d2_path)
            weights["ln3"] = load_file(ln3_path)
            weights["d4"] = load_file(d4_path)
            logger.info("Successfully loaded classification head weights.")
        else:
            root_sf = os.path.join(self.model_dir, "model.safetensors")
            if os.path.exists(root_sf):
                try:
                    full_weights = load_file(root_sf)
                    d2_tensors = {k: v for k, v in full_weights.items() if "2_Dense" in k or "classifier.dense" in k}
                    ln3_tensors = {k: v for k, v in full_weights.items() if "3_LayerNorm" in k or "classifier.layer_norm" in k or "classifier.norm" in k}
                    d4_tensors = {k: v for k, v in full_weights.items() if "4_Dense" in k or "classifier.out_proj" in k or "classifier.linear" in k}
                    if d2_tensors and ln3_tensors and d4_tensors:
                        weights["d2"] = d2_tensors
                        weights["ln3"] = ln3_tensors
                        weights["d4"] = d4_tensors
                        logger.info("Successfully extracted classification head weights from root model.safetensors.")
                except Exception as e:
                    logger.warning(f"Could not parse root safetensors: {e}")

        if not weights:
            logger.info(
                "Classification head safetensors not found locally. "
                "Will assume ONNX model outputs logits directly."
            )

        return weights

    def _forward_classification_head(self, cls_embeddings: np.ndarray) -> np.ndarray:
        """
        Executes Pooling(CLS) -> Dense(H, H) -> GELU -> LayerNorm(H) -> Dense(H, 1) in NumPy.
        """
        if not self.head_weights:
            raise ValueError("Classification head weights are missing.")

        d2 = self.head_weights["d2"]
        ln3 = self.head_weights["ln3"]
        d4 = self.head_weights["d4"]

        # 1. Dense 2: Linear(H -> H) + GELU
        w2 = _get_tensor(d2, "linear.weight", "weight", "0.weight", "classifier.dense.weight", "2_Dense.linear.weight")
        b2 = _get_tensor(d2, "linear.bias", "bias", "0.bias", "classifier.dense.bias", "2_Dense.linear.bias")
        if w2 is None:
            raise KeyError("Could not find linear.weight in 2_Dense safetensors")

        x = cls_embeddings @ w2.T
        if b2 is not None:
            x = x + b2
        x = _gelu_numpy(x)

        # 2. LayerNorm 3: LayerNorm(H)
        w_ln = _get_tensor(ln3, "norm.weight", "layer_norm.weight", "weight", "gamma", "0.weight", "classifier.layer_norm.weight", "classifier.norm.weight", "3_LayerNorm.norm.weight")
        b_ln = _get_tensor(ln3, "norm.bias", "layer_norm.bias", "bias", "beta", "0.bias", "classifier.layer_norm.bias", "classifier.norm.bias", "3_LayerNorm.norm.bias")

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        eps = 1e-5
        x = (x - mean) / np.sqrt(var + eps)
        if w_ln is not None:
            x = x * w_ln
        if b_ln is not None:
            x = x + b_ln

        # 3. Dense 4: Linear(H -> 1)
        w4 = _get_tensor(d4, "linear.weight", "weight", "0.weight", "classifier.out_proj.weight", "classifier.linear.weight", "4_Dense.linear.weight")
        b4 = _get_tensor(d4, "linear.bias", "bias", "0.bias", "classifier.out_proj.bias", "classifier.linear.bias", "4_Dense.linear.bias")
        if w4 is None:
            raise KeyError("Could not find linear.weight in 4_Dense safetensors")

        scores = x @ w4.T
        if b4 is not None:
            scores = scores + b4

        return scores.squeeze(-1)

    def _sigmoid_rescale(self, score: float) -> float:
        """
        Standard Sigmoid normalization: 1 / (1 + exp(-score)).
        """
        return 1.0 / (1.0 + math.exp(-score))

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
    ) -> tuple[list[float], int]:
        """
        Predict relevance scores for a list of (query, document) tuples.
        """
        if not pairs:
            return [], 0

        all_scores = []
        total_tokens = 0

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]

            encodings = self.tokenizer.encode_batch(batch_pairs)

            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            for e in encodings:
                total_tokens += len(e.ids)

            onnx_inputs = {}
            if "input_ids" in self.input_names:
                onnx_inputs["input_ids"] = input_ids
            if "attention_mask" in self.input_names:
                onnx_inputs["attention_mask"] = attention_mask
            if "token_type_ids" in self.input_names:
                token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
                onnx_inputs["token_type_ids"] = token_type_ids

            outputs = self.session.run(None, onnx_inputs)

            raw_output = None
            if len(outputs) == 1:
                raw_output = outputs[0]
            else:
                for idx, out_meta in enumerate(self.session.get_outputs()):
                    name_lower = out_meta.name.lower()
                    if any(k in name_lower for k in ("logits", "score", "output")):
                        raw_output = outputs[idx]
                        break
                if raw_output is None:
                    raw_output = outputs[0]

            arr = np.asarray(raw_output)

            # Case 1: Model outputs last_hidden_state (batch_size, seq_len, hidden_dim)
            if arr.ndim == 3:
                cls_embeddings = arr[:, 0, :]
                raw_scores = self._forward_classification_head(cls_embeddings)
                batch_scores = [float(s) for s in raw_scores]

            # Case 2: Model outputs 2D or 1D scalar logits directly
            elif arr.ndim == 2:
                batch_scores = [float(x) for x in arr[:, 0 if arr.shape[1] == 1 else -1]]
            elif arr.ndim == 1:
                batch_scores = [float(x) for x in arr]
            else:
                batch_scores = [float(arr)]

            for score in batch_scores:
                if self.normalize_scores:
                    score = self._sigmoid_rescale(score)
                all_scores.append(score)

        return all_scores, total_tokens
