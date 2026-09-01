import os
import json
import math
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
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

        # Configure thread concurrency for optimal CPU execution
        cpu_count = os.cpu_count() or 4
        intra_threads = int(os.getenv("ORT_INTRA_OP_THREADS", str(cpu_count)))
        inter_threads = int(os.getenv("ORT_INTER_OP_THREADS", "1"))
        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = inter_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info(
            f"Initializing ONNX InferenceSession (intra_threads={intra_threads}, inter_threads={inter_threads}) with providers: {providers}"
        )
        self.session = ort.InferenceSession(actual_onnx_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"ONNX Model inputs expected: {self.input_names}")
        logger.info(f"ONNX Model outputs available: {self.output_names}")

        # 3. Load classification head weights if required for backbone-only ONNX graphs
        self.head_weights = self._load_classification_head()

        # Fail fast with an actionable message if backbone-only ONNX graph lacks head weights
        backbone_only = (
            any("last_hidden_state" in out.lower() or "token_embeddings" in out.lower() for out in self.output_names)
            and not any("logits" in out.lower() or "score" in out.lower() for out in self.output_names)
        )
        if backbone_only and not self.head_weights:
            repo_id = self._resolve_hf_repo_id()
            raise FileNotFoundError(
                f"The ONNX model in '{model_dir}' outputs 'last_hidden_state' (backbone only) and requires "
                f"the classification head weights ('2_Dense', '3_LayerNorm', '4_Dense').\n"
                f"Please ensure all model repository files are downloaded:\n"
                f"  huggingface-cli download {repo_id} --local-dir {model_dir}"
            )

    def _detect_model_name(self) -> str:
        """Attempts to infer the Ettin model variant from config.json or directory path."""
        return self._resolve_hf_repo_id()

    def _resolve_hf_repo_id(self) -> str:
        """Resolves the canonical Hugging Face repository ID for the current model directory."""
        config_file = os.path.join(self.model_dir, "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    name = str(data.get("_name_or_path", ""))
                    if "cross-encoder/ettin-reranker-" in name:
                        return name
                    for size in ["17m", "32m", "68m", "150m", "400m", "1b"]:
                        if size in name.lower():
                            return f"cross-encoder/ettin-reranker-{size}-v1"
            except Exception:
                pass

        combined = f"{self.model_dir}".lower()
        for size in ["17m", "32m", "68m", "150m", "400m", "1b"]:
            if size in combined:
                return f"cross-encoder/ettin-reranker-{size}-v1"

        return "cross-encoder/ettin-reranker-150m-v1"

    @staticmethod
    def _download_file(url: str, dest_path: str, timeout: int = 30) -> None:
        """Downloads a remote file with standard headers and ensures parent directory exists."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Ettin-ONNX-Server)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())

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
            os.path.join(model_dir, "model_quantized.onnx"),
            os.path.join(model_dir, "onnx", "model_quantized.onnx"),
            os.path.join(model_dir, "onnx", "model_qint8_avx512_vnni.onnx"),
            os.path.join(model_dir, "model_qint8_avx512_vnni.onnx"),
            os.path.join(model_dir, "onnx", "model_qint8_avx512.onnx"),
            os.path.join(model_dir, "model_qint8_avx512.onnx"),
            os.path.join(model_dir, "onnx", "model_quint8_avx2.onnx"),
            os.path.join(model_dir, "model_quint8_avx2.onnx"),
            os.path.join(model_dir, "onnx", "model_qint8_arm64.onnx"),
            os.path.join(model_dir, "model_qint8_arm64.onnx"),
            os.path.join(model_dir, "onnx", "model_O4.onnx"),
            os.path.join(model_dir, "model_O4.onnx"),
            os.path.join(model_dir, "onnx", "model_O3.onnx"),
            os.path.join(model_dir, "model_O3.onnx"),
            os.path.join(model_dir, "onnx", "model_O2.onnx"),
            os.path.join(model_dir, "model_O2.onnx"),
            os.path.join(model_dir, "onnx", "model_O1.onnx"),
            os.path.join(model_dir, "model_O1.onnx"),
            os.path.join(model_dir, "onnx", "model.onnx"),
            os.path.join(model_dir, "model.onnx"),
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

    @staticmethod
    def _is_valid_safetensors(path: str) -> bool:
        """Validates that a safetensors file exists, is not an unresolved Git LFS pointer, and is readable."""
        if not path or not os.path.isfile(path):
            return False
        try:
            size = os.path.getsize(path)
            if size < 1000:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(100)
                    if "git-lfs" in head or "oid sha256" in head:
                        return False
            load_file(path)
            return True
        except Exception:
            return False

    def _load_classification_head(self) -> dict:
        """
        Loads classification head weights (2_Dense, 3_LayerNorm, 4_Dense) from local safetensors,
        falling back to root safetensors or automatic download from Hugging Face Hub if missing.
        """
        weights = {}

        def find_module_safetensor(module_names: list[str]) -> Optional[str]:
            for mod in module_names:
                for fname in ["model.safetensors", f"{mod}.safetensors", "dense.safetensors", "layer_norm.safetensors"]:
                    cand = os.path.join(self.model_dir, mod, fname)
                    if self._is_valid_safetensors(cand):
                        return cand
                    cand = os.path.join(self.model_dir, fname)
                    if mod.lower() in fname.lower() and self._is_valid_safetensors(cand):
                        return cand

            for root, _, files in os.walk(self.model_dir):
                for f in files:
                    if f.endswith(".safetensors"):
                        full_path = os.path.join(root, f)
                        parent = os.path.basename(root).lower()
                        for mod in module_names:
                            if (mod.lower() in parent or mod.lower() in f.lower()) and self._is_valid_safetensors(full_path):
                                return full_path
            return None

        # 1. Look for per-module safetensors subdirectories
        d2_path = find_module_safetensor(["2_Dense", "2_dense", "dense"])
        ln3_path = find_module_safetensor(["3_LayerNorm", "3_layernorm", "layernorm"])
        d4_path = find_module_safetensor(["4_Dense", "4_dense", "out_proj"])

        if d2_path and ln3_path and d4_path:
            logger.info("Loading classification head weights from module safetensors...")
            weights["d2"] = load_file(d2_path)
            weights["ln3"] = load_file(ln3_path)
            weights["d4"] = load_file(d4_path)
            logger.info("Successfully loaded classification head weights.")
            return weights

        # 2. Look for unified root model.safetensors
        for cand in [os.path.join(self.model_dir, "model.safetensors"), os.path.join(self.model_dir, "pytorch_model.safetensors")]:
            if self._is_valid_safetensors(cand):
                try:
                    full_weights = load_file(cand)
                    d2_tensors = {k: v for k, v in full_weights.items() if any(x in k for x in ["2_Dense", "2.linear", "classifier.dense"])}
                    ln3_tensors = {k: v for k, v in full_weights.items() if any(x in k for x in ["3_LayerNorm", "3.norm", "classifier.layer_norm", "classifier.norm"])}
                    d4_tensors = {k: v for k, v in full_weights.items() if any(x in k for x in ["4_Dense", "4.linear", "classifier.out_proj", "classifier.linear"])}
                    if d2_tensors and ln3_tensors and d4_tensors:
                        weights["d2"] = d2_tensors
                        weights["ln3"] = ln3_tensors
                        weights["d4"] = d4_tensors
                        logger.info("Successfully extracted classification head weights from root model.safetensors.")
                        return weights
                except Exception as e:
                    logger.warning(f"Could not parse root safetensors: {e}")

        # 3. Auto-download tiny head files from Hugging Face if model outputs raw embeddings
        repo_id = self._resolve_hf_repo_id()
        if repo_id:
            logger.info(f"Classification head safetensors missing locally. Attempting to fetch from Hugging Face Hub ({repo_id})...")
            files_to_download = [
                ("2_Dense/model.safetensors", os.path.join(self.model_dir, "2_Dense", "model.safetensors")),
                ("3_LayerNorm/model.safetensors", os.path.join(self.model_dir, "3_LayerNorm", "model.safetensors")),
                ("4_Dense/model.safetensors", os.path.join(self.model_dir, "4_Dense", "model.safetensors")),
            ]
            all_downloaded = True
            for hf_subpath, local_dest in files_to_download:
                url = f"https://huggingface.co/{repo_id}/resolve/main/{hf_subpath}"
                try:
                    logger.info(f"Downloading classification head component: {hf_subpath}...")
                    self._download_file(url, local_dest)
                    if not self._is_valid_safetensors(local_dest):
                        all_downloaded = False
                        break
                except Exception as dl_err:
                    logger.warning(f"Failed to auto-download {hf_subpath}: {dl_err}")
                    all_downloaded = False
                    break

            if all_downloaded:
                weights["d2"] = load_file(files_to_download[0][1])
                weights["ln3"] = load_file(files_to_download[1][1])
                weights["d4"] = load_file(files_to_download[2][1])
                logger.info("Successfully downloaded and initialized classification head weights.")
                return weights

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
        normalize: Optional[bool] = None,
    ) -> tuple[list[float], int]:
        """
        Predict relevance scores for a list of (query, document) tuples.
        """
        if not pairs:
            return [], 0

        use_normalize = self.normalize_scores if normalize is None else normalize
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
                if use_normalize:
                    score = self._sigmoid_rescale(score)
                all_scores.append(score)

        return all_scores, total_tokens
