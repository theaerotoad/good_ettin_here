import os
import logging
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger("ettin-reranker")


class EmbeddingGemmaONNX:
    """
    Pure ONNX-based Text Embedding model loader (e.g. EmbeddingGemma / Gemma Embeddings).
    Loads local ONNX model and tokenizer.json directly.
    Computes dense vector representations with mean-pooling and L2 normalization.
    """

    def __init__(
        self,
        model_dir: str = "./model",
        onnx_path: str = None,
        max_length: int = 2048,
        use_gpu: bool = False,
        normalize_embeddings: bool = True,
    ):
        self.model_dir = model_dir
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # 1. Load Tokenizer from local directory
        tokenizer_file = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_file}")

        logger.info(f"Loading EmbeddingGemma tokenizer from {tokenizer_file}...")
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
        if onnx_path and os.path.exists(onnx_path):
            actual_onnx_path = onnx_path
        else:
            candidates = [
                os.path.join(model_dir, "model_quantized.onnx"),
                os.path.join(model_dir, "onnx", "model_quantized.onnx"),
                os.path.join(model_dir, "model.onnx"),
                os.path.join(model_dir, "onnx", "model.onnx"),
                os.path.join(model_dir, "embedding_model.onnx"),
            ]
            actual_onnx_path = None
            for cand in candidates:
                if os.path.exists(cand):
                    actual_onnx_path = cand
                    break

            if not actual_onnx_path:
                raise FileNotFoundError(
                    f"No ONNX embedding model found in '{model_dir}'. Checked: {candidates}"
                )

        logger.info(f"Loading EmbeddingGemma ONNX model from {actual_onnx_path}...")

        # Select ONNX execution providers
        available_providers = ort.get_available_providers()
        providers = []
        if use_gpu and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info(f"Initializing ONNX Embedding InferenceSession with providers: {providers}")
        self.session = ort.InferenceSession(actual_onnx_path, sess_options, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"Embedding ONNX inputs expected: {self.input_names}")
        logger.info(f"Embedding ONNX outputs expected: {self.output_names}")

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> tuple[list[list[float]], int]:
        """
        Generates dense embedding vectors for a list of text strings.
        Returns a tuple of (list_of_float_vectors, total_token_count).
        """
        if not texts:
            return [], 0

        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer.encode_batch(batch_texts)

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
            raw_output = outputs[0]
            arr = np.asarray(raw_output, dtype=np.float32)

            # If 3D (batch_size, seq_len, hidden_dim): mean pool using attention mask
            if arr.ndim == 3:
                mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
                sum_embeddings = np.sum(arr * mask_expanded, axis=1)
                sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
                batch_embeds = sum_embeddings / sum_mask
            elif arr.ndim == 2:
                batch_embeds = arr
            else:
                batch_embeds = arr.reshape(len(batch_texts), -1)

            # L2 Normalization
            if self.normalize_embeddings:
                norms = np.linalg.norm(batch_embeds, ord=2, axis=-1, keepdims=True)
                norms = np.where(norms == 0, 1e-12, norms)
                batch_embeds = batch_embeds / norms

            for vec in batch_embeds:
                all_embeddings.append([float(x) for x in vec])

        return all_embeddings, total_tokens
