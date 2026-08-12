# Ettin 150M ONNX Reranker API (Ultra-Lightweight Flask Edition)

An ultra-fast, production-ready Python API server for **`cross-encoder/ettin-reranker-150m-v1`** using **ONNX Runtime and Flask ONLY**.

Only 5 dependencies! Zero PyTorch, Zero Transformers, Zero FastAPI/Uvicorn, Zero HuggingFace Hub.  Goal is to have a drop in python library for llama-swap that runs a llamacpp / OpenAI compatible re-ranker for use in things like qmdpy.

---

## Local Directory Structure Expected

Place your local model files in a single directory (e.g., `./model`):

```text
model/
├── tokenizer.json
├── onnx/
│   └── model.onnx
├── 2_Dense/
│   └── model.safetensors
├── 3_LayerNorm/
│   └── model.safetensors
└── 4_Dense/
    └── model.safetensors

```

> **Tip:** You can download the model folder once using `git lfs clone https://huggingface.co/cross-encoder/ettin-reranker-150m-v1 ./model` or `huggingface-cli download cross-encoder/ettin-reranker-150m-v1 --local-dir ./model`.

---

## CLI Options

```bash
python -m app.server --help

```

| Argument | Default | Description |
| --- | --- | --- |
| `--model-dir` | `./model` | Path to local directory containing model weights |
| `--onnx-path` | `None` | Optional direct path to `model.onnx` |
| `--host` | `0.0.0.0` | Host address to bind server |
| `--port` | `8000` | Port to bind server |
| `--max-length` | `8192` | Maximum context length |
| `--batch-size` | `32` | Batch size for ONNX inference |
| `--use-gpu` | `False` | Enable CUDA Execution Provider if available |
| `--normalize-scores` | `True` | Apply sigmoid score normalization `sigmoid(score / 5.0)` |
| `--no-normalize-scores` | `False` | Disable normalization and output raw logits |

---

## Running the Server

### 1. Direct Python Execution

```bash
pip install -r requirements.txt

# Run with custom model path and port
python -m app.server --model-dir /path/to/my_model --host 0.0.0.0 --port 8000

```

### 2. Docker Execution

```bash
docker build -t ettin-reranker-flask .
docker run -p 8000:8000 -v /path/to/my_model:/app/model ettin-reranker-flask

```

---

## API Endpoints

### 1. `/v1/embeddings` (OpenAI Compatible)

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cross-encoder/ettin-reranker-150m-v1",
    "query": "Which planet is known as the Red Planet?",
    "input": [
      "Venus is often called Earth twin because of its size.",
      "Mars, known for its reddish appearance, is often referred to as the Red Planet."
    ]
  }'

```

### 2. `/v1/rerank` or `/rerank` (Cohere Compatible)

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cross-encoder/ettin-reranker-150m-v1",
    "query": "Which planet is known as the Red Planet?",
    "documents": [
      "Venus is often called Earth twin because of its size.",
      "Mars, known for its reddish appearance, is often referred to as the Red Planet."
    ],
    "top_n": 2,
    "return_documents": true
  }'

```
