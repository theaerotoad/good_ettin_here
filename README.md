# Ettin 150M ONNX Reranker & EmbeddingGemma API Server (Ultra-Lightweight Flask Edition)

An ultra-lightweight, high-performance Flask API server for running local ONNX models for document reranking and text embeddings without heavy PyTorch dependencies. 

Limited dependancies.  Goal is to have a drop in python library for llama-swap that runs a llamacpp / OpenAI compatible re-ranker for use in things like qmdpy.

---

## tldr; Getting Set up

Clone this repository, then set up a virtual environemnt (recommended):

```bash
git clone https://github.com/theaerotoad/good_ettin_here
cd good_ettin_here
pip install -r requirements.txt
```

### Downloading Embedings

Next, download the pre-quantized ONNX version of EmbeddingGemma directly using Hugging Face CLI:

```bash
hf download onnx-community/embeddinggemma-300m-ONNX  --local-dir ./ettinreranker_model

hf download onnx-community/embeddinggemma-300m-ONNX --local-dir ./embeddinggemma_model

```

### Launch the v1 endpoint compatible server

```bash
./app/server.py --model-dir ettinreranker_model --embedding-model-dir /home/aerotoad/software/pyplayground/ettin-rerank_share/embeddinggemma --model-type both --host 127.0.0.1 --port 8000
```

You can kill it with Ctrl-C.

## Local Directory Structure Expected

Your local directory structure for models will look like this:

```text
.
├── ettinreranker_model/                     # Ettin Reranker files
│   ├── model.onnx (or onnx/model.onnx)
│   ├── tokenizer.json
│   ├── 2_Dense.safetensors
│   ├── 3_LayerNorm.safetensors
│   └── 4_Dense.safetensors
└── embeddinggemma_model/     # EmbeddingGemma ONNX files
    ├── model.onnx (or model_quantized.onnx)
    └── tokenizer.json
```



## Other CLI Options

The server accepts command-line arguments (overriding environment variables):

| Option | Environment Variable | Default | Description |
| --- | --- | --- | --- |
| `--model-type` | `MODEL_TYPE` | `ettin` | Model hosting mode: `ettin`, `embeddinggemma`, `gemma`, `both`, or `auto` |
| `--model-dir` | `MODEL_DIR` | `./model` | Path to directory containing Ettin model files & tokenizer |
| `--onnx-path` | `ONNX_PATH` | `None` | Direct path to Ettin `model.onnx` file |
| `--embedding-model-dir` | `EMBEDDING_MODEL_DIR` | `None` | Directory containing EmbeddingGemma ONNX model & tokenizer (defaults to `--model-dir` if unassigned) |
| `--embedding-onnx-path` | `EMBEDDING_ONNX_PATH` | `None` | Direct path to EmbeddingGemma `model.onnx` file |
| `--model-name` | `MODEL_NAME` | `cross-encoder/ettin-reranker-150m-v1` | Model name identifier in OpenAI API responses for reranker |
| `--embedding-model-name` | `EMBEDDING_MODEL_NAME` | `google/embeddinggemma-300m` | Model name identifier in OpenAI API responses for EmbeddingGemma |
| `--host` | `HOST` | `0.0.0.0` | IP address to bind server |
| `--port` | `PORT` | `8000` | Port to bind server |
| `--max-length` | `MAX_LENGTH` | `8192` | Maximum token sequence length |
| `--batch-size` | `BATCH_SIZE` | `32` | Batch size for inference |
| `--use-gpu` | `USE_GPU` | `false` | Enable CUDA GPU execution provider if available |
| `--normalize-scores` | `NORMALIZE_SCORES` | `false` | Apply sigmoid score normalization for reranking scores |

---

## API Endpoints

### 1. `/health` or `/ready` (GET)

Checks server initialization status and loaded models.

```bash
curl http://localhost:8000/health

```

### 2. `/v1/models` (GET)

Returns a list of actively loaded models in OpenAI-compatible format.

```bash
curl http://localhost:8000/v1/models

```

### 3. `/v1/embeddings` (POST - OpenAI Compatible)

#### For Dense Vector Embeddings (EmbeddingGemma Mode):

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Artificial intelligence and machine learning.", "Deep learning models."],
    "model": "google/embeddinggemma-300m"
  }'

```

#### Response Example:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.012, -0.045, 0.089, "..."],
      "index": 0
    }
  ],
  "model": "google/embeddinggemma-300m",
  "usage": {
    "prompt_tokens": 14,
    "total_tokens": 14
  }
}
```

### 4. `/v1/rerank` or `/rerank` (POST - Cohere Compatible)

```bash
curl -X POST http://localhost:8000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "Tokyo is the capital of Japan."
    ],
    "top_n": 2
  }'
```

---

## Verification & Test Scripts

* **EmbeddingGemma Verification Test:**
Tests single string embedding, batch processing, L2 unit-norm constraint, and semantic cosine similarity checks:
```bash
python verify_embeddings.py --server-url http://localhost:8000
```


* **Ettin Reranker Logits Verification:**
```bash
python verify_reference.py --server-url http://localhost:8000
```


* **Client Example Integration Suite:**
```bash
python client_example.py
```
