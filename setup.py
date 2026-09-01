#!/usr/bin/env python3
"""
Interactive Setup and Model Provisioning Script for Ettin ONNX Server.

Downloads required ONNX model weights (Ettin Rerankers 17M-1B, EmbeddingGemma,
YOLOv8 DocLayNet, and SLANet) and generates a tailored config.yaml.
"""

import os
import sys
import shutil
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, List, Dict, Any

# ==============================================================================
# Terminal Color & UI Helpers
# ==============================================================================
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner():
    print(f"\n{CYAN}{BOLD}================================================================{RESET}")
    print(f"{CYAN}{BOLD}     Ettin ONNX Server - Model Setup & Configuration Wizard     {RESET}")
    print(f"{CYAN}{BOLD}================================================================{RESET}\n")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompts the user for a Yes/No answer with a specified default."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        try:
            choice = input(f"{BOLD}{question}{RESET}{suffix}").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nOperation cancelled by user.")
            sys.exit(0)

        if not choice:
            return default
        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        print(f"{YELLOW}Please enter 'y' for yes or 'n' for no.{RESET}")


def prompt_choice(question: str, choices: List[str], default: str) -> str:
    """Prompts the user to select from a list of options."""
    choices_str = "/".join(choices)
    while True:
        try:
            resp = input(f"{BOLD}{question}{RESET} ({choices_str}) [default: {default}]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nOperation cancelled by user.")
            sys.exit(0)

        if not resp:
            return default
        if resp in [c.lower() for c in choices]:
            return resp
        print(f"{YELLOW}Invalid option. Please choose one of: {', '.join(choices)}{RESET}")


def prompt_input(question: str, default: str) -> str:
    """Prompts the user for arbitrary string input with a default."""
    try:
        resp = input(f"{BOLD}{question}{RESET} [default: {default}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    return resp if resp else default


# ==============================================================================
# Download Engine with Real-Time Progress Bar
# ==============================================================================
def download_file_stream(url: str, dest_path: str, desc: str = "") -> bool:
    """Downloads a file from a URL to dest_path with progress bar and LFS validation."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = f"{dest_path}.tmp"

    headers = {"User-Agent": "Mozilla/5.0 (Ettin-ONNX-Setup-Tool)"}
    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=45) as response, open(temp_path, "wb") as out_file:
            total_size = response.getheader("Content-Length")
            total_bytes = int(total_size) if total_size else 0
            downloaded = 0
            block_size = 64 * 1024  # 64 KB chunks

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                downloaded += len(buffer)

                if total_bytes > 0:
                    percent = downloaded / total_bytes * 100
                    mb_curr = downloaded / (1024 * 1024)
                    mb_total = total_bytes / (1024 * 1024)
                    bar_len = 30
                    filled = int(bar_len * downloaded // total_bytes)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    status_line = f"\r  {CYAN}[{bar}]{RESET} {percent:5.1f}% ({mb_curr:5.1f}/{mb_total:5.1f} MB) {desc}"
                else:
                    mb_curr = downloaded / (1024 * 1024)
                    status_line = f"\r  {CYAN}[Downloading]{RESET} {mb_curr:5.1f} MB downloaded {desc}"

                sys.stdout.write(status_line)
                sys.stdout.flush()

        print()  # Finalize newline

        # Validate git-lfs text pointer artifact check
        if os.path.exists(temp_path) and os.path.getsize(temp_path) < 1024:
            with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(128)
                if "git-lfs" in content or "oid sha256" in content:
                    print(f"  {RED}Error: Received a Git-LFS text pointer rather than binary payload.{RESET}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return False

        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        return True

    except Exception as e:
        print(f"\n  {RED}Download error for {url}: {e}{RESET}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def is_valid_file(path: str, min_size_kb: int = 1) -> bool:
    """Verifies that a local file exists and is larger than min_size_kb."""
    if not os.path.isfile(path):
        return False
    return os.path.getsize(path) >= (min_size_kb * 1024)


# ==============================================================================
# Model Manifests & Retrieval Handlers
# ==============================================================================
ETTIN_SIZES = ["17m", "32m", "68m", "150m", "400m", "1b"]

DOCLAYNET_VARIANTS = {
    "yolov8n": {
        "repo": "Oblix/yolov8n-doclaynet_ONNX",
        "onnx_file": "yolov8n-doclaynet.onnx",
        "name": "yolov8n-doclaynet",
        "desc": "Nano model (~13 MB) - ultra fast CPU inference",
    },
    "yolov8s": {
        "repo": "Oblix/yolov8s-doclaynet_ONNX",
        "onnx_file": "yolov8s-doclaynet.onnx",
        "name": "yolov8s-doclaynet",
        "desc": "Small model (~44 MB) - balanced speed & accuracy",
    },
    "yolov8m": {
        "repo": "Oblix/yolov8m-doclaynet_ONNX",
        "onnx_file": "yolov8m-doclaynet.onnx",
        "name": "yolov8m-doclaynet",
        "desc": "Medium model (~100 MB) - higher visual detail",
    },
    "yolov8x": {
        "repo": "Oblix/yolov8x-doclaynet_ONNX",
        "onnx_file": "yolov8x-doclaynet.onnx",
        "name": "yolov8x-doclaynet",
        "desc": "Extra Large model (~270 MB) - maximum bounding box precision (Recommended)",
    },
}


def download_ettin_reranker(dest_dir: str, size: str) -> Optional[str]:
    """Downloads Ettin Cross-Encoder model, tokenizer, and classification head weights."""
    repo_id = f"cross-encoder/ettin-reranker-{size}-v1"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    files_to_fetch = [
        ("config.json", "config.json"),
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
        ("onnx/model_O4.onnx", "model.onnx"),
        ("2_Dense/model.safetensors", "2_Dense/model.safetensors"),
        ("3_LayerNorm/model.safetensors", "3_LayerNorm/model.safetensors"),
        ("4_Dense/model.safetensors", "4_Dense/model.safetensors"),
    ]

    print(f"\n{CYAN}Target Directory:{RESET} {dest_dir}")
    print(f"{CYAN}Hugging Face Source:{RESET} {repo_id}")

    os.makedirs(dest_dir, exist_ok=True)
    all_ok = True

    for remote_subpath, local_subpath in files_to_fetch:
        target_path = os.path.join(dest_dir, local_subpath)
        if is_valid_file(target_path):
            print(f"  {GREEN}Found locally:{RESET} {local_subpath}")
            continue

        url = f"{base_url}/{remote_subpath}"
        # Fallback for ONNX file name if model_O4 is not available
        success = download_file_stream(url, target_path, desc=f"({local_subpath})")
        if not success and remote_subpath == "onnx/model_O4.onnx":
            fallback_url = f"{base_url}/onnx/model.onnx"
            print(f"  {YELLOW}model_O4.onnx not found, trying fallback: model.onnx...{RESET}")
            success = download_file_stream(fallback_url, target_path, desc="(model.onnx)")

        if not success:
            all_ok = False

    return repo_id if all_ok else None


def download_embedding_gemma(dest_dir: str) -> Optional[str]:
    """Downloads EmbeddingGemma ONNX dense vector model files."""
    repo_id = "onnx-community/embeddinggemma-300m-ONNX"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    files_to_fetch = [
        ("config.json", "config.json"),
        ("tokenizer.json", "tokenizer.json"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("special_tokens_map.json", "special_tokens_map.json"),
        ("onnx/model_quantized.onnx", "model_quantized.onnx"),
    ]

    print(f"\n{CYAN}Target Directory:{RESET} {dest_dir}")
    print(f"{CYAN}Hugging Face Source:{RESET} {repo_id}")

    os.makedirs(dest_dir, exist_ok=True)
    all_ok = True

    for remote_subpath, local_subpath in files_to_fetch:
        target_path = os.path.join(dest_dir, local_subpath)
        if is_valid_file(target_path):
            print(f"  {GREEN}Found locally:{RESET} {local_subpath}")
            continue

        url = f"{base_url}/{remote_subpath}"
        success = download_file_stream(url, target_path, desc=f"({local_subpath})")
        if not success and remote_subpath == "onnx/model_quantized.onnx":
            fallback_url = f"{base_url}/onnx/model.onnx"
            print(f"  {YELLOW}model_quantized.onnx not found, trying fallback: model.onnx...{RESET}")
            target_fallback = os.path.join(dest_dir, "model.onnx")
            success = download_file_stream(fallback_url, target_fallback, desc="(model.onnx)")

        if not success:
            all_ok = False

    return "google/embeddinggemma-300m" if all_ok else None


def download_doclaynet(dest_dir: str, variant: str) -> tuple[Optional[str], Optional[str]]:
    """Downloads YOLOv8 DocLayNet layout analysis model and config."""
    meta = DOCLAYNET_VARIANTS.get(variant, DOCLAYNET_VARIANTS["yolov8x"])
    repo_id = meta["repo"]
    onnx_name = meta["onnx_file"]
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    files_to_fetch = [
        ("config.json", "config.json"),
        (onnx_name, onnx_name),
    ]

    print(f"\n{CYAN}Target Directory:{RESET} {dest_dir}")
    print(f"{CYAN}Hugging Face Source:{RESET} {repo_id} ({meta['desc']})")

    os.makedirs(dest_dir, exist_ok=True)
    all_ok = True

    for remote_subpath, local_subpath in files_to_fetch:
        target_path = os.path.join(dest_dir, local_subpath)
        if is_valid_file(target_path):
            print(f"  {GREEN}Found locally:{RESET} {local_subpath}")
            continue

        url = f"{base_url}/{remote_subpath}"
        success = download_file_stream(url, target_path, desc=f"({local_subpath})")
        if not success:
            all_ok = False

    return (meta["name"], repo_id) if all_ok else (None, None)


def download_slanet(dest_dir: str) -> Optional[str]:
    """Downloads the standalone SLANet ONNX table structure recognizer."""
    url = "https://huggingface.co/SWHL/RapidStructure/resolve/main/table/ch_ppstructure_mobile_v2_SLANet.onnx"
    filename = "ch_ppstructure_mobile_v2_SLANet.onnx"
    target_path = os.path.join(dest_dir, filename)

    print(f"\n{CYAN}Target Path:{RESET} {target_path}")
    if is_valid_file(target_path, min_size_kb=5000):
        print(f"  {GREEN}Found locally:{RESET} {filename}")
        return target_path

    os.makedirs(dest_dir, exist_ok=True)
    success = download_file_stream(url, target_path, desc="(SLANet ~7.3MB)")
    return target_path if success else None


# ==============================================================================
# YAML Config Generation Engine
# ==============================================================================
def generate_yaml_config(
    output_path: str,
    server_host: str,
    server_port: int,
    use_gpu: bool,
    model_type: str,
    reranker_dir: Optional[str],
    reranker_model_name: Optional[str],
    embedding_dir: Optional[str],
    embedding_model_name: Optional[str],
    doclaynet_dir: Optional[str],
    doclaynet_model_name: Optional[str],
    slanet_path: Optional[str],
) -> None:
    """Generates the formatted config.yaml with detailed comments."""
    yaml_lines = [
        "# ==============================================================================",
        "# Ettin ONNX Reranker, EmbeddingGemma & Vision Server Configuration",
        "# Generated automatically by setup.py",
        "# ==============================================================================",
        "",
        "# ------------------------------------------------------------------------------",
        "# Server Network & Execution Settings",
        "# ------------------------------------------------------------------------------",
        "server:",
        f'  host: "{server_host}"',
        f"  port: {server_port}",
        f"  use_gpu: {'true' if use_gpu else 'false'}",
        "",
        "# ------------------------------------------------------------------------------",
        "# Active Model Selector",
        "# Options: 'ettin', 'embeddinggemma', 'doclaynet', 'both', 'all', or 'auto'",
        "# ------------------------------------------------------------------------------",
        f'model_type: "{model_type}"',
        "",
        "# ------------------------------------------------------------------------------",
        "# Reranker (Ettin Cross-Encoder) Configuration",
        "# ------------------------------------------------------------------------------",
        "reranker:",
        f'  model_dir: "{reranker_dir if reranker_dir else "./ettinreranker_model"}"',
        "  onnx_path: null",
        f'  model_name: "{reranker_model_name if reranker_model_name else "cross-encoder/ettin-reranker-150m-v1"}"',
        "  max_length: 8192",
        "  batch_size: 32",
        "",
        "# ------------------------------------------------------------------------------",
        "# Dense Vector Embeddings (EmbeddingGemma) Configuration",
        "# ------------------------------------------------------------------------------",
        "embedding:",
        f'  model_dir: "{embedding_dir if embedding_dir else "./embeddinggemma_model"}"',
        f'  model_name: "{embedding_model_name if embedding_model_name else "google/embeddinggemma-300m"}"',
        "  max_length: 2048",
        "  batch_size: 32",
        "",
        "# ------------------------------------------------------------------------------",
        "# Vision & Layout Analysis (DocLayNet YOLOv8 & SLANet Table Recognition)",
        "# ------------------------------------------------------------------------------",
        "vision:",
        "  layout:",
        f'    model_dir: "{doclaynet_dir if doclaynet_dir else "./doclaynet_model"}"',
        f'    model_name: "{doclaynet_model_name if doclaynet_model_name else "yolov8x-doclaynet"}"',
        "    conf_threshold: 0.25",
        "    iou_threshold: 0.45",
        "    image_size: 640",
        "  table:",
        "    enable: true",
        f'    model_path: {"\"" + slanet_path + "\"" if slanet_path else "null"}',
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))

    print(f"\n{GREEN}{BOLD}Configuration successfully generated at:{RESET} {os.path.abspath(output_path)}")


# ==============================================================================
# Main Interactive Wizard Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Ettin ONNX Server Setup Wizard & Model Downloader")
    parser.add_argument("--yes", "-y", action="store_true", help="Accept default choices for all prompts")
    parser.add_argument("--config-out", default="config.yaml", help="Destination path for generated YAML config")
    parser.add_argument("--reranker-size", choices=ETTIN_SIZES, default="150m", help="Default Ettin model size")
    args = parser.parse_args()

    print_banner()

    cfg_state = {
        "reranker_dir": "./ettinreranker_model",
        "reranker_name": None,
        "embedding_dir": "./embeddinggemma_model",
        "embedding_name": None,
        "doclaynet_dir": "./doclaynet_model",
        "doclaynet_name": None,
        "slanet_path": "./table_model/ch_ppstructure_mobile_v2_SLANet.onnx",
    }

    # --------------------------------------------------------------------------
    # 1. Ettin Cross-Encoder Reranker
    # --------------------------------------------------------------------------
    print(f"{BOLD}1. Ettin Cross-Encoder Reranker{RESET}")
    print(f"{DIM}Select from ultra-lightweight (17M, 32M, 68M) to flagship (150M, 400M, 1B) ONNX models.{RESET}")

    dl_reranker = args.yes or prompt_yes_no("Download Ettin Reranker weights?", default=True)
    if dl_reranker:
        size = args.reranker_size if args.yes else prompt_choice("Select model size", ETTIN_SIZES, default="150m")
        target_dir = "./ettinreranker_model" if args.yes else prompt_input("Target directory", "./ettinreranker_model")
        cfg_state["reranker_dir"] = target_dir
        repo_id = download_ettin_reranker(target_dir, size)
        cfg_state["reranker_name"] = repo_id or f"cross-encoder/ettin-reranker-{size}-v1"
    else:
        print(f"  {YELLOW}Skipping Ettin Reranker download.{RESET}\n")

    # --------------------------------------------------------------------------
    # 2. EmbeddingGemma Dense Vector Embeddings
    # --------------------------------------------------------------------------
    print(f"\n{BOLD}2. EmbeddingGemma ONNX Embeddings (300M){RESET}")
    print(f"{DIM}Dense vector embedder providing OpenAI-compatible /v1/embeddings.{RESET}")

    dl_embed = args.yes or prompt_yes_no("Download EmbeddingGemma ONNX weights?", default=True)
    if dl_embed:
        target_dir = "./embeddinggemma_model" if args.yes else prompt_input("Target directory", "./embeddinggemma_model")
        cfg_state["embedding_dir"] = target_dir
        emb_name = download_embedding_gemma(target_dir)
        cfg_state["embedding_name"] = emb_name or "google/embeddinggemma-300m"
    else:
        print(f"  {YELLOW}Skipping EmbeddingGemma download.{RESET}\n")

    # --------------------------------------------------------------------------
    # 3. YOLOv8 DocLayNet Document Layout Analysis
    # --------------------------------------------------------------------------
    print(f"\n{BOLD}3. YOLOv8 DocLayNet Document Layout Analysis{RESET}")
    print(f"{DIM}Parses PDF/Image regions: Text, Titles, Tables, Headers, Footers, Pictures.{RESET}")

    dl_doclaynet = args.yes or prompt_yes_no("Download YOLOv8 DocLayNet model weights?", default=True)
    if dl_doclaynet:
        variant = "yolov8x" if args.yes else prompt_choice(
            "Select YOLOv8 model variant (yolov8n=fast/light, yolov8x=highest accuracy)",
            list(DOCLAYNET_VARIANTS.keys()),
            default="yolov8x",
        )
        target_dir = "./doclaynet_model" if args.yes else prompt_input("Target directory", "./doclaynet_model")
        cfg_state["doclaynet_dir"] = target_dir
        doc_name, _ = download_doclaynet(target_dir, variant)
        cfg_state["doclaynet_name"] = doc_name or DOCLAYNET_VARIANTS[variant]["name"]
    else:
        print(f"  {YELLOW}Skipping DocLayNet download.{RESET}\n")

    # --------------------------------------------------------------------------
    # 4. SLANet Table Structure Recognition
    # --------------------------------------------------------------------------
    print(f"\n{BOLD}4. SLANet Neural Table Recognizer{RESET}")
    print(f"{DIM}Converts detected table image crops into clean HTML and Markdown tables.{RESET}")

    dl_slanet = args.yes or prompt_yes_no("Download SLANet ONNX table weights (~7.3MB)?", default=True)
    if dl_slanet:
        target_dir = "./table_model" if args.yes else prompt_input("Target directory", "./table_model")
        slanet_file = download_slanet(target_dir)
        cfg_state["slanet_path"] = slanet_file or os.path.join(target_dir, "ch_ppstructure_mobile_v2_SLANet.onnx")
    else:
        print(f"  {YELLOW}Skipping SLANet download.{RESET}\n")

    # --------------------------------------------------------------------------
    # 5. Server Settings & config.yaml Generation
    # --------------------------------------------------------------------------
    print(f"\n{BOLD}5. Server & Execution Settings{RESET}")
    host = "0.0.0.0" if args.yes else prompt_input("Server Host IP", "0.0.0.0")
    port_str = "8000" if args.yes else prompt_input("Server Port", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    use_gpu = False if args.yes else prompt_yes_no("Enable CUDA GPU execution provider if available?", default=False)
    config_dest = args.config_out if args.yes else prompt_input("Config output destination path", args.config_out)

    generate_yaml_config(
        output_path=config_dest,
        server_host=host,
        server_port=port,
        use_gpu=use_gpu,
        model_type="auto",
        reranker_dir=cfg_state["reranker_dir"],
        reranker_model_name=cfg_state["reranker_name"],
        embedding_dir=cfg_state["embedding_dir"],
        embedding_model_name=cfg_state["embedding_name"],
        doclaynet_dir=cfg_state["doclaynet_dir"],
        doclaynet_model_name=cfg_state["doclaynet_name"],
        slanet_path=cfg_state["slanet_path"],
    )

    print(f"\n{GREEN}{BOLD}Setup Completed!{RESET}")
    print(f"To launch the server with your generated configuration:\n")
    print(f"  {CYAN}python app/server.py --config {config_dest}{RESET}\n")


if __name__ == "__main__":
    main()
