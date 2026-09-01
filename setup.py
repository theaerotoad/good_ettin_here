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
# Download Engine with Real-Time Progress Bar & Candidate Fallbacks
# ==============================================================================
def is_valid_file(path: str, min_size_kb: int = 1) -> bool:
    """Verifies that a local file exists and is larger than min_size_kb."""
    if not os.path.isfile(path):
        return False
    return os.path.getsize(path) >= (min_size_kb * 1024)


def download_file_stream(
    url: str,
    dest_path: str,
    desc: str = "",
    optional: bool = False,
    silent_404: bool = False,
) -> bool:
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

    except urllib.error.HTTPError as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if not silent_404 and not optional:
            print(f"\n  {YELLOW}HTTP Error {e.code} for {url}{RESET}")
        return False
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if not silent_404 and not optional:
            print(f"\n  {RED}Download error for {url}: {e}{RESET}")
        return False


def download_file_with_candidates(
    urls: List[str],
    dest_path: str,
    desc: str = "",
    optional: bool = False,
    min_size_kb: int = 1,
) -> bool:
    """Tries a list of candidate URLs in order until one succeeds."""
    if is_valid_file(dest_path, min_size_kb=min_size_kb):
        filename = os.path.basename(dest_path)
        print(f"  {GREEN}Found locally:{RESET} {filename}")
        return True

    for i, url in enumerate(urls):
        is_last = (i == len(urls) - 1)
        silent = (not is_last) or optional
        success = download_file_stream(url, dest_path, desc=desc, optional=optional, silent_404=silent)
        if success:
            return True

    if optional:
        return True
    return False


# ==============================================================================
# Model Existence Validators
# ==============================================================================
def check_reranker_exists(dest_dir: str) -> bool:
    """Checks if valid Ettin reranker ONNX and tokenizer files exist locally."""
    has_onnx = any(
        is_valid_file(os.path.join(dest_dir, f), min_size_kb=1000)
        for f in ["model.onnx", "model_O4.onnx", "onnx/model_O4.onnx", "onnx/model.onnx"]
    )
    has_tok = is_valid_file(os.path.join(dest_dir, "tokenizer.json"), min_size_kb=1)
    return has_onnx and has_tok


def check_embedding_exists(dest_dir: str) -> bool:
    """Checks if valid EmbeddingGemma ONNX (and companion external data) and tokenizer exist locally."""
    has_tok = is_valid_file(os.path.join(dest_dir, "tokenizer.json"), min_size_kb=1)
    if not has_tok:
        return False

    for fname in ["model_quantized.onnx", "model.onnx", "onnx/model_quantized.onnx", "onnx/model.onnx"]:
        onnx_p = os.path.join(dest_dir, fname)
        if is_valid_file(onnx_p, min_size_kb=50):
            # Standalone ONNX (> 50 MB) containing embedded weights
            if os.path.getsize(onnx_p) >= 50 * 1024 * 1024:
                return True
            # ONNX graph using external tensor data file (.onnx_data / .onnx.data)
            data_candidates = [
                f"{onnx_p}_data",
                f"{onnx_p}.data",
                os.path.join(dest_dir, "model_quantized.onnx_data"),
                os.path.join(dest_dir, "model_quantized.onnx.data"),
                os.path.join(dest_dir, "model.onnx_data"),
                os.path.join(dest_dir, "model.onnx.data"),
            ]
            if any(is_valid_file(df, min_size_kb=1000) for df in data_candidates):
                return True
    return False


def check_doclaynet_exists(dest_dir: str, variant: str = "yolov8x") -> bool:
    """Checks if valid DocLayNet YOLOv8 ONNX model exists locally."""
    onnx_file = DOCLAYNET_VARIANTS.get(variant, {}).get("onnx_file", f"{variant}-doclaynet.onnx")
    candidates = [onnx_file, "best.onnx", "model.onnx", f"{variant}.onnx", f"{variant}-doclaynet.onnx"]
    return any(is_valid_file(os.path.join(dest_dir, f), min_size_kb=5000) for f in candidates)


def check_slanet_exists(dest_path_or_dir: str) -> bool:
    """Checks if valid SLANet ONNX table model exists locally."""
    if os.path.isfile(dest_path_or_dir):
        return is_valid_file(dest_path_or_dir, min_size_kb=5000)
    target = os.path.join(dest_path_or_dir, "ch_ppstructure_mobile_v2_SLANet.onnx")
    return is_valid_file(target, min_size_kb=5000)


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

    files_spec = [
        (["config.json"], "config.json", False, 0),
        (["tokenizer.json"], "tokenizer.json", False, 1),
        (["tokenizer_config.json"], "tokenizer_config.json", True, 0),
        (["special_tokens_map.json"], "special_tokens_map.json", True, 0),
        (["onnx/model_O4.onnx", "onnx/model.onnx", "model.onnx", "onnx/model_quantized.onnx"], "model.onnx", False, 1000),
        (["onnx/model_O4.onnx_data", "onnx/model.onnx_data", "model.onnx_data"], "model.onnx_data", True, 1000),
        (["2_Dense/model.safetensors", "dense/model.safetensors"], "2_Dense/model.safetensors", True, 10),
        (["3_LayerNorm/model.safetensors", "layernorm/model.safetensors"], "3_LayerNorm/model.safetensors", True, 1),
        (["4_Dense/model.safetensors", "dense_1/model.safetensors"], "4_Dense/model.safetensors", True, 1),
        (["model.safetensors"], "model.safetensors", True, 10),
    ]

    print(f"\n{CYAN}Target Directory:{RESET} {dest_dir}")
    print(f"{CYAN}Hugging Face Source:{RESET} {repo_id}")

    os.makedirs(dest_dir, exist_ok=True)
    all_ok = True

    for remote_candidates, local_subpath, optional, min_kb in files_spec:
        target_path = os.path.join(dest_dir, local_subpath)
        urls = [f"{base_url}/{p}" for p in remote_candidates]
        ok = download_file_with_candidates(urls, target_path, desc=f"({local_subpath})", optional=optional, min_size_kb=min_kb)
        if not ok and not optional:
            all_ok = False

    return repo_id if all_ok else None


def download_embedding_gemma(dest_dir: str) -> Optional[str]:
    """Downloads EmbeddingGemma ONNX dense vector model files and external weight data."""
    repo_id = "onnx-community/embeddinggemma-300m-ONNX"
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    files_spec = [
        (["config.json"], "config.json", False, 0),
        (["tokenizer.json"], "tokenizer.json", False, 1),
        (["tokenizer_config.json"], "tokenizer_config.json", True, 0),
        (["special_tokens_map.json"], "special_tokens_map.json", True, 0),
        (["onnx/model_quantized.onnx", "onnx/model.onnx", "model_quantized.onnx", "model.onnx"], "model_quantized.onnx", False, 50),
        ([
            "onnx/model_quantized.onnx_data",
            "onnx/model_quantized.onnx.data",
            "model_quantized.onnx_data",
            "onnx/model.onnx_data",
            "onnx/model.onnx.data",
            "model.onnx_data"
        ], "model_quantized.onnx_data", True, 1000),
    ]

    print(f"\n{CYAN}Target Directory:{RESET} {dest_dir}")
    print(f"{CYAN}Hugging Face Source:{RESET} {repo_id}")

    os.makedirs(dest_dir, exist_ok=True)
    all_ok = True

    for remote_candidates, local_subpath, optional, min_kb in files_spec:
        target_path = os.path.join(dest_dir, local_subpath)
        urls = [f"{base_url}/{p}" for p in remote_candidates]
        ok = download_file_with_candidates(urls, target_path, desc=f"({local_subpath})", optional=optional, min_size_kb=min_kb)
        if not ok and not optional:
            all_ok = False

    return "google/embeddinggemma-300m" if all_ok else None


def download_doclaynet(dest_dir: str, variant: str) -> tuple[Optional[str], Optional[str]]:
    """Downloads YOLOv8 DocLayNet layout analysis model and config."""
    meta = DOCLAYNET_VARIANTS.get(variant, DOCLAYNET_VARIANTS["yolov8x"])
    repo_id = meta["repo"]
    onnx_name = meta["onnx_file"]
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    # Try common filename patterns found across YOLO HuggingFace repos
    onnx_candidates = [
        onnx_name,
        "best.onnx",
        "model.onnx",
        f"{variant}.onnx",
        f"{variant}_doclaynet.onnx",
        "weights/best.onnx",
    ]

    files_spec = [
        (["config.json"], "config.json", True, 0),
        (onnx_candidates, onnx_name, False, 5000),
    ]

    print(f"\n{CYAN}Target Directory:{RESET} {dest_dir}")
    print(f"{CYAN}Hugging Face Source:{RESET} {repo_id} ({meta['desc']})")

    os.makedirs(dest_dir, exist_ok=True)
    all_ok = True

    for remote_candidates, local_subpath, optional, min_kb in files_spec:
        target_path = os.path.join(dest_dir, local_subpath)
        urls = [f"{base_url}/{p}" for p in remote_candidates]
        ok = download_file_with_candidates(urls, target_path, desc=f"({local_subpath})", optional=optional, min_size_kb=min_kb)
        if not ok and not optional:
            all_ok = False

    return (meta["name"], repo_id) if all_ok else (None, None)


def download_slanet(dest_dir: str) -> Optional[str]:
    """Downloads the standalone SLANet ONNX table structure recognizer."""
    filename = "ch_ppstructure_mobile_v2_SLANet.onnx"
    target_path = os.path.join(dest_dir, filename)

    print(f"\n{CYAN}Target Path:{RESET} {target_path}")
    if is_valid_file(target_path, min_size_kb=5000):
        print(f"  {GREEN}Found locally:{RESET} {filename}")
        return target_path

    os.makedirs(dest_dir, exist_ok=True)
    urls = [
        "https://huggingface.co/SWHL/RapidStructure/resolve/main/table/ch_ppstructure_mobile_v2_SLANet.onnx",
        "https://github.com/RapidAI/RapidTable/releases/download/v0.0.1/ch_ppstructure_mobile_v2_SLANet.onnx",
    ]
    ok = download_file_with_candidates(urls, target_path, desc="(SLANet ~7.3MB)", optional=False, min_size_kb=5000)
    return target_path if ok else None


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
    r_dir = Path(reranker_dir).as_posix() if reranker_dir else "./ettinreranker_model"
    e_dir = Path(embedding_dir).as_posix() if embedding_dir else "./embeddinggemma_model"
    d_dir = Path(doclaynet_dir).as_posix() if doclaynet_dir else "./doclaynet_model"
    s_path = f'"{Path(slanet_path).as_posix()}"' if slanet_path else "null"

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
        f'  model_dir: "{r_dir}"',
        "  onnx_path: null",
        f'  model_name: "{reranker_model_name if reranker_model_name else "cross-encoder/ettin-reranker-150m-v1"}"',
        "  max_length: 8192",
        "  batch_size: 32",
        "",
        "# ------------------------------------------------------------------------------",
        "# Dense Vector Embeddings (EmbeddingGemma) Configuration",
        "# ------------------------------------------------------------------------------",
        "embedding:",
        f'  model_dir: "{e_dir}"',
        f'  model_name: "{embedding_model_name if embedding_model_name else "google/embeddinggemma-300m"}"',
        "  max_length: 2048",
        "  batch_size: 32",
        "",
        "# ------------------------------------------------------------------------------",
        "# Vision & Layout Analysis (DocLayNet YOLOv8 & SLANet Table Recognition)",
        "# ------------------------------------------------------------------------------",
        "vision:",
        "  layout:",
        f'    model_dir: "{d_dir}"',
        f'    model_name: "{doclaynet_model_name if doclaynet_model_name else "yolov8x-doclaynet"}"',
        "    conf_threshold: 0.25",
        "    iou_threshold: 0.45",
        "    image_size: 640",
        "  table:",
        "    enable: true",
        f'    model_path: {s_path}',
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
        "reranker_dir": f"./ettinreranker_model_{args.reranker_size}",
        "reranker_name": f"cross-encoder/ettin-reranker-{args.reranker_size}-v1",
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

    size = args.reranker_size if args.yes else prompt_choice("Select model size", ETTIN_SIZES, default="150m")
    default_r_dir = f"./ettinreranker_model_{size}"
    target_dir = default_r_dir if args.yes else prompt_input("Target directory", default_r_dir)
    cfg_state["reranker_dir"] = target_dir

    exists_r = check_reranker_exists(target_dir)
    if exists_r:
        print(f"  {GREEN}Existing Ettin Reranker ({size}) weights found in {target_dir}.{RESET}")
        dl_reranker = False if args.yes else prompt_yes_no(f"Re-download / update Ettin Reranker ({size}) weights?", default=False)
    else:
        dl_reranker = True if args.yes else prompt_yes_no(f"Download Ettin Reranker ({size}) weights?", default=True)

    if dl_reranker:
        repo_id = download_ettin_reranker(target_dir, size)
        cfg_state["reranker_name"] = repo_id or f"cross-encoder/ettin-reranker-{size}-v1"
    else:
        cfg_state["reranker_name"] = f"cross-encoder/ettin-reranker-{size}-v1"
        if not exists_r:
            print(f"  {YELLOW}Skipping Ettin Reranker download.{RESET}\n")
        else:
            print()

    # --------------------------------------------------------------------------
    # 2. EmbeddingGemma Dense Vector Embeddings
    # --------------------------------------------------------------------------
    print(f"{BOLD}2. EmbeddingGemma ONNX Embeddings (300M){RESET}")
    print(f"{DIM}Dense vector embedder providing OpenAI-compatible /v1/embeddings.{RESET}")

    default_e_dir = "./embeddinggemma_model"
    target_dir = default_e_dir if args.yes else prompt_input("Target directory", default_e_dir)
    cfg_state["embedding_dir"] = target_dir

    exists_e = check_embedding_exists(target_dir)
    if exists_e:
        print(f"  {GREEN}Existing EmbeddingGemma weights found in {target_dir}.{RESET}")
        dl_embed = False if args.yes else prompt_yes_no("Re-download / update EmbeddingGemma weights?", default=False)
    else:
        dl_embed = True if args.yes else prompt_yes_no("Download EmbeddingGemma ONNX weights?", default=True)

    if dl_embed:
        emb_name = download_embedding_gemma(target_dir)
        cfg_state["embedding_name"] = emb_name or "google/embeddinggemma-300m"
    else:
        cfg_state["embedding_name"] = "google/embeddinggemma-300m"
        if not exists_e:
            print(f"  {YELLOW}Skipping EmbeddingGemma download.{RESET}\n")
        else:
            print()

    # --------------------------------------------------------------------------
    # 3. YOLOv8 DocLayNet Document Layout Analysis
    # --------------------------------------------------------------------------
    print(f"{BOLD}3. YOLOv8 DocLayNet Document Layout Analysis{RESET}")
    print(f"{DIM}Parses PDF/Image regions: Text, Titles, Tables, Headers, Footers, Pictures.{RESET}")

    default_d_dir = "./doclaynet_model"
    target_dir = default_d_dir if args.yes else prompt_input("Target directory", default_d_dir)
    cfg_state["doclaynet_dir"] = target_dir

    exists_d = check_doclaynet_exists(target_dir)
    if exists_d:
        print(f"  {GREEN}Existing DocLayNet model found in {target_dir}.{RESET}")
        dl_doclaynet = False if args.yes else prompt_yes_no("Re-download / update DocLayNet model weights?", default=False)
    else:
        dl_doclaynet = True if args.yes else prompt_yes_no("Download YOLOv8 DocLayNet model weights?", default=True)

    if dl_doclaynet:
        variant = "yolov8x" if args.yes else prompt_choice(
            "Select YOLOv8 model variant (yolov8n=fast/light, yolov8x=highest accuracy)",
            list(DOCLAYNET_VARIANTS.keys()),
            default="yolov8x",
        )
        doc_name, _ = download_doclaynet(target_dir, variant)
        cfg_state["doclaynet_name"] = doc_name or DOCLAYNET_VARIANTS[variant]["name"]
    else:
        cfg_state["doclaynet_name"] = "yolov8x-doclaynet"
        if not exists_d:
            print(f"  {YELLOW}Skipping DocLayNet download.{RESET}\n")
        else:
            print()

    # --------------------------------------------------------------------------
    # 4. SLANet Table Structure Recognition
    # --------------------------------------------------------------------------
    print(f"{BOLD}4. SLANet Neural Table Recognizer{RESET}")
    print(f"{DIM}Converts detected table image crops into clean HTML and Markdown tables.{RESET}")

    default_t_dir = "./table_model"
    target_dir = default_t_dir if args.yes else prompt_input("Target directory", default_t_dir)
    expected_file = os.path.join(target_dir, "ch_ppstructure_mobile_v2_SLANet.onnx")
    cfg_state["slanet_path"] = expected_file

    exists_s = check_slanet_exists(target_dir)
    if exists_s:
        print(f"  {GREEN}Existing SLANet model found in {target_dir}.{RESET}")
        dl_slanet = False if args.yes else prompt_yes_no("Re-download / update SLANet ONNX table weights (~7.3MB)?", default=False)
    else:
        dl_slanet = True if args.yes else prompt_yes_no("Download SLANet ONNX table weights (~7.3MB)?", default=True)

    if dl_slanet:
        slanet_file = download_slanet(target_dir)
        cfg_state["slanet_path"] = slanet_file or expected_file
    else:
        if not exists_s:
            print(f"  {YELLOW}Skipping SLANet download.{RESET}\n")
        else:
            print()

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
