#!/usr/bin/env python3
"""
Verification and test script for the YOLOv8 DocLayNet ONNX document layout analysis endpoint.
Accepts an image file or URL, queries the server, and outputs formatted bounding boxes and region labels.
"""

import sys
import os
import time
import base64
import argparse
import requests
from pathlib import Path


def check_server_health(server_url: str) -> bool:
    """Checks whether the server is reachable and reports loaded models."""
    health_url = f"{server_url.rstrip('/')}/health"
    try:
        resp = requests.get(health_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"[✓] Server connected at {server_url}")
            print(f"    - Engine: {data.get('engine', 'unknown')}")
            print(f"    - DocLayNet Loaded: {data.get('doclaynet_loaded', False)}")
            print(f"    - DocLayNet Model Name: {data.get('doclaynet_model_name', 'N/A')}")
            return True
        else:
            print(f"[!] Server returned HTTP {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"[✗] Could not connect to server at {server_url}: {e}")
        return False


def encode_image(image_path: str) -> str:
    """Reads a local image file and converts it to a base64 data URI string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    ext = path.suffix.lower().lstrip(".")
    mime_type = "jpeg" if ext in ("jpg", "jpeg") else ext if ext in ("png", "webp", "bmp") else "jpeg"

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/{mime_type};base64,{encoded}"


def query_layout_detection(
    server_url: str,
    image_input: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    extract_tables: bool = True,
    use_multipart: bool = False,
) -> dict:
    """Sends an inference request to the layout detection endpoint."""
    endpoint = f"{server_url.rstrip('/')}/v1/vision/layout"

    start_time = time.time()

    if use_multipart:
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"File not found for multipart upload: {image_input}")

        with open(path, "rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            data = {
                "confidence_threshold": str(conf_threshold),
                "iou_threshold": str(iou_threshold),
                "extract_tables": str(extract_tables).lower(),
            }
            resp = requests.post(endpoint, files=files, data=data, timeout=45)
    else:
        # Check if input is a URL or a local file
        if image_input.startswith("http://") or image_input.startswith("https://") or image_input.startswith("data:image"):
            payload_img = image_input
        else:
            payload_img = encode_image(image_input)

        payload = {
            "image": payload_img,
            "confidence_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "extract_tables": extract_tables,
        }
        resp = requests.post(endpoint, json=payload, timeout=45)

    elapsed_ms = (time.time() - start_time) * 1000

    if resp.status_code != 200:
        print(f"[✗] Request failed with HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()
    result["_client_latency_ms"] = round(elapsed_ms, 2)
    return result


def display_results(result: dict, image_source: str, show_html: bool = False):
    """Formats and prints detection results in a clean tabular view."""
    detections = result.get("detections", [])
    if not detections and "results" in result and len(result["results"]) > 0:
        detections = result["results"][0].get("detections", [])

    img_size = result.get("image_size", {})
    if not img_size and "results" in result and len(result["results"]) > 0:
        img_size = {
            "width": result["results"][0].get("width", "N/A"),
            "height": result["results"][0].get("height", "N/A"),
        }

    latency = result.get("_client_latency_ms", 0.0)

    print("\n" + "=" * 70)
    print(f"  DocLayNet Layout Detection Results")
    print("=" * 70)
    print(f" Source:      {image_source}")
    print(f" Image Size:  {img_size.get('width', 'N/A')} x {img_size.get('height', 'N/A')} px")
    print(f" Model:       {result.get('model', 'N/A')}")
    print(f" Detections:  {len(detections)} region(s) found")
    print(f" Latency:     {latency:.2f} ms")
    print("-" * 70)

    if not detections:
        print("  No document layout regions detected above confidence threshold.")
        print("=" * 70 + "\n")
        return

    # Count breakdown by region type
    label_counts = {}
    for d in detections:
        lbl = d.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(" Region Summary:")
    for lbl, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {lbl:<18}: {count}")

    print("-" * 70)
    print(f" {'#':<3} | {'Label':<16} | {'Conf':<7} | {'Bounding Box (x1, y1, x2, y2)':<32}")
    print("-" * 70)

    for idx, d in enumerate(detections, start=1):
        label = d.get("label", "unknown")
        conf = f"{d.get('confidence', 0.0) * 100:.1f}%"
        bbox = d.get("bbox", [0, 0, 0, 0])
        bbox_str = f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
        has_table_data = " [Table Extracted]" if (d.get("markdown") or d.get("html")) else ""
        print(f" {idx:<3} | {label:<16} | {conf:<7} | {bbox_str:<32}{has_table_data}")

    # Display rendered Markdown for each extracted table
    tables = [d for d in detections if d.get("label", "").lower() == "table" and (d.get("markdown") or d.get("html"))]
    if tables:
        print("\n" + "=" * 70)
        print("  Extracted Table Structures (Markdown)")
        print("=" * 70)
        for t_idx, t in enumerate(tables, start=1):
            conf = f"{t.get('confidence', 0.0) * 100:.1f}%"
            bbox = t.get("bbox", [])
            print(f"\n[Table #{t_idx}] Confidence: {conf} | Bounding Box: {bbox}")
            if t.get("markdown"):
                print(t["markdown"])
            elif t.get("html"):
                print(t["html"])
            
            if show_html and t.get("html"):
                print(f"\n--- Raw HTML for Table #{t_idx} ---")
                print(t["html"])
                print("-" * 35)

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test DocLayNet ONNX Document Layout Server"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to local image file or HTTP/HTTPS image URL",
    )
    parser.add_argument(
        "--server-url", "-s",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression (default: 0.45)",
    )
    parser.add_argument(
        "--no-tables",
        action="store_false",
        dest="extract_tables",
        default=True,
        help="Disable automatic table HTML/Markdown conversion for detected tables",
    )
    parser.add_argument(
        "--multipart",
        action="store_true",
        help="Send image via multipart/form-data upload instead of base64 JSON",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip pre-check health endpoint query",
    )
    parser.add_argument(
        "--show-html",
        action="store_true",
        help="Display raw HTML output for extracted tables",
    )

    args = parser.parse_args()

    if not args.skip_health:
        if not check_server_health(args.server_url):
            print("[!] Warning: Server health check failed or DocLayNet is not loaded. Attempting request anyway...")

    print(f"\n[→] Sending inference request for '{args.input}'...")
    res = query_layout_detection(
        server_url=args.server_url,
        image_input=args.input,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        extract_tables=args.extract_tables,
        use_multipart=args.multipart,
    )

    display_results(res, args.input, show_html=args.show_html)


if __name__ == "__main__":
    main()
