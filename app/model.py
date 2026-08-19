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

logger = logging.getLogger("ettin-reranker")
logging.basicConfig(level=logging.INFO)


try:
    from scipy.special import erf

    def _gelu_numpy(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
except ImportError:

    def _gelu_numpy(x: np.ndarray) -> np.ndarray:
        """GELU activation in pure NumPy using tanh approximation."""
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def html_table_to_markdown(html_content: str) -> str:
    """Converts HTML table markup into GitHub-Flavored Markdown (GFM) format."""
    if not html_content or "<table" not in html_content.lower():
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return ""

        md_tables = []
        for table in tables:
            rows = table.find_all("tr")
            if not rows:
                continue

            grid_dict = {}
            r = 0
            for row in rows:
                c = 0
                cells = row.find_all(["th", "td"])
                for cell in cells:
                    # Skip cells that are already filled by a previous rowspan/colspan
                    while (r, c) in grid_dict:
                        c += 1
                    
                    text = cell.get_text(separator=" ", strip=True)
                    text = text.replace("|", "\\|").replace("\n", " ").strip()
                    
                    try:
                        colspan = int(cell.get("colspan", 1))
                    except (ValueError, TypeError):
                        colspan = 1
                        
                    try:
                        rowspan = int(cell.get("rowspan", 1))
                    except (ValueError, TypeError):
                        rowspan = 1
                        
                    # Populate the grid coordinates covered by this cell
                    for i in range(rowspan):
                        for j in range(colspan):
                            grid_dict[(r + i, c + j)] = text
                    
                    c += colspan
                r += 1

            if not grid_dict:
                continue

            max_r = max(k[0] for k in grid_dict.keys())
            max_c = max(k[1] for k in grid_dict.keys())
            max_cols = max_c + 1

            grid = []
            for i in range(max_r + 1):
                row_data = [grid_dict.get((i, j), "") for j in range(max_cols)]
                grid.append(row_data)

            col_widths = [3] * max_cols
            for row in grid:
                for col_idx, cell in enumerate(row):
                    col_widths[col_idx] = max(col_widths[col_idx], len(cell))

            lines = []
            header = grid[0]
            lines.append("| " + " | ".join(header[i].ljust(col_widths[i]) for i in range(max_cols)) + " |")
            lines.append("| " + " | ".join("-" * col_widths[i] for i in range(max_cols)) + " |")
            for row in grid[1:]:
                lines.append("| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(max_cols)) + " |")

            md_tables.append("\n".join(lines))

        return "\n\n".join(md_tables)
    except Exception as e:
        logger.warning(f"Failed to convert HTML table to Markdown: {e}")
        return ""


class TableRecognizerONNX:
    """
    ONNX-based Table Structure Recognizer using RapidTable (SLANet) and RapidOCR.
    Combines SLANet cell bounding boxes with OCR tokens to construct HTML & Markdown tables.
    """

    def __init__(self, table_model_path: str = None, use_gpu: bool = False):
        self.table_engine = None
        self.ocr_engine = None
        self._init_engines(table_model_path, use_gpu)

    def _init_engines(self, table_model_path: str = None, use_gpu: bool = False):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            try:
                from rapidocr import RapidOCR
            except ImportError:
                RapidOCR = None

        try:
            from rapid_table import RapidTable
        except ImportError:
            RapidTable = None

        if RapidTable is None:
            logger.warning(
                "rapid-table not installed. Table structure recognition disabled. "
                "Install via: pip install rapid-table rapidocr-onnxruntime beautifulsoup4"
            )
            return

        if RapidOCR is not None:
            try:
                ocr_params = {}
                if use_gpu:
                    ocr_params["use_cuda"] = True
                self.ocr_engine = RapidOCR(**ocr_params)
                logger.info("Initialized RapidOCR engine for table text recognition.")
            except Exception as e:
                logger.warning(f"Could not initialize RapidOCR: {e}")
                self.ocr_engine = None

        try:
            table_params = {}
            if table_model_path and os.path.exists(table_model_path):
                table_params["model_path"] = table_model_path
            if use_gpu:
                table_params["use_cuda"] = True
            self.table_engine = RapidTable(**table_params)
            logger.info("Initialized RapidTable SLANet engine for table structure extraction.")
        except Exception as e:
            logger.warning(f"Could not initialize RapidTable: {e}")
            self.table_engine = None

    def _extract_ocr_tokens(self, img_bgr: np.ndarray) -> list[dict]:
        """Runs OCR on image crop and returns normalized text boxes with centers and coordinates."""
        ocr_tokens = []
        if self.ocr_engine is None:
            return ocr_tokens

        try:
            ocr_out = self.ocr_engine(img_bgr)
            raw_boxes = ocr_out[0] if isinstance(ocr_out, (list, tuple)) and len(ocr_out) > 0 else ocr_out
            if not raw_boxes or not isinstance(raw_boxes, list):
                return ocr_tokens

            for item in raw_boxes:
                if not item or len(item) < 2:
                    continue

                box_pts = item[0]
                text_val = item[1]
                if isinstance(text_val, (list, tuple)):
                    text_str = str(text_val[0])
                else:
                    text_str = str(text_val)

                if not text_str.strip():
                    continue

                pts = np.asarray(box_pts)
                if pts.ndim == 2 and len(pts) >= 4:
                    x1 = float(np.min(pts[:, 0]))
                    y1 = float(np.min(pts[:, 1]))
                    x2 = float(np.max(pts[:, 0]))
                    y2 = float(np.max(pts[:, 1]))
                elif len(box_pts) == 4 and not isinstance(box_pts[0], (list, tuple, np.ndarray)):
                    x1, y1, x2, y2 = float(box_pts[0]), float(box_pts[1]), float(box_pts[2]), float(box_pts[3])
                else:
                    continue

                ocr_tokens.append({
                    "bbox": [x1, y1, x2, y2],
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "text": text_str.strip(),
                })
        except Exception as e:
            logger.warning(f"OCR step failed on table crop: {e}")

        return ocr_tokens

    @staticmethod
    def _normalize_box(b) -> list[float]:
        """
        Converts varied bounding box formats (polygon points (4, 2), 8-point coordinates,
        or 4-point coordinates) into standardized [x1, y1, x2, y2].
        """
        if b is None:
            return None
        try:
            # If b is a tuple/list wrapping box and confidence (e.g. [box, score])
            if (
                isinstance(b, (list, tuple))
                and len(b) == 2
                and isinstance(b[1], (int, float, np.floating, np.integer))
                and not isinstance(b[0], (int, float, np.floating, np.integer))
            ):
                b = b[0]

            arr = np.asarray(b, dtype=np.float32)
            if arr.size == 0:
                return None
            arr = np.squeeze(arr)

            # Case A: Polygon with shape (N, 2) e.g. [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            if arr.ndim == 2 and arr.shape[-1] == 2:
                x1 = float(np.min(arr[:, 0]))
                y1 = float(np.min(arr[:, 1]))
                x2 = float(np.max(arr[:, 0]))
                y2 = float(np.max(arr[:, 1]))
                return [x1, y1, x2, y2]

            # Case B: 1D array
            if arr.ndim == 1:
                if len(arr) == 4:
                    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
                if len(arr) == 8:
                    x1 = float(np.min(arr[0::2]))
                    y1 = float(np.min(arr[1::2]))
                    x2 = float(np.max(arr[0::2]))
                    y2 = float(np.max(arr[1::2]))
                    return [x1, y1, x2, y2]
                if len(arr) >= 4:
                    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]
        except Exception:
            pass
        return None

    def _extract_slanet_structure(self, img_bgr: np.ndarray) -> tuple[list, list]:
        """Extracts HTML tags and cell bounding boxes using SLANet ONNX engine."""
        pred_structures = []
        pred_bboxes = []

        if self.table_engine is None:
            return pred_structures, pred_bboxes

        raw_struct = None

        # 1. Try to bypass RapidTable's internal OCR wrapper by accessing the core SLANet model directly
        for attr in ["table_model", "table_structure", "structure_model", "model"]:
            if hasattr(self.table_engine, attr):
                internal_model = getattr(self.table_engine, attr)
                if callable(internal_model):
                    try:
                        raw_struct = internal_model(img_bgr)
                        if raw_struct is not None:
                            break
                    except Exception as e:
                        logger.warning(f"Internal model '{attr}' call failed: {e}")

        # 2. Fallback to direct call on table_engine
        if raw_struct is None:
            try:
                raw_struct = self.table_engine(img_bgr)
            except Exception as e:
                logger.warning(f"SLANet direct call failed: {e}")

        if raw_struct is None:
            return pred_structures, pred_bboxes

        # 3. Robust unpacking: Find structures (strings/tags) and bboxes (coords) dynamically
        candidates = []
        if isinstance(raw_struct, (list, tuple)):
            for item in raw_struct:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], (list, tuple)):
                    candidates.extend([item[0], item[1]])
                else:
                    candidates.append(item)
        else:
            candidates = [raw_struct]

        for cand in candidates:
            if isinstance(cand, (list, tuple, np.ndarray)) and len(cand) > 0:
                # Check if it contains structure tags
                if isinstance(cand[0], str) or (isinstance(cand[0], dict) and "tag" in cand[0]):
                    pred_structures = cand
                # Check if it contains numeric bounding boxes
                elif isinstance(cand[0], (list, tuple, np.ndarray)) and len(cand[0]) >= 2:
                    if not isinstance(cand[0][0], str):
                        pred_bboxes = cand

        if not pred_structures and isinstance(raw_struct, str):
            pred_structures = raw_struct

        # Unwrap batch dimension if present
        if isinstance(pred_structures, list) and len(pred_structures) == 1 and isinstance(pred_structures[0], list):
            pred_structures = pred_structures[0]
        if isinstance(pred_bboxes, list) and len(pred_bboxes) == 1 and isinstance(pred_bboxes[0], (list, np.ndarray)):
            pred_bboxes = pred_bboxes[0]
        elif isinstance(pred_bboxes, np.ndarray) and pred_bboxes.ndim >= 3 and pred_bboxes.shape[0] == 1:
            pred_bboxes = pred_bboxes[0]

        return pred_structures, pred_bboxes

    def _match_cells_and_build_html(
        self, pred_structures: list, pred_bboxes: list, ocr_tokens: list[dict]
    ) -> str:
        """Matches OCR tokens to SLANet cell coordinates and reconstructs full HTML table."""
        # Standardize cell bounding boxes to [[x1, y1, x2, y2], ...]. Append None to maintain index alignment!
        cell_boxes = []
        if pred_bboxes is not None and not isinstance(pred_bboxes, (int, float)):
            for b in pred_bboxes:
                cell_boxes.append(self._normalize_box(b))

        # Map each OCR token to its best overlapping or enclosing cell box
        cell_to_tokens = {i: [] for i in range(len(cell_boxes))}
        for token in ocr_tokens:
            cx, cy = token["center"]
            ox1, oy1, ox2, oy2 = token["bbox"]
            ocr_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))

            matched_idx = None
            best_overlap = 0.0

            for c_idx, norm_b in enumerate(cell_boxes):
                if norm_b is None:
                    continue
                    
                bx1, by1, bx2, by2 = norm_b
                
                # 1. Point-in-box check (with 3px tolerance)
                if (bx1 - 3.0) <= cx <= (bx2 + 3.0) and (by1 - 3.0) <= cy <= (by2 + 3.0):
                    matched_idx = c_idx
                    break

                # 2. Area overlap check
                ix1 = max(bx1, ox1)
                iy1 = max(by1, oy1)
                ix2 = min(bx2, ox2)
                iy2 = min(by2, oy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    ratio = inter_area / ocr_area
                    if ratio > best_overlap:
                        best_overlap = ratio
                        matched_idx = c_idx

            if matched_idx is not None:
                cell_to_tokens[matched_idx].append(token)

        # Sort tokens in each cell top-to-bottom, left-to-right and join
        cell_texts = {}
        for c_idx, tokens in cell_to_tokens.items():
            tokens.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
            cell_texts[c_idx] = " ".join(t["text"] for t in tokens)

        # Assemble HTML table using pred_structures tokens
        if isinstance(pred_structures, (list, tuple)) and pred_structures:
            html_tokens = []
            cell_idx = 0
            for item in pred_structures:
                if isinstance(item, (list, tuple)):
                    tag = str(item[0]) if len(item) > 0 else ""
                elif isinstance(item, dict):
                    tag = str(item.get("tag", item.get("text", "")))
                else:
                    tag = str(item) if item is not None else ""

                if not tag:
                    continue

                tag_lower = tag.lower().strip()
                if tag_lower.startswith("<td") or tag_lower.startswith("<th"):
                    text_val = cell_texts.get(cell_idx, "")
                    cell_idx += 1
                    if tag_lower.endswith("</td>"):
                        tag_open = tag[: -len("</td>")]
                        html_tokens.append(f"{tag_open}{text_val}</td>")
                    elif tag_lower.endswith("</th>"):
                        tag_open = tag[: -len("</th>")]
                        html_tokens.append(f"{tag_open}{text_val}</th>")
                    else:
                        html_tokens.append(f"{tag}{text_val}")
                else:
                    html_tokens.append(tag)

            html_output = "".join(html_tokens)
        elif isinstance(pred_structures, str) and pred_structures:
            html_output = pred_structures
        else:
            html_output = ""

        # Check if we successfully mapped any text into the structured table
        has_text = any(bool(text.strip()) for text in cell_texts.values())

        # Fallback: if SLANet output was empty, invalid, or we couldn't map any text, build table rows directly from clustered OCR tokens
        if (not html_output or "<table" not in html_output.lower() or not has_text) and ocr_tokens:
            sorted_ocr = sorted(ocr_tokens, key=lambda t: t["bbox"][1])
            rows = []
            curr_row = []
            curr_y = None
            for t in sorted_ocr:
                y_mid = t["center"][1]
                if curr_y is None or abs(y_mid - curr_y) < 18.0:
                    curr_row.append(t)
                    curr_y = y_mid if curr_y is None else (curr_y + y_mid) / 2.0
                else:
                    curr_row.sort(key=lambda item: item["bbox"][0])
                    rows.append(curr_row)
                    curr_row = [t]
                    curr_y = y_mid
            if curr_row:
                curr_row.sort(key=lambda item: item["bbox"][0])
                rows.append(curr_row)

            # Cluster columns based on x-coordinates
            all_x = [t["center"][0] for row in rows for t in row]
            all_x.sort()
            
            cols = []
            if all_x:
                curr_c = [all_x[0]]
                for x in all_x[1:]:
                    # 25px threshold to group items into the same column
                    if x - (sum(curr_c) / len(curr_c)) < 25.0:
                        curr_c.append(x)
                    else:
                        cols.append(sum(curr_c) / len(curr_c))
                        curr_c = [x]
                if curr_c:
                    cols.append(sum(curr_c) / len(curr_c))

            table_rows_html = []
            for row in rows:
                row_tds = []
                col_idx = 0
                for t in row:
                    cx = t["center"][0]
                    # Find closest column
                    best_c = min(range(len(cols)), key=lambda i: abs(cols[i] - cx))
                    
                    # Fill with empty cells if we skipped columns (e.g., indentation)
                    while col_idx < best_c:
                        row_tds.append("<td></td>")
                        col_idx += 1
                        
                    row_tds.append(f"<td>{t['text']}</td>")
                    # Advance col_idx to prevent overwriting if multiple tokens share a column
                    col_idx = max(col_idx + 1, best_c + 1)
                    
                # Fill remaining columns
                while col_idx < len(cols):
                    row_tds.append("<td></td>")
                    col_idx += 1
                    
                table_rows_html.append(f"<tr>{''.join(row_tds)}</tr>")
            html_output = f"<table>{''.join(table_rows_html)}</table>"

        return html_output

    def extract(self, image_input) -> dict:
        if self.table_engine is None and self.ocr_engine is None:
            return {"html": "", "markdown": ""}

        # Standardize input to OpenCV BGR numpy array
        if isinstance(image_input, Image.Image):
            rgb_arr = np.asarray(image_input.convert("RGB"))
            img_bgr = rgb_arr[:, :, ::-1].copy()
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_bgr = np.stack([image_input] * 3, axis=-1).copy()
            elif image_input.shape[2] == 3:
                img_bgr = image_input.copy()
            else:
                img_bgr = image_input[:, :, :3].copy()
        else:
            return {"html": "", "markdown": ""}

        try:
            # 1. OCR text detection on table crop
            ocr_tokens = self._extract_ocr_tokens(img_bgr)

            # 2. SLANet table structure inference
            pred_structures, pred_bboxes = self._extract_slanet_structure(img_bgr)

            # 3. Match cells and construct HTML table
            html_str = self._match_cells_and_build_html(pred_structures, pred_bboxes, ocr_tokens)

            # 4. Convert HTML table to Markdown
            markdown_str = html_table_to_markdown(html_str)

            return {
                "html": html_str or "",
                "markdown": markdown_str or "",
            }
        except Exception as e:
            logger.exception("Error extracting table structure")
            return {"html": "", "markdown": ""}


def _get_tensor(tensor_dict: dict, *candidate_keys):
    """Safely retrieves a tensor matching candidate keys from a safetensors dictionary."""
    if not tensor_dict:
        return None
    for k in candidate_keys:
        if k in tensor_dict:
            return tensor_dict[k]
    for k, v in tensor_dict.items():
        for cand in candidate_keys:
            if k.endswith(cand):
                return v
    return None


class EttinONNXReranker:
    """
    Pure ONNX-based Cross-Encoder reranker using Ettin 150M.
    Loads local ONNX, tokenizer, and classification head files directly without huggingface_hub.
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
        if onnx_path and os.path.exists(onnx_path):
            actual_onnx_path = onnx_path
        else:
            candidates = [
                os.path.join(model_dir, "onnx", "model.onnx"),
                os.path.join(model_dir, "model.onnx"),
                os.path.join(model_dir, "model_quantized.onnx"),
            ]
            actual_onnx_path = None
            for cand in candidates:
                if os.path.exists(cand):
                    actual_onnx_path = cand
                    break

            if not actual_onnx_path:
                raise FileNotFoundError(
                    f"No ONNX model file found in '{model_dir}'. Checked: {candidates}"
                )

        logger.info(f"Loading ONNX model from {actual_onnx_path}...")

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
        logger.info(f"ONNX Model inputs expected: {self.input_names}")

        # 3. Load classification head weights from local safetensors
        self.head_weights = self._load_classification_head()

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
            logger.warning(
                "Classification head safetensors not found locally. "
                "Will assume ONNX model outputs logits directly if output is 1D/2D."
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
        w2 = _get_tensor(d2, "linear.weight", "weight", "0.weight")
        b2 = _get_tensor(d2, "linear.bias", "bias", "0.bias")
        if w2 is None:
            raise KeyError("Could not find linear.weight in 2_Dense safetensors")

        x = cls_embeddings @ w2.T
        if b2 is not None:
            x = x + b2
        x = _gelu_numpy(x)

        # 2. LayerNorm 3: LayerNorm(H)
        w_ln = _get_tensor(ln3, "norm.weight", "layer_norm.weight", "weight", "gamma", "0.weight")
        b_ln = _get_tensor(ln3, "norm.bias", "layer_norm.bias", "bias", "beta", "0.bias")

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        eps = 1e-5
        x = (x - mean) / np.sqrt(var + eps)
        if w_ln is not None:
            x = x * w_ln
        if b_ln is not None:
            x = x + b_ln

        # 3. Dense 4: Linear(H -> 1)
        w4 = _get_tensor(d4, "linear.weight", "weight", "0.weight")
        b4 = _get_tensor(d4, "linear.bias", "bias", "0.bias")
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
                os.path.join(model_dir, "onnx", "model.onnx"),
                os.path.join(model_dir, "model.onnx"),
                os.path.join(model_dir, "model_quantized.onnx"),
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


class DocLayNetONNX:
    """
    Pure ONNX-based YOLOv8 document layout detection model for DocLayNet.
    Detects 11 distinct document regions:
    0: Caption, 1: Footnote, 2: Formula, 3: List-item, 4: Page-footer,
    5: Page-header, 6: Picture, 7: Section-header, 8: Table, 9: Text, 10: Title
    """

    DOCLAYNET_LABELS = [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
    ]

    def __init__(
        self,
        model_dir: str = "./model",
        onnx_path: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        use_gpu: bool = False,
        enable_table_rec: bool = True,
        table_model_path: str = None,
    ):
        self.model_dir = model_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.labels = self.DOCLAYNET_LABELS
        self.table_recognizer = None

        if enable_table_rec:
            try:
                self.table_recognizer = TableRecognizerONNX(
                    table_model_path=table_model_path,
                    use_gpu=use_gpu,
                )
            except Exception as e:
                logger.warning(f"Could not load TableRecognizer: {e}")

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

        # 2. Select ONNX execution providers
        available_providers = ort.get_available_providers()
        providers = []
        if use_gpu and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        logger.info(f"Initializing ONNX DocLayNet InferenceSession with providers: {providers}")
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

    def _load_image(self, image_input) -> Image.Image:
        """Loads and converts image inputs (PIL Image, bytes, base64 data URI, HTTP URL, or local path) to RGB."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                base64_data = image_input.split(",", 1)[1]
                return Image.open(io.BytesIO(base64.b64decode(base64_data))).convert("RGB")
            if image_input.startswith("http://") or image_input.startswith("https://"):
                resp = requests.get(image_input, timeout=15)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            if os.path.exists(image_input):
                return Image.open(image_input).convert("RGB")
            try:
                raw_bytes = base64.b64decode(image_input)
                return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            except Exception:
                raise ValueError(f"Could not decode image string: {image_input[:60]}...")
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _letterbox(
        self, image: Image.Image, target_size: int = 640
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """
        Resize image with constant aspect ratio and pad to (target_size, target_size).
        Returns normalized tensor (1, 3, target_size, target_size), scale factor, and (pad_w, pad_h).
        """
        orig_w, orig_h = image.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_w = (target_size - new_w) / 2.0
        pad_h = (target_size - new_h) / 2.0

        canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
        canvas.paste(resized, (int(round(pad_w)), int(round(pad_h))))

        img_data = np.asarray(canvas, dtype=np.float32) / 255.0  # HWC
        img_data = np.transpose(img_data, (2, 0, 1))            # CHW
        img_tensor = np.expand_dims(img_data, axis=0)           # 1, C, H, W

        return img_tensor, scale, (pad_w, pad_h)

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
    ) -> list[int]:
        """Class-aware Non-Maximum Suppression (NMS)."""
        if len(boxes) == 0:
            return []

        keep = []
        unique_classes = np.unique(class_ids)

        for c in unique_classes:
            cls_indices = np.where(class_ids == c)[0]
            cls_boxes = boxes[cls_indices]
            cls_scores = scores[cls_indices]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            while order.size > 0:
                i = order[0]
                keep.append(cls_indices[i])
                if order.size == 1:
                    break

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

                inds = np.where(iou <= iou_threshold)[0]
                order = order[inds + 1]

        return sorted(keep, key=lambda idx: scores[idx], reverse=True)

    def predict(
        self,
        image_input,
        conf_threshold: float = None,
        iou_threshold: float = None,
        extract_tables: bool = True,
    ) -> tuple[list[dict], tuple[int, int]]:
        """
        Detect document layout elements in an image.
        If extract_tables=True, automatically parses regions labeled 'Table' into HTML/Markdown.
        Returns (list_of_detections, (orig_width, orig_height)).
        """
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold

        img = self._load_image(image_input)
        orig_w, orig_h = img.size

        img_tensor, scale, (pad_w, pad_h) = self._letterbox(img, target_size=self.image_size)

        onnx_inputs = {self.input_names[0]: img_tensor}
        outputs = self.session.run(None, onnx_inputs)
        raw_output = outputs[0]  # Shape: (1, 15, anchors) or (1, anchors, 15)

        predictions = np.squeeze(raw_output, axis=0)

        # Standard YOLOv8 output is (num_features, num_anchors) where features = 4 + num_classes
        if predictions.shape[0] == (4 + len(self.labels)):
            predictions = np.transpose(predictions, (1, 0))  # (num_anchors, 15)

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Confidence filter
        mask = max_scores >= conf_thresh
        if not np.any(mask):
            return [], (orig_w, orig_h)

        valid_boxes = boxes_cxcywh[mask]
        valid_scores = max_scores[mask]
        valid_class_ids = class_ids[mask]

        # Convert [center_x, center_y, width, height] in letterbox space to [x1, y1, x2, y2] in original image space
        x1 = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2.0 - pad_w) / scale
        y1 = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2.0 - pad_h) / scale
        x2 = (valid_boxes[:, 0] + valid_boxes[:, 2] / 2.0 - pad_w) / scale
        y2 = (valid_boxes[:, 1] + valid_boxes[:, 3] / 2.0 - pad_h) / scale

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes_xyxy = np.column_stack([x1, y1, x2, y2])

        # Apply NMS
        keep_indices = self._nms(boxes_xyxy, valid_scores, valid_class_ids, iou_threshold=iou_thresh)

        detections = []
        for idx in keep_indices:
            cid = int(valid_class_ids[idx])
            label_name = self.labels[cid] if cid < len(self.labels) else f"class_{cid}"
            det_item = {
                "bbox": [
                    round(float(boxes_xyxy[idx, 0]), 2),
                    round(float(boxes_xyxy[idx, 1]), 2),
                    round(float(boxes_xyxy[idx, 2]), 2),
                    round(float(boxes_xyxy[idx, 3]), 2),
                ],
                "confidence": round(float(valid_scores[idx]), 4),
                "class_id": cid,
                "label": label_name,
            }

            # If the detected region is a Table and table extraction is active, convert to HTML/Markdown
            if extract_tables and self.table_recognizer is not None and label_name.lower() == "table":
                bx1, by1, bx2, by2 = boxes_xyxy[idx]
                cx1 = max(0, int(bx1) - 4)
                cy1 = max(0, int(by1) - 4)
                cx2 = min(orig_w, int(bx2) + 4)
                cy2 = min(orig_h, int(by2) + 4)
                if cx2 > cx1 and cy2 > cy1:
                    crop = img.crop((cx1, cy1, cx2, cy2))
                    table_data = self.table_recognizer.extract(crop)
                    if table_data.get("html"):
                        det_item["html"] = table_data["html"]
                    if table_data.get("markdown"):
                        det_item["markdown"] = table_data["markdown"]

            detections.append(det_item)

        return detections, (orig_w, orig_h)
