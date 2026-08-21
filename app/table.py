import os
import logging
import numpy as np
from PIL import Image

from .utils import html_table_to_markdown

logger = logging.getLogger("ettin-reranker")


class TableRecognizerONNX:
    """
    Pure ONNX-based Table Structure Recognizer using SLANet.
    Decodes table structure HTML tags and cell bounding boxes natively without
    rapid-table or rapidocr dependencies.
    """

    # 41-token SLANet / PP-Structure table structure vocabulary
    VOCAB = [
        "beg",
        "end",
        "<html>",
        "<body>",
        "<table>",
        "<thead>",
        "<tbody>",
        "<tr>",
        "<td>",
        "<td",
        ">",
        "</td>",
        "<th>",
        "<th",
        "</th>",
        "</tr>",
        "</thead>",
        "</tbody>",
        "</table>",
        "</body>",
        "</html>",
        'colspan="2"',
        'colspan="3"',
        'colspan="4"',
        'colspan="5"',
        'colspan="6"',
        'colspan="7"',
        'colspan="8"',
        'colspan="9"',
        'colspan="10"',
        'rowspan="2"',
        'rowspan="3"',
        'rowspan="4"',
        'rowspan="5"',
        'rowspan="6"',
        'rowspan="7"',
        'rowspan="8"',
        'rowspan="9"',
        'rowspan="10"',
        "<td></td>",
        "<th></th>",
    ]

    INPUT_SHAPE = (488, 488)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, table_model_path: str = None, use_gpu: bool = False):
        self.session = None
        self.input_name = None
        self.output_names = []
        self.use_tesseract = False
        self._init_engines(table_model_path, use_gpu)

    def _find_or_download_model(self, table_model_path: str = None) -> str:
        """Locates the SLANet ONNX model file or downloads it automatically if missing."""
        candidates = []
        if table_model_path:
            candidates.append(table_model_path)

        candidates.extend([
            "./table_model/ch_ppstructure_mobile_v2_SLANet.onnx",
            "./table_model/slanet.onnx",
            "./model/ch_ppstructure_mobile_v2_SLANet.onnx",
            "./model/slanet.onnx",
            "./model/table/ch_ppstructure_mobile_v2_SLANet.onnx",
        ])

        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand

        # Check table_model directory for any .onnx file
        for search_dir in ["./table_model", "./model/table", "./model"]:
            if os.path.isdir(search_dir):
                for root, _, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".onnx") and ("slanet" in file.lower() or "table" in file.lower()):
                            return os.path.join(root, file)

        # Automatic fallback download if no local weights exist
        target_path = "./table_model/ch_ppstructure_mobile_v2_SLANet.onnx"
        try:
            import urllib.request
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            url = "https://huggingface.co/SWHL/RapidStructure/resolve/main/table/ch_ppstructure_mobile_v2_SLANet.onnx"
            logger.info(f"Downloading SLANet ONNX weights from {url} to {target_path}...")
            urllib.request.urlretrieve(url, target_path)
            logger.info("SLANet weights downloaded successfully.")
            return target_path
        except Exception as e:
            logger.warning(f"Could not auto-download SLANet model: {e}")

        return None

    def _init_engines(self, table_model_path: str = None, use_gpu: bool = False):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.use_tesseract = True
            logger.info("Initialized Pytesseract OCR engine for table text recognition.")
        except Exception:
            logger.info("Pytesseract not found or tesseract binary missing. Cell text OCR will use heuristic clustering.")

        model_path = self._find_or_download_model(table_model_path)
        if not model_path or not os.path.exists(model_path):
            logger.warning(
                f"SLANet ONNX model not found. Checked: {table_model_path}. "
                "Download weights with: curl -L https://huggingface.co/SWHL/RapidStructure/resolve/main/table/ch_ppstructure_mobile_v2_SLANet.onnx -o table_model/ch_ppstructure_mobile_v2_SLANet.onnx"
            )
            return

        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            providers = []
            if use_gpu and "CUDAExecutionProvider" in available_providers:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            logger.info(f"Initialized native SLANet ONNX session from {model_path} with providers: {providers}")
        except Exception as e:
            logger.warning(f"Could not initialize SLANet ONNX session: {e}")
            self.session = None

    def _extract_ocr_tokens(self, img_bgr: np.ndarray) -> list[dict]:
        """Extracts text tokens and bounding boxes from image using Pytesseract."""
        if not self.use_tesseract:
            return []

        toks = []
        try:
            import cv2
            import pytesseract

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)
            n_boxes = len(data.get("text", []))

            for i in range(n_boxes):
                text_str = str(data["text"][i]).strip()
                if not text_str:
                    continue

                x = float(data["left"][i])
                y = float(data["top"][i])
                w = float(data["width"][i])
                h = float(data["height"][i])

                if w <= 0 or h <= 0:
                    continue

                x1, y1, x2, y2 = x, y, x + w, y + h
                toks.append({
                    "bbox": [x1, y1, x2, y2],
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "text": text_str,
                })
        except Exception as e:
            logger.debug(f"Pytesseract table OCR failed: {e}")

        return toks

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

    def _preprocess_slanet(self, img_bgr: np.ndarray) -> np.ndarray:
        """Prepares input image for SLANet: resize to (488, 488) and standard ImageNet normalize."""
        import cv2
        resized = cv2.resize(img_bgr, self.INPUT_SHAPE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.MEAN) / self.STD
        tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(tensor, axis=0).astype(np.float32)

    def _extract_slanet_structure(self, img_bgr: np.ndarray) -> tuple[list, list]:
        """Runs native SLANet ONNX inference and decodes HTML structure tokens and cell bboxes."""
        pred_structures = []
        pred_bboxes = []

        if self.session is None:
            return pred_structures, pred_bboxes

        try:
            h, w = img_bgr.shape[:2]
            img_tensor = self._preprocess_slanet(img_bgr)
            outputs = self.session.run(None, {self.input_name: img_tensor})

            # Identify structure_probs (3D, last dim ~41) and loc_preds (3D, last dim == 4)
            structure_probs = None
            loc_preds = None

            for out in outputs:
                if out.ndim == 3:
                    if out.shape[-1] == 4:
                        loc_preds = out[0]
                    else:
                        structure_probs = out[0]

            if structure_probs is None:
                return pred_structures, pred_bboxes

            pred_token_indices = np.argmax(structure_probs, axis=-1)

            for t in range(len(pred_token_indices)):
                idx = int(pred_token_indices[t])
                if idx >= len(self.VOCAB):
                    continue

                token = self.VOCAB[idx]
                if token in ("end", "eos"):
                    break
                if token in ("beg", "sos"):
                    continue

                if token in ("<td>", "<td></td>", "<td", "<th>", "<th></th>", "<th") and loc_preds is not None:
                    bbox = loc_preds[t]
                    # Rescale if output is in 488px space vs [0, 1] normalized
                    if np.max(bbox) > 1.5:
                        bx1, by1, bx2, by2 = bbox[0] / 488.0, bbox[1] / 488.0, bbox[2] / 488.0, bbox[3] / 488.0
                    else:
                        bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]

                    real_x1 = max(0.0, min(float(w), float(bx1 * w)))
                    real_y1 = max(0.0, min(float(h), float(by1 * h)))
                    real_x2 = max(0.0, min(float(w), float(bx2 * w)))
                    real_y2 = max(0.0, min(float(h), float(by2 * h)))
                    pred_bboxes.append([real_x1, real_y1, real_x2, real_y2])

                pred_structures.append(token)
                if token == "</html>":
                    break

        except Exception as e:
            logger.warning(f"Native SLANet inference failed: {e}")

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

        # Assemble HTML table using pred_structures tokens with cell text injection
        if isinstance(pred_structures, (list, tuple)) and pred_structures:
            html_tokens = []
            cell_idx = 0
            i = 0
            n = len(pred_structures)
            while i < n:
                item = pred_structures[i]
                tag = str(item[0]) if isinstance(item, (list, tuple)) and len(item) > 0 else str(item)
                tag_lower = tag.lower().strip()

                if tag_lower in ("<td></td>", "<th></th>"):
                    cell_tag = "td" if "td" in tag_lower else "th"
                    text_val = cell_texts.get(cell_idx, "")
                    cell_idx += 1
                    html_tokens.append(f"<{cell_tag}>{text_val}</{cell_tag}>")
                    i += 1
                elif tag_lower in ("<td", "<th"):
                    cell_tag = "td" if "td" in tag_lower else "th"
                    attrs = []
                    i += 1
                    while i < n and pred_structures[i].strip() != ">":
                        attrs.append(pred_structures[i].strip())
                        i += 1
                    if i < n and pred_structures[i].strip() == ">":
                        i += 1
                    if i < n and pred_structures[i].strip().lower() in ("</td>", "</th>"):
                        i += 1
                    attr_str = (" " + " ".join(attrs)) if attrs else ""
                    text_val = cell_texts.get(cell_idx, "")
                    cell_idx += 1
                    html_tokens.append(f"<{cell_tag}{attr_str}>{text_val}</{cell_tag}>")
                elif tag_lower in ("<td>", "<th>"):
                    cell_tag = "td" if "td" in tag_lower else "th"
                    i += 1
                    if i < n and pred_structures[i].strip().lower() in ("</td>", "</th>"):
                        i += 1
                    text_val = cell_texts.get(cell_idx, "")
                    cell_idx += 1
                    html_tokens.append(f"<{cell_tag}>{text_val}</{cell_tag}>")
                else:
                    html_tokens.append(tag)
                    i += 1

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

    def _safe_load_bgr(self, source) -> np.ndarray:
        """Helper to load image with ImageMagick fallback for WMF/EPS formats."""
        import tempfile
        import subprocess
        import io
        img = None
        try:
            if isinstance(source, bytes):
                img = Image.open(io.BytesIO(source))
            else:
                img = Image.open(source)
            rgb_arr = np.asarray(img.convert("RGB"))
            return rgb_arr[:, :, ::-1].copy()
        except Exception as e:
            logger.debug(f"Pillow load failed: {e}. Falling back to ImageMagick.")
            with tempfile.TemporaryDirectory() as tmpdir:
                header = b""
                if isinstance(source, (bytes, bytearray)):
                    header = source[:128]
                elif isinstance(source, str) and os.path.exists(source):
                    with open(source, "rb") as f:
                        header = f.read(128)

                ext = "tmp"
                if img is not None and hasattr(img, 'format') and img.format:
                    ext = str(img.format).lower()
                if ext == "wmf" and b"EMF" in header:
                    ext = "emf"

                if isinstance(source, (bytes, bytearray)):
                    in_path = os.path.join(tmpdir, f"input.{ext}")
                    with open(in_path, "wb") as f:
                        f.write(source)
                else:
                    in_path = str(source)
                    
                out_path = os.path.join(tmpdir, "output.png")
                
                cmds = [
                    ["magick", "-density", "300", in_path, out_path],
                    ["convert", "-density", "300", in_path, out_path],
                    ["inkscape", in_path, "--export-type=png", f"--export-filename={out_path}", "--export-dpi=300", "--export-background=white"],
                    ["soffice", "--headless", "--convert-to", "png", "--outdir", tmpdir, in_path]
                ]
                
                success = False
                last_err = ""
                for cmd in cmds:
                    try:
                        subprocess.run(
                            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                        )
                        success = True
                        break
                    except subprocess.CalledProcessError as cpe:
                        last_err = f"{cmd[0]} failed: {cpe.stderr.strip()}"
                        continue
                    except FileNotFoundError:
                        continue
                
                final_path = None
                for candidate in [out_path, os.path.join(tmpdir, "output-0.png"), os.path.join(tmpdir, f"input.png")]:
                    if os.path.exists(candidate):
                        final_path = candidate
                        break

                if not final_path:
                    err_msg = f"Vector fallback failed. Pillow err: {e}."
                    if last_err:
                        err_msg += f" Last tool err: {last_err}"
                    raise ValueError(err_msg)
                    
                with Image.open(final_path) as tmp_img:
                    rgb_arr = np.asarray(tmp_img.convert("RGB"))
                    return rgb_arr[:, :, ::-1].copy()

    def extract(self, image_input) -> dict:
        if self.session is None and not self.use_tesseract:
            return {"html": "", "markdown": ""}

        # Standardize input to OpenCV BGR numpy array
        if isinstance(image_input, Image.Image):
            try:
                rgb_arr = np.asarray(image_input.convert("RGB"))
                img_bgr = rgb_arr[:, :, ::-1].copy()
            except Exception as e:
                # If Pillow lazy-loading fails (e.g. truncated image)
                logger.warning(f"PIL conversion failed, skipping: {e}")
                return {"html": "", "markdown": ""}
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_bgr = np.stack([image_input] * 3, axis=-1).copy()
            elif image_input.shape[2] == 3:
                img_bgr = image_input.copy()
            else:
                img_bgr = image_input[:, :, :3].copy()
        elif isinstance(image_input, (str, bytes)):
            try:
                img_bgr = self._safe_load_bgr(image_input)
            except Exception as e:
                logger.warning(f"Fallback loading failed: {e}")
                return {"html": "", "markdown": ""}
        else:
            return {"html": "", "markdown": ""}

        # Scale images to a "sweet spot" resolution (longest edge ~736px)
        # to preserve text strokes and grid lines for SLANet and OCR accuracy,
        # while preventing OOM errors and degraded feature matching on massive crops.
        try:
            import cv2
            h, w = img_bgr.shape[:2]
            target_long_edge = 736
            max_edge = max(h, w)
            if max_edge > 0 and abs(max_edge - target_long_edge) > 10:
                scale = target_long_edge / max_edge
                new_w = int(w * scale)
                new_h = int(h * scale)
                interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
        except Exception as e:
            logger.debug(f"Could not resize table crop: {e}")

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
