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

    # Canonical 50-token vocabulary for PP-Structure SLANet (ch_ppstructure_mobile_v2_SLANet)
    VOCAB_50 = [
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
        'colspan="11"',
        'colspan="12"',
        'colspan="13"',
        'colspan="14"',
        'colspan="15"',
        'colspan="16"',
        'colspan="17"',
        'colspan="18"',
        'colspan="19"',
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
        "beg",
        "end",
    ]

    # 41-token vocabulary for English SLANet variant
    VOCAB_41 = [
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
        "beg",
        "end",
    ]

    VOCAB = VOCAB_50

    INPUT_SHAPE = (488, 488)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, table_model_path: str = None, use_gpu: bool = False):
        self.session = None
        self.input_name = None
        self.output_names = []
        self.use_tesseract = False
        self._init_engines(table_model_path, use_gpu)

    @staticmethod
    def _is_valid_onnx_file(path: str) -> bool:
        """Validates that a file exists, is non-empty (>100KB), and is not a Git LFS pointer."""
        if not path or not os.path.isfile(path):
            return False
        try:
            # Model weights are ~7MB; Git LFS text pointers or empty files are < 1KB
            if os.path.getsize(path) < 100 * 1024:
                return False
            with open(path, "rb") as f:
                header = f.read(64)
                if b"version https://git-lfs" in header:
                    return False
            return True
        except Exception:
            return False

    def _get_candidate_model_paths(self, table_model_path: str = None) -> list[str]:
        """Collects candidate paths for the SLANet ONNX model."""
        candidates = []
        if table_model_path:
            candidates.append(table_model_path)

        home = os.path.expanduser("~")
        candidates.extend([
            "./table_model/ch_ppstructure_mobile_v2_SLANet.onnx",
            "./table_model/slanet.onnx",
            os.path.join(home, ".rapid_table", "ch_ppstructure_mobile_v2_SLANet.onnx"),
            os.path.join(home, ".cache", "rapid_table", "ch_ppstructure_mobile_v2_SLANet.onnx"),
            "./model/ch_ppstructure_mobile_v2_SLANet.onnx",
            "./model/slanet.onnx",
            "./model/table/ch_ppstructure_mobile_v2_SLANet.onnx",
        ])

        for search_dir in ["./table_model", "./model/table", "./model", os.path.join(home, ".rapid_table")]:
            if os.path.isdir(search_dir):
                for root, _, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".onnx") and ("slanet" in file.lower() or "table" in file.lower()):
                            candidates.append(os.path.join(root, file))

        seen = set()
        unique_candidates = []
        for c in candidates:
            abs_c = os.path.abspath(c)
            if abs_c not in seen:
                seen.add(abs_c)
                unique_candidates.append(c)

        return unique_candidates

    def _init_engines(self, table_model_path: str = None, use_gpu: bool = False):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.use_tesseract = True
            logger.info("Initialized Pytesseract OCR engine for table text recognition.")
        except Exception:
            logger.info("Pytesseract not found or tesseract binary missing. Cell text OCR will use heuristic clustering.")

        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        providers = []
        if use_gpu and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        candidates = self._get_candidate_model_paths(table_model_path)

        # 1. Try loading all valid candidate files in priority order
        for cand in candidates:
            if self._is_valid_onnx_file(cand):
                try:
                    self.session = ort.InferenceSession(cand, sess_options, providers=providers)
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_names = [out.name for out in self.session.get_outputs()]
                    logger.info(f"Initialized native SLANet ONNX session from {cand} with providers: {providers}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load candidate ONNX model from {cand}: {e}")

        # 2. If no valid local ONNX model loaded, auto-download canonical SLANet weights from Hugging Face
        download_target = "./table_model/ch_ppstructure_mobile_v2_SLANet.onnx"
        try:
            import urllib.request
            os.makedirs(os.path.dirname(download_target), exist_ok=True)
            url = "https://huggingface.co/SWHL/RapidStructure/resolve/main/table/ch_ppstructure_mobile_v2_SLANet.onnx"
            logger.info(f"Downloading SLANet ONNX weights from {url} to {download_target}...")
            urllib.request.urlretrieve(url, download_target)
            if self._is_valid_onnx_file(download_target):
                self.session = ort.InferenceSession(download_target, sess_options, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [out.name for out in self.session.get_outputs()]
                logger.info(f"Successfully loaded auto-downloaded SLANet ONNX session from {download_target}")
                return
        except Exception as e:
            logger.warning(f"Could not auto-download SLANet model: {e}")

        logger.warning(
            "SLANet ONNX model could not be initialized. Table extraction will rely on OCR geometric clustering fallback. "
            "To enable neural structure detection, download the weights manually:\n"
            "curl -L https://huggingface.co/SWHL/RapidStructure/resolve/main/table/ch_ppstructure_mobile_v2_SLANet.onnx -o table_model/ch_ppstructure_mobile_v2_SLANet.onnx"
        )

    def _extract_ocr_tokens(self, img_bgr: np.ndarray) -> list[dict]:
        """
        Extracts high-precision text tokens and bounding boxes from table image
        using multi-scale and dual-polarity (normal + inverted) Pytesseract passes
        to capture standard text as well as white-on-dark/colored header text.
        """
        if not self.use_tesseract:
            return []

        try:
            import cv2
            import pytesseract

            h, w = img_bgr.shape[:2]
            scale_factor = max(1.0, 1400.0 / max(h, w)) if max(h, w) < 1400 else 1.0

            if scale_factor > 1.0:
                new_w = int(round(w * scale_factor))
                new_h = int(round(h * scale_factor))
                scaled_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                scaled_bgr = img_bgr

            def run_tesseract_pass(img_arr: np.ndarray) -> list[dict]:
                rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                custom_config = r"--psm 6"
                data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT, config=custom_config)
                n_boxes = len(data.get("text", []))
                pass_toks = []

                for i in range(n_boxes):
                    text_str = str(data["text"][i]).strip()
                    if not text_str:
                        continue

                    conf = float(data.get("conf", [100])[i])
                    if conf < 0:
                        continue

                    x = float(data["left"][i]) / scale_factor
                    y = float(data["top"][i]) / scale_factor
                    bw = float(data["width"][i]) / scale_factor
                    bh = float(data["height"][i]) / scale_factor

                    if bw <= 0 or bh <= 0:
                        continue

                    x1, y1, x2, y2 = x, y, x + bw, y + bh

                    import re
                    stripped = text_str.strip()
                    if not stripped:
                        continue

                    # Filter out grid line segmentation noise and low-confidence punctuation
                    if conf < 35.0 and not any(c.isalnum() for c in stripped):
                        continue
                    if re.fullmatch(r"[~—–`'\"|_+=^]+", stripped):
                        continue
                    if re.fullmatch(r"[~—–`'\"|_.\-,;:!?\s]+", stripped) and len(stripped) > 1:
                        if not re.fullmatch(r"[-–—]+", stripped) and not re.fullmatch(r"\.{2,}", stripped):
                            continue

                    text_clean = re.sub(r'([.,:;!?])([A-Za-z])', r'\1 \2', stripped)

                    pass_toks.append({
                        "bbox": [x1, y1, x2, y2],
                        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        "text": text_clean,
                    })
                return pass_toks

            # Pass 1: Standard contrast (dark text on light background)
            normal_tokens = run_tesseract_pass(scaled_bgr)

            # Pass 2: Inverted contrast (white/light text on dark/colored header background)
            negated_tokens = run_tesseract_pass(255 - scaled_bgr)

            # Merge and deduplicate tokens (Intersection over Min Area)
            ocr_tokens = list(normal_tokens)
            for n_tok in negated_tokens:
                is_dup = False
                nx1, ny1, nx2, ny2 = n_tok["bbox"]
                n_area = max(0.0, (nx2 - nx1) * (ny2 - ny1))

                for m_tok in ocr_tokens:
                    mx1, my1, mx2, my2 = m_tok["bbox"]
                    m_area = max(0.0, (mx2 - mx1) * (my2 - my1))

                    ix1, iy1 = max(nx1, mx1), max(ny1, my1)
                    ix2, iy2 = min(nx2, mx2), min(ny2, my2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

                    if min(n_area, m_area) > 0 and (inter / min(n_area, m_area)) > 0.4:
                        is_dup = True
                        if len(n_tok["text"]) > len(m_tok["text"]) + 1:
                            m_tok["text"] = n_tok["text"]
                        break

                if not is_dup:
                    ocr_tokens.append(n_tok)

            return ocr_tokens
        except Exception as e:
            logger.debug(f"Pytesseract table OCR failed: {e}")
            return []

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

    def _preprocess_slanet(self, img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Prepares input image for SLANet: aspect-preserving resize with padding to (488, 488)."""
        import cv2
        h, w = img_bgr.shape[:2]
        max_edge = max(h, w)
        scale = 488.0 / max_edge if max_edge > 0 else 1.0
        resize_w = int(round(w * scale))
        resize_h = int(round(h * scale))
        resized = cv2.resize(img_bgr, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 3), dtype=np.uint8)
        padded[:resize_h, :resize_w, :] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.MEAN) / self.STD
        tensor = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(tensor, axis=0).astype(np.float32), scale

    def _extract_slanet_structure(self, img_bgr: np.ndarray) -> tuple[list, list]:
        """Runs native SLANet ONNX inference and decodes HTML structure tokens and cell bboxes."""
        pred_structures = []
        pred_bboxes = []

        if self.session is None:
            logger.warning("SLANet ONNX session is None. Skipping neural table extraction.")
            return pred_structures, pred_bboxes

        try:
            h, w = img_bgr.shape[:2]
            img_tensor, scale = self._preprocess_slanet(img_bgr)
            outputs = self.session.run(None, {self.input_name: img_tensor})

            structure_probs = None
            loc_preds = None

            for out in outputs:
                arr = np.squeeze(out)
                if arr.ndim == 2:
                    # Bounding box coordinates output: shape [N, 4] or [N, 8]
                    if arr.shape[-1] in (4, 8):
                        loc_preds = arr
                    elif arr.shape[0] in (4, 8) and arr.shape[1] > 8:
                        loc_preds = arr.T
                    # Structure tag probabilities output: shape [N, vocab_size]
                    elif arr.shape[-1] > 8:
                        structure_probs = arr
                    elif arr.shape[0] > 8 and arr.shape[1] <= 8:
                        structure_probs = arr.T

            if structure_probs is None:
                logger.warning(f"Could not identify structure_probs from ONNX output shapes: {[o.shape for o in outputs]}")
                return pred_structures, pred_bboxes

            # Select appropriate vocabulary based on output classification dimension
            vocab_dim = structure_probs.shape[-1]
            if vocab_dim == 50:
                vocab = self.VOCAB_50
            elif vocab_dim == 41:
                vocab = self.VOCAB_41
            elif len(self.VOCAB) == vocab_dim:
                vocab = self.VOCAB
            else:
                vocab = self.VOCAB_50 if vocab_dim >= 50 else self.VOCAB_41

            pred_token_indices = np.argmax(structure_probs, axis=-1)

            for t in range(len(pred_token_indices)):
                idx = int(pred_token_indices[t])
                if idx >= len(vocab):
                    continue

                token = vocab[idx]
                if token in ("end", "eos"):
                    break
                if token in ("beg", "sos"):
                    continue

                if token in ("<td>", "<td></td>", "<td", "<th>", "<th></th>", "<th") and loc_preds is not None and t < len(loc_preds):
                    raw_b = loc_preds[t]
                    if len(raw_b) == 8:
                        bx1 = float(min(raw_b[0], raw_b[2], raw_b[4], raw_b[6]))
                        by1 = float(min(raw_b[1], raw_b[3], raw_b[5], raw_b[7]))
                        bx2 = float(max(raw_b[0], raw_b[2], raw_b[4], raw_b[6]))
                        by2 = float(max(raw_b[1], raw_b[3], raw_b[5], raw_b[7]))
                    else:
                        bx1, by1, bx2, by2 = float(raw_b[0]), float(raw_b[1]), float(raw_b[2]), float(raw_b[3])

                    # Convert normalized coords (0..1) to 488 canvas coords if needed
                    if max(bx1, by1, bx2, by2) <= 1.5:
                        bx1, by1, bx2, by2 = bx1 * 488.0, by1 * 488.0, bx2 * 488.0, by2 * 488.0

                    # Unscale from padded 488x488 canvas back to original image dimensions
                    real_x1 = max(0.0, min(float(w), bx1 / scale))
                    real_y1 = max(0.0, min(float(h), by1 / scale))
                    real_x2 = max(0.0, min(float(w), bx2 / scale))
                    real_y2 = max(0.0, min(float(h), by2 / scale))

                    min_x, max_x = min(real_x1, real_x2), max(real_x1, real_x2)
                    min_y, max_y = min(real_y1, real_y2), max(real_y1, real_y2)
                    pred_bboxes.append([min_x, min_y, max_x, max_y])

                pred_structures.append(token)
                if token == "</html>":
                    break

            logger.info(f"SLANet extracted {len(pred_structures)} structure tokens and {len(pred_bboxes)} cell bboxes.")
        except Exception as e:
            logger.exception(f"Native SLANet inference failed: {e}")

        return pred_structures, pred_bboxes

    def _match_cells_and_build_html(
        self, pred_structures: list, pred_bboxes: list, ocr_tokens: list[dict], img_bgr: np.ndarray = None
    ) -> str:
        """Matches OCR tokens to SLANet cell coordinates and reconstructs full HTML table."""
        cell_boxes = []
        if pred_bboxes is not None and not isinstance(pred_bboxes, (int, float)):
            for b in pred_bboxes:
                cell_boxes.append(self._normalize_box(b))

        cell_to_tokens = {i: [] for i in range(len(cell_boxes))}
        for token in ocr_tokens:
            cx, cy = token["center"]
            ox1, oy1, ox2, oy2 = token["bbox"]
            ocr_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))

            best_idx = None
            best_score = -1.0

            for c_idx, norm_b in enumerate(cell_boxes):
                if norm_b is None:
                    continue

                bx1, by1, bx2, by2 = norm_b

                # Compute intersection area
                ix1 = max(bx1, ox1)
                iy1 = max(by1, oy1)
                ix2 = min(bx2, ox2)
                iy2 = min(by2, oy2)

                inter_w = max(0.0, ix2 - ix1)
                inter_h = max(0.0, iy2 - iy1)
                inter_area = inter_w * inter_h

                overlap_ratio = inter_area / ocr_area
                center_inside = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)

                score = overlap_ratio + (1.0 if center_inside else 0.0)
                if score > best_score and (score > 0.35 or center_inside):
                    best_score = score
                    best_idx = c_idx

            if best_idx is not None:
                cell_to_tokens[best_idx].append(token)

        cell_texts = {}
        for c_idx, tokens in cell_to_tokens.items():
            if not tokens:
                continue
            # Sort top-to-bottom, then left-to-right
            tokens.sort(key=lambda t: (t["center"][1], t["center"][0]))
            lines = []
            curr_line = []
            curr_y = None
            for t in tokens:
                t_y = t["center"][1]
                t_h = max(8.0, t["bbox"][3] - t["bbox"][1])
                if curr_y is None or abs(t_y - curr_y) < (t_h * 0.5):
                    curr_line.append(t)
                    curr_y = t_y if curr_y is None else (curr_y + t_y) / 2.0
                else:
                    curr_line.sort(key=lambda x: x["bbox"][0])
                    lines.append(" ".join(x["text"] for x in curr_line))
                    curr_line = [t]
                    curr_y = t_y
            if curr_line:
                curr_line.sort(key=lambda x: x["bbox"][0])
                lines.append(" ".join(x["text"] for x in curr_line))
            cell_texts[c_idx] = " ".join(lines).strip()

        # Cell OCR fallback for empty cells (with dual-polarity support for dark/colored cells)
        if self.use_tesseract and img_bgr is not None:
            import pytesseract
            import cv2
            h, w = img_bgr.shape[:2]
            for c_idx, norm_b in enumerate(cell_boxes):
                if not cell_texts.get(c_idx) and norm_b is not None:
                    bx1, by1, bx2, by2 = norm_b
                    cx1, cy1 = max(0, int(bx1) - 2), max(0, int(by1) - 2)
                    cx2, cy2 = min(w, int(bx2) + 2), min(h, int(by2) + 2)
                    if (cx2 - cx1) > 8 and (cy2 - cy1) > 8:
                        crop = img_bgr[cy1:cy2, cx1:cx2]
                        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        # Skip blank/solid background cells to save latency and avoid noise
                        if float(np.std(crop_gray)) < 7.0 or int(np.ptp(crop_gray)) < 20:
                            continue
                        cw, ch = crop.shape[1], crop.shape[0]
                        crop_scaled = cv2.resize(crop, (max(cw * 3, 60), max(ch * 3, 30)), interpolation=cv2.INTER_CUBIC)
                        rgb_crop = cv2.cvtColor(crop_scaled, cv2.COLOR_BGR2RGB)
                        txt = pytesseract.image_to_string(rgb_crop, config="--psm 7").strip()
                        if not txt:
                            txt = pytesseract.image_to_string(255 - rgb_crop, config="--psm 7").strip()
                        if txt:
                            import re
                            if not re.fullmatch(r"[~—–`'\"|_+=^.\-,;:!?\s]+", txt):
                                cell_texts[c_idx] = txt

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
                elif tag_lower in ("</td>", "</th>"):
                    i += 1
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
            logger.warning("SLANet structure generation failed or returned no text. Triggering OCR geometric clustering fallback.")
            sorted_ocr = sorted(ocr_tokens, key=lambda t: t["bbox"][1])
            raw_rows = []
            curr_row = []
            curr_y = None
            for t in sorted_ocr:
                y_mid = t["center"][1]
                t_h = max(10.0, t["bbox"][3] - t["bbox"][1])
                if curr_y is None or abs(y_mid - curr_y) < (t_h * 0.65):
                    curr_row.append(t)
                    curr_y = y_mid if curr_y is None else (curr_y + y_mid) / 2.0
                else:
                    curr_row.sort(key=lambda item: item["bbox"][0])
                    raw_rows.append(curr_row)
                    curr_row = [t]
                    curr_y = y_mid
            if curr_row:
                curr_row.sort(key=lambda item: item["bbox"][0])
                raw_rows.append(curr_row)

            # Pre-merge horizontally adjacent tokens into phrase blocks within each row
            phrase_rows = []
            for row in raw_rows:
                phrases = []
                for t in row:
                    if not phrases:
                        phrases.append({"bbox": list(t["bbox"]), "text": t["text"]})
                    else:
                        prev = phrases[-1]
                        prev_h = max(10.0, prev["bbox"][3] - prev["bbox"][1])
                        gap = t["bbox"][0] - prev["bbox"][2]
                        if gap < max(24.0, prev_h * 1.5):
                            prev["bbox"][2] = max(prev["bbox"][2], t["bbox"][2])
                            prev["bbox"][3] = max(prev["bbox"][3], t["bbox"][3])
                            prev["text"] += " " + t["text"]
                        else:
                            phrases.append({"bbox": list(t["bbox"]), "text": t["text"]})
                phrase_rows.append(phrases)

            # Cluster column start boundaries across rows
            all_starts = sorted([p["bbox"][0] for row in phrase_rows for p in row])
            col_lefts = []
            if all_starts:
                curr_cluster = [all_starts[0]]
                for x in all_starts[1:]:
                    if x - (sum(curr_cluster) / len(curr_cluster)) < 60.0:
                        curr_cluster.append(x)
                    else:
                        col_lefts.append(sum(curr_cluster) / len(curr_cluster))
                        curr_cluster = [x]
                if curr_cluster:
                    col_lefts.append(sum(curr_cluster) / len(curr_cluster))

            col_bounds = []
            for idx, c_left in enumerate(col_lefts):
                next_left = col_lefts[idx + 1] if idx + 1 < len(col_lefts) else float("inf")
                col_bounds.append((c_left, next_left))

            table_rows_html = []
            for row in phrase_rows:
                col_buckets = {i: [] for i in range(len(col_bounds))}
                for p in row:
                    px = p["bbox"][0]
                    best_col = 0
                    min_dist = float("inf")
                    for c_idx, c_left in enumerate(col_lefts):
                        dist = abs(px - c_left)
                        if dist < min_dist:
                            min_dist = dist
                            best_col = c_idx
                    col_buckets[best_col].append(p["text"])

                row_tds = []
                for c_idx in range(len(col_bounds)):
                    texts = col_buckets[c_idx]
                    cell_str = " ".join(texts).strip()
                    row_tds.append(f"<td>{cell_str}</td>")
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

        # Ensure minimum resolution for small table crops while preserving text stroke fidelity
        try:
            import cv2
            h, w = img_bgr.shape[:2]
            max_edge = max(h, w)
            if 0 < max_edge < 600:
                scale = 800.0 / max_edge
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            logger.debug(f"Could not resize table crop: {e}")

        try:
            # 1. OCR text detection on table crop
            ocr_tokens = self._extract_ocr_tokens(img_bgr)

            # 2. SLANet table structure inference
            pred_structures, pred_bboxes = self._extract_slanet_structure(img_bgr)

            # 3. Match cells and construct HTML table
            html_str = self._match_cells_and_build_html(pred_structures, pred_bboxes, ocr_tokens, img_bgr=img_bgr)

            # 4. Convert HTML table to Markdown
            markdown_str = html_table_to_markdown(html_str)

            return {
                "html": html_str or "",
                "markdown": markdown_str or "",
                "cell_bboxes": pred_bboxes or [],
            }
        except Exception as e:
            logger.exception("Error extracting table structure")
            return {"html": "", "markdown": ""}
