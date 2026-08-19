import os
import logging
import numpy as np
from PIL import Image

from .utils import html_table_to_markdown

logger = logging.getLogger("ettin-reranker")


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
        """Runs 2-pass OCR (normal + negated) on image crop and returns deduplicated text boxes."""
        if self.ocr_engine is None:
            return []

        def run_pass(img_array):
            toks = []
            try:
                ocr_out = self.ocr_engine(img_array)
                raw_boxes = ocr_out[0] if isinstance(ocr_out, (list, tuple)) and len(ocr_out) > 0 else ocr_out
                if not raw_boxes or not isinstance(raw_boxes, list):
                    return toks

                for item in raw_boxes:
                    if not item or len(item) < 2:
                        continue

                    box_pts = item[0]
                    text_val = item[1]
                    text_str = str(text_val[0]) if isinstance(text_val, (list, tuple)) else str(text_val)
                    text_str = text_str.strip()
                    if not text_str:
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

                    # Heuristic cleanup for missing spaces
                    import re
                    text_clean = re.sub(r'([.,:;!?])([A-Za-z])', r'\1 \2', text_str)
                    text_clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_clean)

                    toks.append({
                        "bbox": [x1, y1, x2, y2],
                        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                        "text": text_clean,
                    })
            except Exception as e:
                logger.debug(f"OCR pass failed: {e}")
            return toks

        # Heuristic to skip unnecessary OCR passes based on crop polarity
        gray = np.dot(img_bgr[..., :3], [0.114, 0.587, 0.299])
        total_px = gray.size
        run_normal = True
        run_negated = True

        if total_px > 0:
            light_pct = np.sum(gray > 170) / total_px
            dark_pct = np.sum(gray < 85) / total_px
            if light_pct > 0.85:
                run_negated = False
            elif dark_pct > 0.85:
                run_normal = False

        normal_tokens = run_pass(img_bgr) if run_normal else []
        negated_tokens = run_pass(255 - img_bgr) if run_negated else []

        # Merge and deduplicate tokens (Intersection over Min Area)
        ocr_tokens = list(normal_tokens)
        for n_tok in negated_tokens:
            is_dup = False
            nx1, ny1, nx2, ny2 = n_tok["bbox"]
            n_area = max(0, nx2 - nx1) * max(0, ny2 - ny1)

            for m_tok in ocr_tokens:
                mx1, my1, mx2, my2 = m_tok["bbox"]
                m_area = max(0, mx2 - mx1) * max(0, my2 - my1)

                ix1, iy1 = max(nx1, mx1), max(ny1, my1)
                ix2, iy2 = min(nx2, mx2), min(ny2, my2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

                # If the overlap is significant relative to the smaller bounding box
                if inter / (min(n_area, m_area) + 1e-9) > 0.4:
                    is_dup = True
                    # Favor the longer string (cleaner OCR read on high contrast)
                    if len(n_tok["text"]) > len(m_tok["text"]) + 2:
                        m_tok["text"] = n_tok["text"]
                    break

            if not is_dup:
                ocr_tokens.append(n_tok)

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
