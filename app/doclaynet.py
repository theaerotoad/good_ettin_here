import os
import io
import json
import base64
import logging
import requests
import numpy as np
import onnxruntime as ort
from PIL import Image

from .table import TableRecognizerONNX

logger = logging.getLogger("ettin-reranker")


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

        self.use_tesseract = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.use_tesseract = True
            logger.info("Initialized Pytesseract for highly accurate text region extraction.")
        except Exception:
            logger.info("Pytesseract not found. Install it (`apt install tesseract-ocr` & `pip install pytesseract`) for better word spacing.")
            pass

        self.ocr_engine = None
        if not self.use_tesseract:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self.ocr_engine = RapidOCR(use_cuda=use_gpu)
                logger.info("Initialized RapidOCR engine for DocLayNet text extraction.")
            except ImportError:
                pass

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

    def _safe_load_pil(self, source) -> Image.Image:
        """Helper to load an image with Pillow and a fallback to ImageMagick for WMF/EPS."""
        import tempfile
        import subprocess
        try:
            if isinstance(source, bytes):
                return Image.open(io.BytesIO(source)).convert("RGB")
            elif isinstance(source, io.BytesIO):
                source.seek(0)
                return Image.open(source).convert("RGB")
            else:
                return Image.open(source).convert("RGB")
        except Exception as e:
            logger.debug(f"Pillow load failed: {e}. Falling back to ImageMagick.")
            with tempfile.TemporaryDirectory() as tmpdir:
                if isinstance(source, (bytes, bytearray)):
                    in_path = os.path.join(tmpdir, "input.tmp")
                    with open(in_path, "wb") as f:
                        f.write(source)
                elif isinstance(source, io.BytesIO):
                    in_path = os.path.join(tmpdir, "input.tmp")
                    with open(in_path, "wb") as f:
                        f.write(source.getvalue())
                else:
                    in_path = str(source)
                    
                out_path = os.path.join(tmpdir, "output.png")
                success = False
                for cmd in ["magick", "convert"]:
                    try:
                        subprocess.run(
                            [cmd, "-density", "300", in_path, out_path],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        success = True
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                
                if not success:
                    raise ValueError(f"ImageMagick fallback failed. Original Pillow error: {e}")
                
                if os.path.exists(out_path):
                    final_path = out_path
                elif os.path.exists(os.path.join(tmpdir, "output-0.png")):
                    final_path = os.path.join(tmpdir, "output-0.png")
                else:
                    raise ValueError(f"ImageMagick fallback produced no output. Original Pillow error: {e}")
                    
                with Image.open(final_path) as tmp_img:
                    return tmp_img.convert("RGB").copy()

    def _load_image(self, image_input) -> Image.Image:
        """Loads and converts image inputs (PIL Image, bytes, base64 data URI, HTTP URL, or local path) to RGB."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, bytes):
            return self._safe_load_pil(image_input)
        if isinstance(image_input, str):
            if image_input.startswith("data:image"):
                base64_data = image_input.split(",", 1)[1]
                return self._safe_load_pil(base64.b64decode(base64_data))
            if image_input.startswith("http://") or image_input.startswith("https://"):
                resp = requests.get(image_input, timeout=15)
                resp.raise_for_status()
                return self._safe_load_pil(resp.content)
            if os.path.exists(image_input):
                return self._safe_load_pil(image_input)
            try:
                raw_bytes = base64.b64decode(image_input)
                return self._safe_load_pil(raw_bytes)
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
        extract_text: bool = True,
    ) -> tuple[list[dict], tuple[int, int]]:
        """
        Detect document layout elements in an image.
        If extract_tables=True, automatically parses regions labeled 'Table' into HTML/Markdown.
        If extract_text=True, runs OCR and maps text to text-like layout regions in reading order.
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

        raw_detections = []
        for idx in keep_indices:
            cid = int(valid_class_ids[idx])
            label_name = self.labels[cid] if cid < len(self.labels) else f"class_{cid}"
            raw_detections.append({
                "bbox": [
                    round(float(boxes_xyxy[idx, 0]), 2),
                    round(float(boxes_xyxy[idx, 1]), 2),
                    round(float(boxes_xyxy[idx, 2]), 2),
                    round(float(boxes_xyxy[idx, 3]), 2),
                ],
                "confidence": round(float(valid_scores[idx]), 4),
                "class_id": cid,
                "label": label_name,
            })

        # Class-agnostic deduplication: remove lower confidence boxes that heavily overlap
        # across different classes (e.g., a Caption and Section-header predicted on the exact same text).
        # We prioritize Tables so they are never accidentally suppressed by a text region.
        raw_detections.sort(key=lambda x: (1 if x["label"].lower() == "table" else 0, x["confidence"]), reverse=True)
        
        detections = []
        for d in raw_detections:
            bx1, by1, bx2, by2 = d["bbox"]
            b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
            is_dup = False
            for keep_d in detections:
                kx1, ky1, kx2, ky2 = keep_d["bbox"]
                k_area = max(0, kx2 - kx1) * max(0, ky2 - ky1)
                
                ix1, iy1 = max(bx1, kx1), max(by1, ky1)
                ix2, iy2 = min(bx2, kx2), min(by2, ky2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                
                if min(b_area, k_area) > 0:
                    ioma = inter / min(b_area, k_area)
                    if ioma > 0.85:  # If 85% of the smaller box is covered, consider it a duplicate
                        is_dup = True
                        break
            
            if not is_dup:
                detections.append(d)

        # Extract Tables
        for d in detections:
            if extract_tables and self.table_recognizer is not None and d["label"].lower() == "table":
                bx1, by1, bx2, by2 = d["bbox"]
                cx1 = max(0, int(bx1) - 4)
                cy1 = max(0, int(by1) - 4)
                cx2 = min(orig_w, int(bx2) + 4)
                cy2 = min(orig_h, int(by2) + 4)
                if cx2 > cx1 and cy2 > cy1:
                    crop = img.crop((cx1, cy1, cx2, cy2))
                    table_data = self.table_recognizer.extract(crop)
                    if table_data.get("html"):
                        d["html"] = table_data["html"]
                    if table_data.get("markdown"):
                        d["markdown"] = table_data["markdown"]

        # Sort detections in reading order (top-to-bottom, left-to-right)
        # We use a 20-pixel vertical bucket to group items on the same visual line
        detections.sort(key=lambda d: (int(d["bbox"][1] / 20.0), d["bbox"][0]))

        # Extract text for text-like elements (including Pictures) if requested
        text_labels = {"Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture", "Section-header", "Text", "Title"}
        if extract_text and (self.use_tesseract or self.ocr_engine is not None):
            needs_ocr = any(d["label"] in text_labels for d in detections)
            
            # Tesseract Path (Superior word spacing and formatting for Latin text)
            if needs_ocr and self.use_tesseract:
                import pytesseract
                from PIL import ImageOps
                for det in detections:
                    if det["label"] in text_labels:
                        bx1, by1, bx2, by2 = det["bbox"]
                        cx1, cy1 = max(0, int(bx1)), max(0, int(by1))
                        cx2, cy2 = min(orig_w, int(bx2)), min(orig_h, int(by2))
                        
                        if cx2 > cx1 and cy2 > cy1:
                            crop = img.crop((cx1, cy1, cx2, cy2))
                            
                            # Invert crop if it's predominantly dark (white text on dark background)
                            gray = np.dot(np.array(crop)[..., :3], [0.114, 0.587, 0.299])
                            if np.sum(gray < 85) / max(1, gray.size) > 0.6:
                                crop = ImageOps.invert(crop)
                            
                            # Upscale slightly for Tesseract clarity
                            cw, ch = crop.size
                            if max(cw, ch) < 800:
                                crop = crop.resize((cw * 2, ch * 2), Image.Resampling.BICUBIC)
                            
                            try:
                                # psm 6 assumes a single uniform block of text
                                text = pytesseract.image_to_string(crop, config='--psm 6').strip()
                                if text:
                                    if det["label"] == "Picture":
                                        text = text.replace('\n', '\\n')
                                    det["text"] = text
                            except Exception as e:
                                logger.debug(f"Tesseract extraction failed: {e}")

            # RapidOCR Fallback Path (Portable, no system binaries)
            elif needs_ocr and self.ocr_engine is not None:
                try:
                    img_bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
                    
                    # Heuristic to skip unnecessary OCR passes based on region polarity
                    gray = np.dot(img_bgr[..., :3], [0.114, 0.587, 0.299])
                    run_normal = False
                    run_negated = False
                    
                    for det in detections:
                        if det["label"] in text_labels:
                            bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
                            bx1, by1 = max(0, bx1), max(0, by1)
                            bx2, by2 = min(gray.shape[1], bx2), min(gray.shape[0], by2)
                            
                            if bx2 > bx1 and by2 > by1:
                                crop = gray[by1:by2, bx1:bx2]
                                total_px = crop.size
                                if total_px > 0:
                                    light_pct = np.sum(crop > 170) / total_px
                                    dark_pct = np.sum(crop < 85) / total_px
                                    if light_pct < 0.85:
                                        run_negated = True
                                    if dark_pct < 0.85:
                                        run_normal = True
                                        
                    # Fallback if no valid regions triggered it
                    if not run_normal and not run_negated:
                        run_normal = True

                    # Upscale image for OCR to improve space character detection
                    # (PaddleOCR/RapidOCR natively struggles with spaces on small text)
                    scale_factor = 1.0
                    h, w = img_bgr.shape[:2]
                    if max(h, w) < 1600:
                        scale_factor = 2.0
                        import cv2
                        img_bgr_ocr = cv2.resize(img_bgr, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
                    else:
                        img_bgr_ocr = img_bgr

                    def run_ocr_pass(image_array):
                        out = self.ocr_engine(image_array)
                        raw = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
                        toks = []
                        if raw and isinstance(raw, list):
                            for item in raw:
                                if not item or len(item) < 2: continue
                                box_pts = item[0]
                                text_val = item[1]
                                text_str = str(text_val[0]) if isinstance(text_val, (list, tuple)) else str(text_val)
                                if not text_str.strip(): continue
                                
                                pts = np.asarray(box_pts)
                                if pts.ndim == 2 and len(pts) >= 4:
                                    x1, y1 = float(np.min(pts[:, 0])) / scale_factor, float(np.min(pts[:, 1])) / scale_factor
                                    x2, y2 = float(np.max(pts[:, 0])) / scale_factor, float(np.max(pts[:, 1])) / scale_factor
                                elif len(box_pts) == 4 and not isinstance(box_pts[0], (list, tuple, np.ndarray)):
                                    x1, y1, x2, y2 = float(box_pts[0]) / scale_factor, float(box_pts[1]) / scale_factor, float(box_pts[2]) / scale_factor, float(box_pts[3]) / scale_factor
                                else:
                                    continue
                                    
                                # Heuristic cleanup for missing spaces (e.g. after punctuation)
                                import re
                                text_clean = re.sub(r'([.,:;!?])([A-Za-z])', r'\1 \2', text_str.strip())
                                text_clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', text_clean)
                                
                                toks.append({
                                    "bbox": [x1, y1, x2, y2],
                                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                                    "text": text_clean,
                                    "used": False
                                })
                        return toks

                    normal_tokens = run_ocr_pass(img_bgr_ocr) if run_normal else []
                    negated_tokens = run_ocr_pass(255 - img_bgr_ocr) if run_negated else []
                    
                    # Merge and deduplicate tokens (Spatial NMS based on Intersection over Min Area)
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
                    
                    # Map OCR tokens to detected layout regions
                    for det in detections:
                        if det["label"] in text_labels:
                            bx1, by1, bx2, by2 = det["bbox"]
                            region_tokens = []
                            for t in ocr_tokens:
                                if t["used"]: continue
                                cx, cy = t["center"]
                                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                                    region_tokens.append(t)
                                    t["used"] = True
                            
                            if region_tokens:
                                # Sort tokens top-to-bottom, left-to-right within the region
                                region_tokens.sort(key=lambda t: (int(t["bbox"][1] / 10.0), t["bbox"][0]))
                                
                                lines = []
                                curr_line = []
                                curr_y = None
                                for t in region_tokens:
                                    y_mid = t["center"][1]
                                    if curr_y is None or abs(y_mid - curr_y) < 12.0:
                                        curr_line.append(t["text"])
                                        curr_y = y_mid if curr_y is None else (curr_y + y_mid) / 2.0
                                    else:
                                        lines.append(" ".join(curr_line))
                                        curr_line = [t["text"]]
                                        curr_y = y_mid
                                if curr_line:
                                    lines.append(" ".join(curr_line))

                                if det["label"] == "Picture":
                                    det["text"] = "\\n".join(lines)
                                else:
                                    det["text"] = "\n".join(lines)
                except Exception as e:
                    logger.warning(f"Text extraction failed during layout analysis: {e}")

        return detections, (orig_w, orig_h)
