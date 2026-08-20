"""
RapidTable / SLANet ONNX Table Recognizer Engine.
Extracts tabular data from images using ONNX runtime and formats into HTML/Markdown.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from rapid_table import RapidTable
from rapidocr_onnxruntime import RapidOCR

from .converter import html_table_to_markdown


@dataclass
class TableResult:
    """Container for table recognition outputs."""
    html: str
    markdown: str
    elapse: float
    cell_bboxes: Optional[List[List[float]]] = field(default_factory=list)
    raw_ocr_result: Optional[List[Any]] = field(default_factory=list)


class TableRecognizer:
    """
    Extracts tabular structures and cell text from images using SLANet (ONNX)
    and RapidOCR, with support for HTML and Markdown outputs.
    """

    def __init__(
        self,
        table_model_path: Optional[Union[str, Path]] = None,
        ocr_model_dir: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
    ):
        """
        Initialize RapidTable and RapidOCR engines.

        Args:
            table_model_path: Optional custom path to the SLANet ONNX model.
            ocr_model_dir: Optional custom directory for OCR ONNX models.
            use_cuda: Whether to use CUDA execution provider for ONNX runtime.
        """
        table_kwargs: Dict[str, Any] = {}
        if table_model_path:
            table_kwargs["model_path"] = str(table_model_path)
        if use_cuda:
            table_kwargs["use_cuda"] = True

        ocr_kwargs: Dict[str, Any] = {}
        if ocr_model_dir:
            ocr_kwargs["model_dir"] = str(ocr_model_dir)
        if use_cuda:
            ocr_kwargs["use_cuda"] = True

        self.ocr_engine = RapidOCR(**ocr_kwargs)
        self.table_engine = RapidTable(**table_kwargs)

    def _safe_load_bgr(self, source) -> np.ndarray:
        """Helper to load image via Pillow with ImageMagick fallback, then return BGR numpy array."""
        import tempfile
        import subprocess
        import io
        img = None
        try:
            if isinstance(source, bytes):
                img = Image.open(io.BytesIO(source))
            else:
                img = Image.open(source)
            rgb_arr = np.array(img.convert("RGB"))
            return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception as e:
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
                    rgb_arr = np.array(tmp_img.convert("RGB"))
                    return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    def _prepare_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Converts different image inputs into a standard BGR numpy array.
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image not found at path: {path_str}")
            image = cv2.imread(path_str)
            if image is None:
                # cv2.imread fails on EPS/WMF natively, apply fallback
                return self._safe_load_bgr(path_str)
            return image

        if isinstance(image_input, bytes):
            return self._safe_load_bgr(image_input)

        if isinstance(image_input, Image.Image):
            try:
                rgb_image = np.array(image_input.convert("RGB"))
            except Exception:
                raise ValueError("Failed to convert PIL Image to RGB")
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def extract(
        self,
        image: Union[str, Path, bytes, np.ndarray, Image.Image],
    ) -> TableResult:
        """
        Performs OCR text recognition and SLANet structure extraction on the image.

        Args:
            image: Image path, PIL Image, or OpenCV/numpy image array.

        Returns:
            TableResult containing HTML, Markdown, bounding boxes, and latency.
        """
        img_bgr = self._prepare_image(image)

        # 1. Run OCR on the full table image to locate text tokens
        ocr_result, _ = self.ocr_engine(img_bgr)

        # 2. Run SLANet Table Recognition via RapidTable
        table_html_str, table_cell_bboxes, elapse = self.table_engine(img_bgr, ocr_result)

        # 3. Convert generated HTML table structure to Markdown
        markdown_str = html_table_to_markdown(table_html_str)

        return TableResult(
            html=table_html_str,
            markdown=markdown_str,
            elapse=elapse,
            cell_bboxes=table_cell_bboxes,
            raw_ocr_result=ocr_result,
        )
