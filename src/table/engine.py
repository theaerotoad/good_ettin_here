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

    def _prepare_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Converts different image inputs into a standard BGR numpy array.
        """
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image not found at path: {path_str}")
            image = cv2.imread(path_str)
            if image is None:
                raise ValueError(f"Failed to read image from path: {path_str}")
            return image

        if isinstance(image_input, Image.Image):
            rgb_image = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def extract(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
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
