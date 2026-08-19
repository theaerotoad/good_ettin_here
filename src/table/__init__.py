"""
Table Recognition module using RapidTable (SLANet ONNX) and Markdown conversion.
"""

from .converter import html_table_to_markdown
from .engine import TableRecognizer, TableResult

__all__ = [
    "TableRecognizer",
    "TableResult",
    "html_table_to_markdown",
]
