"""
Facade module for backward compatibility.
"""
import logging

logger = logging.getLogger("ettin-reranker")
logging.basicConfig(level=logging.INFO)

from .utils import _gelu_numpy, html_table_to_markdown, _get_tensor
from .table import TableRecognizerONNX
from .reranker import EttinONNXReranker
from .embedding import EmbeddingGemmaONNX
from .doclaynet import DocLayNetONNX

__all__ = [
    "_gelu_numpy",
    "html_table_to_markdown",
    "_get_tensor",
    "TableRecognizerONNX",
    "EttinONNXReranker",
    "EmbeddingGemmaONNX",
    "DocLayNetONNX",
]
