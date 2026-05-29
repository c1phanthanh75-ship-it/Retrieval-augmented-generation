"""
Data models for RAG indexing pipeline
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from pathlib import Path
import uuid


class RegionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FORMULA = "formula"
    IMAGE = "image"
    TITLE = "title"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"


class ChunkType(str, Enum):
    TEXT = "text"
    FORMULA = "formula"
    TABLE = "table"
    IMAGE_CAPTION = "image_caption"
    MIXED = "mixed"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    page: int = 0

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class DocumentRegion:
    """A detected region from OCR / layout analysis"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    region_type: RegionType = RegionType.TEXT
    text: str = ""
    raw_text: str = ""          # original OCR text before cleanup
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None
    page_num: int = 0
    image_data: Optional[bytes] = None   # for formula/image regions
    metadata: dict = field(default_factory=dict)


@dataclass
class FormulaResult:
    """Parsed formula with semantic description"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    latex_source: str = ""
    semantic_description: str = ""
    sympy_expr: Optional[str] = None    # simplified symbolic expression
    variables: list[str] = field(default_factory=list)
    region_id: str = ""
    confidence: float = 1.0


@dataclass
class TableResult:
    """Parsed table"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    markdown: str = ""
    html: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    region_id: str = ""
    semantic_description: str = ""


@dataclass
class DocumentChunk:
    """Final chunk ready for indexing"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    doc_path: str = ""
    chunk_type: ChunkType = ChunkType.TEXT
    content: str = ""                           # main text content
    semantic_content: str = ""                  # enriched content for embedding
    page_num: int = 0
    chunk_index: int = 0
    parent_chunk_id: Optional[str] = None       # for hierarchical chunking
    formula: Optional[FormulaResult] = None
    table: Optional[TableResult] = None
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None     # dense vector


@dataclass
class IndexedDocument:
    """Result after full indexing of one document"""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_path: str = ""
    total_pages: int = 0
    total_chunks: int = 0
    chunks: list[DocumentChunk] = field(default_factory=list)
    regions: list[DocumentRegion] = field(default_factory=list)
    formulas: list[FormulaResult] = field(default_factory=list)
    tables: list[TableResult] = field(default_factory=list)
    processing_time_s: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from hybrid search"""
    chunk: DocumentChunk
    colbert_score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rank: int = 0
