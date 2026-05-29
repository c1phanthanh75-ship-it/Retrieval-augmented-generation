"""
Advanced Chunking Module
- Semantic splitting (sentence boundary aware)
- Hierarchical chunks (parent/child)
- Isolated formula & table chunks
- Sliding window with overlap
"""
import logging
import re
import uuid
from typing import Optional

from .config import ChunkingConfig
from .models import (
    DocumentRegion, DocumentChunk, FormulaResult, TableResult,
    RegionType, ChunkType, IndexedDocument
)

logger = logging.getLogger(__name__)


class AdvancedChunker:
    """
    Converts DocumentRegions + FormulaResults + TableResults
    into final DocumentChunks ready for indexing.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._token_cache: dict[int, int] = {}  # hash → token count

    def count_tokens(self, text: str) -> int:
        """
        Fast token count — dùng word count / 0.75 (thay tiktoken).
        tiktoken gọi hàng nghìn lần → chậm, không cần thiết cho chunking.
        """
        # Cache bằng hash để tránh tính lại cùng 1 string
        h = hash(text)
        if h in self._token_cache:
            return self._token_cache[h]
        # ~0.75 words per token (English average)
        count = max(1, len(text.split()))
        self._token_cache[h] = count
        # Giới hạn cache size
        if len(self._token_cache) > 10000:
            self._token_cache.clear()
        return count

    # ------------------------------------------------------------------ #
    # Main entry point                                                      #
    # ------------------------------------------------------------------ #

    def chunk_document(
        self,
        doc_id: str,
        doc_path: str,
        regions: list[DocumentRegion],
        formulas: list[FormulaResult],
        tables: list[TableResult],
    ) -> list[DocumentChunk]:
        """
        Build all chunks for one document.
        Returns flat list of DocumentChunk objects.
        """
        chunks: list[DocumentChunk] = []
        formula_map = {f.region_id: f for f in formulas}
        table_map = {t.region_id: t for t in tables}

        # Sort regions by page then vertical position
        sorted_regions = sorted(
            regions,
            key=lambda r: (r.page_num, r.bbox.y1 if r.bbox else 0)
        )

        # Group regions by page for context
        page_groups: dict[int, list[DocumentRegion]] = {}
        for r in sorted_regions:
            page_groups.setdefault(r.page_num, []).append(r)

        chunk_index = 0

        for page_num in sorted(page_groups.keys()):
            page_regions = page_groups[page_num]
            text_buffer: list[str] = []
            buffer_region_ids: list[str] = []

            for region in page_regions:

                # --- Isolated formula chunk ---
                if self.config.isolate_formulas and region.region_type == RegionType.FORMULA:
                    if text_buffer:
                        new_chunks = self._flush_text_buffer(
                            text_buffer, doc_id, doc_path, page_num, chunk_index
                        )
                        chunks.extend(new_chunks)
                        chunk_index += len(new_chunks)
                        text_buffer, buffer_region_ids = [], []

                    formula = formula_map.get(region.id)
                    if formula:
                        fc = self._make_formula_chunk(
                            formula, doc_id, doc_path, page_num, chunk_index
                        )
                        chunks.append(fc)
                        chunk_index += 1

                # --- Isolated table chunk ---
                elif self.config.isolate_tables and region.region_type == RegionType.TABLE:
                    if text_buffer:
                        new_chunks = self._flush_text_buffer(
                            text_buffer, doc_id, doc_path, page_num, chunk_index
                        )
                        chunks.extend(new_chunks)
                        chunk_index += len(new_chunks)
                        text_buffer, buffer_region_ids = [], []

                    table = table_map.get(region.id)
                    if table:
                        tc = self._make_table_chunk(
                            table, doc_id, doc_path, page_num, chunk_index
                        )
                        chunks.append(tc)
                        chunk_index += 1
                    elif region.text:
                        tc = self._make_text_chunk(
                            region.text, doc_id, doc_path, page_num, chunk_index
                        )
                        chunks.append(tc)
                        chunk_index += 1

                # --- Image / caption ---
                elif region.region_type in (RegionType.IMAGE, RegionType.CAPTION):
                    if region.text:
                        text_buffer.append(region.text)

                # --- Normal text ---
                elif region.text and region.region_type in (
                    RegionType.TEXT, RegionType.TITLE,
                    RegionType.HEADER, RegionType.FOOTER
                ):
                    text_buffer.append(region.text)

            # Flush remaining text
            if text_buffer:
                new_chunks = self._flush_text_buffer(
                    text_buffer, doc_id, doc_path, page_num, chunk_index
                )
                chunks.extend(new_chunks)
                chunk_index += len(new_chunks)

            # Free memory every 50 pages
            if page_num % 50 == 0 and page_num > 0:
                import gc
                gc.collect()
                logger.debug(f"  GC collect at page {page_num}, chunks so far: {len(chunks)}")

        # Add parent chunks only if doc is small enough (< 1000 chunks)
        if len(chunks) < 1000:
            chunks = self._add_parent_chunks(chunks, doc_id, doc_path)
        else:
            logger.info(f"Skipping parent chunks (doc too large: {len(chunks)} chunks)")

        logger.info(
            f"Chunking complete: {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c.chunk_type == ChunkType.FORMULA)} formulas, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE)} tables)"
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Text buffer → sliding window chunks (FAST char-based)               #
    # ------------------------------------------------------------------ #

    def _flush_text_buffer(
        self,
        text_parts: list[str],
        doc_id: str,
        doc_path: str,
        page_num: int,
        start_index: int,
    ) -> list[DocumentChunk]:
        """Join text buffer và chunk nhanh theo char (không sentence split)."""
        full_text = " ".join(text_parts)
        full_text = full_text.strip()
        if not full_text:
            return []
        return self._fast_char_chunks(full_text, doc_id, doc_path, page_num, start_index)

    def _fast_char_chunks(
        self,
        text: str,
        doc_id: str,
        doc_path: str,
        page_num: int,
        start_index: int,
    ) -> list[DocumentChunk]:
        """
        Chunk text bằng char offset — O(n), không loop per-sentence.
        chunk_size & chunk_overlap tính theo words (1 word ≈ 1 token).
        """
        # Tách thành words một lần
        words = text.split()
        if not words:
            return []

        chunk_size = self.config.chunk_size        # words
        overlap = self.config.chunk_overlap        # words
        min_size = self.config.min_chunk_size      # words

        chunks = []
        start = 0
        total = len(words)

        while start < total:
            end = min(start + chunk_size, total)
            chunk_words = words[start:end]

            if len(chunk_words) >= min_size:
                chunk_text = " ".join(chunk_words)
                chunks.append(self._make_text_chunk(
                    chunk_text, doc_id, doc_path, page_num,
                    start_index + len(chunks)
                ))

            # Slide forward, giữ overlap
            if end >= total:
                break
            start = end - overlap

        return chunks

    def _sentences_to_chunks(
        self,
        sentences: list[str],
        doc_id: str,
        doc_path: str,
        page_num: int,
        start_index: int,
    ) -> list[DocumentChunk]:
        """Legacy — không còn dùng, giữ để tương thích."""
        return self._fast_char_chunks(
            " ".join(sentences), doc_id, doc_path, page_num, start_index
        )

    def _split_sentences(self, text: str) -> list[str]:
        """Legacy — không còn dùng."""
        return text.split()

    # ------------------------------------------------------------------ #
    # Chunk factories                                                       #
    # ------------------------------------------------------------------ #

    def _make_text_chunk(
        self, text: str, doc_id: str, doc_path: str, page_num: int, index: int
    ) -> DocumentChunk:
        return DocumentChunk(
            doc_id=doc_id,
            doc_path=doc_path,
            chunk_type=ChunkType.TEXT,
            content=text,
            semantic_content=text,
            page_num=page_num,
            chunk_index=index,
            metadata={"token_count": self.count_tokens(text)},
        )

    def _make_formula_chunk(
        self,
        formula: FormulaResult,
        doc_id: str,
        doc_path: str,
        page_num: int,
        index: int,
    ) -> DocumentChunk:
        # Semantic content = description + latex (rich for retrieval)
        semantic = formula.semantic_description or f"Formula: {formula.latex_source}"
        if formula.variables:
            semantic += f" Variables: {', '.join(formula.variables)}"
        semantic += f"\nLaTeX: {formula.latex_source}"

        return DocumentChunk(
            doc_id=doc_id,
            doc_path=doc_path,
            chunk_type=ChunkType.FORMULA,
            content=formula.latex_source,
            semantic_content=semantic,
            page_num=page_num,
            chunk_index=index,
            formula=formula,
            metadata={
                "latex": formula.latex_source,
                "variables": formula.variables,
                "has_sympy": formula.sympy_expr is not None,
            },
        )

    def _make_table_chunk(
        self,
        table: TableResult,
        doc_id: str,
        doc_path: str,
        page_num: int,
        index: int,
    ) -> DocumentChunk:
        content = table.markdown or table.html
        semantic = content
        if table.semantic_description:
            semantic = f"{table.semantic_description}\n\n{content}"

        return DocumentChunk(
            doc_id=doc_id,
            doc_path=doc_path,
            chunk_type=ChunkType.TABLE,
            content=content,
            semantic_content=semantic,
            page_num=page_num,
            chunk_index=index,
            table=table,
            metadata={
                "headers": table.headers,
                "rows": len(table.rows),
            },
        )

    # ------------------------------------------------------------------ #
    # Hierarchical chunking (parent-child)                                  #
    # ------------------------------------------------------------------ #

    def _add_parent_chunks(
        self,
        chunks: list[DocumentChunk],
        doc_id: str,
        doc_path: str,
    ) -> list[DocumentChunk]:
        """
        Create larger parent chunks (2x size) that child chunks can reference.
        Used for "small-to-big" retrieval: retrieve small, return big context.
        """
        parent_size = self.config.chunk_size * 2
        text_chunks = [c for c in chunks if c.chunk_type == ChunkType.TEXT]
        parents = []
        current_text = []
        current_tokens = 0
        child_ids = []

        for chunk in text_chunks:
            tok = self.count_tokens(chunk.content)
            if current_tokens + tok <= parent_size:
                current_text.append(chunk.content)
                current_tokens += tok
                child_ids.append(chunk.id)
            else:
                if current_text:
                    parent = self._make_parent(
                        " ".join(current_text), doc_id, doc_path, child_ids
                    )
                    parents.append(parent)
                    # Set parent_id on children
                    for c in chunks:
                        if c.id in child_ids:
                            c.parent_chunk_id = parent.id
                current_text = [chunk.content]
                current_tokens = tok
                child_ids = [chunk.id]

        if current_text:
            parent = self._make_parent(
                " ".join(current_text), doc_id, doc_path, child_ids
            )
            parents.append(parent)
            for c in chunks:
                if c.id in child_ids:
                    c.parent_chunk_id = parent.id

        return chunks + parents

    def _make_parent(
        self, text: str, doc_id: str, doc_path: str, child_ids: list[str]
    ) -> DocumentChunk:
        return DocumentChunk(
            doc_id=doc_id,
            doc_path=doc_path,
            chunk_type=ChunkType.TEXT,
            content=text,
            semantic_content=text,
            chunk_index=-1,  # parent marker
            metadata={"is_parent": True, "child_ids": child_ids},
        )