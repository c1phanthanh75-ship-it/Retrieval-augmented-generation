"""
Advanced RAG Indexing Pipeline - Main Orchestrator
Coordinates: PaddleOCR → LaTeX → Llama 3 → ColBERT
"""
import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Optional

from .config import RAGConfig
from .models import (
    DocumentRegion, DocumentChunk, FormulaResult,
    TableResult, IndexedDocument, RegionType
)
from .ocr_extractor import PaddleOCRExtractor
from .latex_parser import LaTeXParser
from .llama_client import OllamaClient, VLMEnricher
from .chunker import AdvancedChunker
from .colbert_indexer import ColBERTIndexer, DenseVectorStore, BM25Index, HybridRetriever
from .query_translator import QueryTranslator, QueryStrategy, TranslationAwareRetriever
from .query_router import QueryRouter
from .query_constructor import QueryConstructor
from .adaptive_rag import AdaptiveRAG, AdaptiveConfig

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AdvancedRAGIndexer:
    """
    Full end-to-end RAG indexing pipeline.

    Usage:
        config = RAGConfig()
        indexer = AdvancedRAGIndexer(config)
        indexer.index_directory("./documents")
        results = indexer.retrieve("cross entropy loss optimization")
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing Advanced RAG Indexer...")

        # Initialize all modules
        self.ocr = PaddleOCRExtractor(self.config.ocr)
        self.llama = OllamaClient(self.config.llama)
        self.vlm_enricher = VLMEnricher(self.llama)
        self.latex_parser = LaTeXParser(self.config.latex, llm_client=self.llama)
        self.chunker = AdvancedChunker(self.config.chunking)
        self.colbert = ColBERTIndexer(self.config.colbert)
        self.dense_store = DenseVectorStore(self.config.vector_store)
        self.bm25 = BM25Index()

        # In-memory chunk store for retrieval
        self.chunk_store: dict[str, DocumentChunk] = {}
        self.indexed_docs: list[IndexedDocument] = []

        # Query translator
        self.query_translator = QueryTranslator(
            llm_client=self.llama,
            strategy=QueryStrategy.AUTO,
        )

        logger.info("Pipeline initialized.")

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def index_file(self, file_path: Path) -> IndexedDocument:
        """
        Index một file — pipeline đa luồng:
        Stage 1 (I/O thread)  : OCR extraction
        Stage 2 (thread pool) : LaTeX + Table xử lý song song
        Stage 3 (GPU batch)   : Embed tất cả chunks 1 lần
        Stage 4 (main thread) : Upsert ChromaDB
        """
        import concurrent.futures, torch
        start = time.time()
        logger.info(f"Indexing: {file_path}")

        # ── Stage 1: OCR ──────────────────────────────────────────────────
        logger.info("  [1/4] OCR extraction...")
        regions = self.ocr.extract_from_file(file_path)
        logger.info(f"  → {len(regions)} regions")

        # ── Stage 2: LaTeX + Table song song ──────────────────────────────
        logger.info("  [2/4] LaTeX + Table (parallel)...")
        formula_regions = [r for r in regions if r.region_type.value == "formula"]
        table_regions   = [r for r in regions if r.region_type.value == "table"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            fut_formula = ex.submit(self.latex_parser.parse_regions, regions)
            fut_table   = ex.submit(self._process_tables, regions)
            formulas = fut_formula.result()
            tables   = fut_table.result()

        logger.info(f"  → {len(formulas)} formulas, {len(tables)} tables")

        # ── Stage 3: Chunking ─────────────────────────────────────────────
        logger.info("  [3/4] Chunking...")
        doc_id = str(file_path.stem)
        chunks = self.chunker.chunk_document(
            doc_id=doc_id,
            doc_path=str(file_path),
            regions=regions,
            formulas=formulas,
            tables=tables,
        )
        logger.info(f"  → {len(chunks)} chunks")

        # ── Stage 4: Embed tất cả chunks 1 lần (GPU batch) ────────────────
        logger.info("  [4/4] Embedding (batch GPU)...")
        to_embed = [c for c in chunks if c.chunk_index >= 0 and not c.embedding]
        if to_embed:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = 256 if device == "cuda" else 64
            texts = [c.semantic_content or c.content for c in to_embed]
            embeddings = self.colbert._embedder.encode(
                texts,
                batch_size=batch_size,
                device=device,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,   # cosine = dot product → nhanh hơn
            ).tolist()
            for chunk, emb in zip(to_embed, embeddings):
                chunk.embedding = emb
            logger.info(f"  → {len(to_embed)} embeddings on {device.upper()}")

        # ── Store ─────────────────────────────────────────────────────────
        for chunk in chunks:
            self.chunk_store[chunk.id] = chunk

        self.colbert.build_index(chunks)

        elapsed = time.time() - start
        indexed_doc = IndexedDocument(
            doc_id=doc_id,
            source_path=str(file_path),
            total_pages=max((r.page_num for r in regions), default=0) + 1,
            total_chunks=len([c for c in chunks if c.chunk_index >= 0]),
            chunks=chunks,
            regions=regions,
            formulas=formulas,
            tables=tables,
            processing_time_s=elapsed,
        )
        self.indexed_docs.append(indexed_doc)
        logger.info(f"  ✓ Done in {elapsed:.1f}s — {indexed_doc.total_chunks} chunks")
        return indexed_doc

    def index_directory(
        self,
        directory: Path,
        extensions: tuple = (".pdf", ".png", ".jpg", ".jpeg"),
        force_rebuild: bool = False,
    ):
        """Index all supported files in a directory."""
        directory = Path(directory)
        files = [f for f in directory.rglob("*") if f.suffix.lower() in extensions]
        logger.info(f"Found {len(files)} files in {directory}")

        for i, f in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] {f.name}")
            try:
                self.index_file(f)
            except Exception as e:
                logger.error(f"Failed to index {f}: {e}", exc_info=True)

        # Build ColBERT index from all chunks
        logger.info("Building ColBERT index...")
        all_chunks = list(self.chunk_store.values())
        self.colbert.build_index(all_chunks, force_rebuild=force_rebuild)

        # Build BM25 index
        logger.info("Building BM25 sparse index...")
        self.bm25.build(all_chunks)

        logger.info(f"Indexing complete. Total chunks: {len(all_chunks)}")
        self._save_index_manifest()

    def retrieve(
        self, query: str, top_k: int = 10
    ) -> list:
        """
        Hybrid retrieval: ColBERT + Dense + BM25 → RRF fusion.
        Sau đó expand context: lấy chunk liền kề để tránh cắt giữa công thức.
        """
        try:
            query_emb = self.colbert._embedder.encode([query]).tolist()[0]
        except Exception:
            query_emb = []

        retriever = HybridRetriever(
            config=self.config,
            chunk_store=self.chunk_store,
        )
        retriever.colbert = self.colbert
        retriever.dense = self.dense_store
        retriever.bm25 = self.bm25

        results = retriever.retrieve(query, query_emb, top_k=top_k)

        # Context expansion: với top 3 chunks, ghép chunk liền sau vào content
        results = self._expand_chunks(results, n_top=3)

        logger.info(f"Query: '{query[:60]}...' → {len(results)} results")
        return results

    def _expand_chunks(self, results: list, n_top: int = 3) -> list:
        """
        Với top n_top chunks: tìm chunk liền sau (same doc, next index)
        và ghép vào content để tránh cắt giữa công thức/giải thích.
        """
        if not results or not self.chunk_store:
            return results

        # Build lookup: (doc_id, page_num, chunk_index) → chunk
        idx_lookup: dict[tuple, object] = {}
        for chunk in self.chunk_store.values():
            key = (chunk.doc_id, chunk.page_num, chunk.chunk_index)
            idx_lookup[key] = chunk

        for i, r in enumerate(results[:n_top]):
            chunk = r.chunk
            next_key = (chunk.doc_id, chunk.page_num, chunk.chunk_index + 1)
            next_chunk = idx_lookup.get(next_key)
            if next_chunk and next_chunk.chunk_type.value == "text":
                # Ghép 80 words đầu của chunk tiếp theo
                extra = " ".join(next_chunk.content.split()[:80])
                if extra and extra not in chunk.content:
                    chunk.content = chunk.content + " [...] " + extra

        return results

    def retrieve_advanced(
        self,
        query: str,
        top_k: int = 10,
        strategy: QueryStrategy = QueryStrategy.AUTO,
    ) -> tuple:
        """
        Retrieval với Query Translation.
        Trả về (results, TranslatedQuery).

        strategy options:
          QueryStrategy.AUTO        — tự chọn dựa trên query
          QueryStrategy.MULTI_QUERY — tạo nhiều biến thể query
          QueryStrategy.RAG_FUSION  — multi-query + RRF
          QueryStrategy.DECOMPOSE   — tách thành sub-questions
          QueryStrategy.STEP_BACK   — câu hỏi tổng quát hơn
          QueryStrategy.HYDE        — hypothetical document embedding
        """
        retriever = HybridRetriever(
            config=self.config,
            chunk_store=self.chunk_store,
        )
        retriever.colbert = self.colbert
        retriever.dense = self.dense_store
        retriever.bm25 = self.bm25

        translation_retriever = TranslationAwareRetriever(
            retriever=retriever,
            translator=self.query_translator,
        )
        return translation_retriever.retrieve(query, top_k=top_k, strategy=strategy)

    def retrieve_with_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve and format context string for LLM generation.
        Expands child chunks to their parent for more context.
        """
        results = self.retrieve(query, top_k=top_k)
        context_parts = []

        for i, result in enumerate(results, 1):
            chunk = result.chunk
            if chunk.parent_chunk_id and chunk.parent_chunk_id in self.chunk_store:
                content = self.chunk_store[chunk.parent_chunk_id].content
            else:
                content = chunk.content

            context_parts.append(
                f"[{i}] (Source: {Path(chunk.doc_path).name}, "
                f"Page {chunk.page_num + 1}, Score: {result.rrf_score:.3f})\n"
                f"{content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def translate_and_retrieve(
        self,
        query: str,
        strategy: str = "auto",
        top_k: int = 10,
    ) -> tuple:
        """Returns (results, translated_queries_list)"""
        try:
            retriever = HybridRetriever(
                config=self.config,
                chunk_store=self.chunk_store,
            )
            retriever.colbert = self.colbert
            retriever.dense  = self.dense_store
            retriever.bm25   = self.bm25

            translation_retriever = TranslationAwareRetriever(
                retriever=retriever,
                translator=self.query_translator,
            )
            results, translated = translation_retriever.retrieve(
                query=query,
                top_k=top_k,
                strategy=QueryStrategy(strategy),
            )
            logger.info(
                f"Query Translation [{strategy}]: '{query[:40]}' → {len(results)} results"
            )
            return results, translated.queries
        except Exception as e:
            logger.warning(f"Query translation failed ({e}), falling back to direct search")
            return self.retrieve(query, top_k=top_k), [query]

    def adaptive_query(
        self,
        question: str,
        top_k: int = 10,
        generate_answer: bool = True,
    ):
        """
        Full Adaptive RAG pipeline:
        Router → Query Constructor → Iterative Retrieval → CRAG → Answer

        Returns AdaptiveResult with answer, chunks, metadata.
        """
        adaptive = self._get_adaptive_rag()
        return adaptive.query(question, top_k=top_k, generate_answer=generate_answer)

    def _get_adaptive_rag(self) -> "AdaptiveRAG":
        """Lazy init AdaptiveRAG (reuse across calls)."""
        if not hasattr(self, '_adaptive_rag') or self._adaptive_rag is None:
            retriever = HybridRetriever(
                config=self.config,
                chunk_store=self.chunk_store,
            )
            retriever.colbert = self.colbert
            retriever.dense = self.dense_store
            retriever.bm25 = self.bm25

            self._adaptive_rag = AdaptiveRAG(
                retriever=retriever,
                llm_client=self.llama,
                router=QueryRouter(llm_client=self.llama),
                constructor=QueryConstructor(llm_client=self.llama),
                config=AdaptiveConfig(
                    max_iterations=2,
                    use_self_rag=True,
                    use_crag=True,
                    use_iterative=True,
                ),
            )
        return self._adaptive_rag

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _process_tables(self, regions: list[DocumentRegion]) -> list[TableResult]:
        """Extract and enrich table regions."""
        tables = []
        for region in regions:
            if region.region_type != RegionType.TABLE:
                continue
            if not region.text:
                continue

            semantic_desc = self.vlm_enricher.describe_table(region.text)
            table = TableResult(
                markdown=region.text,
                semantic_description=semantic_desc,
                region_id=region.id,
            )
            tables.append(table)

        return tables

    def _enrich_image_regions(self, regions: list[DocumentRegion]):
        """In-place: add VLM descriptions to image regions."""
        for region in regions:
            if region.region_type == RegionType.IMAGE and region.image_data:
                desc = self.vlm_enricher.describe_image_region(region)
                region.text = desc or region.text

    def _embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Embed chunks dùng SentenceTransformers (không cần Ollama embed model)."""
        to_embed = [c for c in chunks if c.chunk_index >= 0 and not c.embedding]
        if not to_embed:
            return chunks

        texts = [c.semantic_content or c.content for c in to_embed]
        try:
            # Dùng lại embedder từ ColBERT indexer (all-MiniLM-L6-v2)
            embeddings = self.colbert._embedder.encode(
                texts, show_progress_bar=False, batch_size=32
            ).tolist()
            for chunk, emb in zip(to_embed, embeddings):
                chunk.embedding = emb
            logger.info(f"Embedded {len(to_embed)} chunks via SentenceTransformers")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")

        return chunks

    def _save_index_manifest(self):
        """Save indexing summary — merge với manifest cũ nếu có."""
        out = self.config.output_dir / "index_manifest.json"

        # Load manifest cũ nếu tồn tại
        existing: dict[str, dict] = {}
        if out.exists():
            try:
                old = json.loads(out.read_text())
                for doc in old.get("documents", []):
                    existing[doc["doc_id"]] = doc
            except Exception:
                pass

        # Merge: session mới ghi đè doc_id trùng, giữ lại doc cũ
        for d in self.indexed_docs:
            existing[d.doc_id] = {
                "doc_id": d.doc_id,
                "source": d.source_path,
                "pages": d.total_pages,
                "chunks": d.total_chunks,
                "formulas": len(d.formulas),
                "tables": len(d.tables),
                "processing_time_s": round(d.processing_time_s, 2),
                "indexed_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            }

        # Tính total từ ChromaDB (source of truth)
        try:
            total_vectors = self.colbert._collection.count()
        except Exception:
            total_vectors = len(self.chunk_store)

        manifest = {
            "total_documents": len(existing),
            "total_chunks": total_vectors,
            "documents": list(existing.values()),
        }
        out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        logger.info(f"Index manifest saved: {out} ({len(existing)} docs, {total_vectors} vectors)")

    # ------------------------------------------------------------------ #
    # Async support                                                         #
    # ------------------------------------------------------------------ #

    async def aindex_directory(self, directory: Path, **kwargs):
        """Async wrapper for concurrent document indexing."""
        directory = Path(directory)
        extensions = kwargs.get("extensions", (".pdf", ".png", ".jpg"))
        files = [f for f in directory.rglob("*") if f.suffix.lower() in extensions]

        sem = asyncio.Semaphore(self.config.max_concurrent_docs)

        async def index_one(f: Path):
            async with sem:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.index_file, f)

        results = await asyncio.gather(*[index_one(f) for f in files], return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Async indexing error: {r}")

        all_chunks = list(self.chunk_store.values())
        self.colbert.build_index(all_chunks)
        self.bm25.build(all_chunks)
        self._save_index_manifest()