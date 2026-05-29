"""
Indexer & Retriever dùng ChromaDB + SentenceTransformers
Thay thế ragatouille (không tương thích Windows)
- ChromaDB: vector store persistent local
- SentenceTransformers: dense embeddings
- BM25: sparse keyword index  
- RRF: Reciprocal Rank Fusion
"""
import logging
import hashlib
from pathlib import Path
from typing import Optional

from .config import ColBERTConfig, VectorStoreConfig, RAGConfig
from .models import DocumentChunk, RetrievalResult, ChunkType

logger = logging.getLogger(__name__)


class ColBERTIndexer:
    """
    Dense retriever dùng ChromaDB + SentenceTransformers.
    Giữ nguyên interface để pipeline.py không cần sửa.
    """

    def __init__(self, config: ColBERTConfig):
        self.config = config
        self._client = None
        self._collection = None
        self._embedder = None
        self._chunks: dict[str, DocumentChunk] = {}
        self._init()

    def _init(self):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Thiếu thư viện. Chạy:\n"
                "  pip install chromadb sentence-transformers"
            )

        db_path = Path(self.config.index_root) / self.config.index_name
        db_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"},
        )
        # Embedding model — auto GPU nếu có CUDA
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        logger.info(
            f"ChromaDB initialized | collection: rag_collection | "
            f"existing docs: {self._collection.count()} | "
            f"embed device: {device.upper()}"
        )

    def build_index(
        self,
        chunks: list[DocumentChunk],
        force_rebuild: bool = False,
    ) -> Path:
        index_path = Path(self.config.index_root) / self.config.index_name

        if force_rebuild:
            try:
                self._client.delete_collection("rag_collection")
                self._collection = self._client.get_or_create_collection(
                    name="rag_collection",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("Index rebuilt from scratch")
            except Exception as e:
                logger.warning(f"Could not reset collection: {e}")

        valid = [c for c in chunks if c.chunk_index >= 0 and c.content.strip()]
        if not valid:
            logger.warning("No valid chunks to index")
            return index_path

        for c in valid:
            self._chunks[c.id] = c

        # Upsert toàn bộ 1 lần với batch lớn (tránh overhead nhiều lần gọi ChromaDB)
        logger.info(f"Indexing {len(valid)} chunks...")
        BATCH = 512
        for i in range(0, len(valid), BATCH):
            batch = valid[i:i + BATCH]
            self._upsert_batch(batch)
            logger.info(f"  {min(i + BATCH, len(valid))}/{len(valid)} indexed")

        logger.info(f"Index complete. Total: {self._collection.count()} vectors")
        return index_path

    def _upsert_batch(self, chunks: list[DocumentChunk]):
        import torch
        texts = [c.semantic_content or c.content for c in chunks]
        ids   = [self._safe_id(c.id) for c in chunks]

        # Dùng embedding đã tính (từ pipeline) nếu có — tránh embed lại
        if all(c.embedding for c in chunks):
            embeddings = [c.embedding for c in chunks]
        else:
            # Fallback: embed những chunk chưa có
            device = "cuda" if torch.cuda.is_available() else "cpu"
            batch_size = 256 if device == "cuda" else 64
            missing_idx = [i for i, c in enumerate(chunks) if not c.embedding]
            missing_texts = [texts[i] for i in missing_idx]
            new_embs = self._embedder.encode(
                missing_texts,
                batch_size=batch_size,
                device=device,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()
            embeddings = [c.embedding for c in chunks]
            for idx, emb in zip(missing_idx, new_embs):
                embeddings[idx] = emb

        metadatas = [
            {
                "chunk_id": c.id,
                "doc_id": c.doc_id,
                "doc_path": c.doc_path,
                "chunk_type": c.chunk_type.value,
                "page_num": c.page_num,
                "chunk_index": c.chunk_index,
                "parent_chunk_id": c.parent_chunk_id or "",
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=[t[:2000] for t in texts],
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float, dict]]:
        if self._collection is None:
            return []
        query_emb = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_emb,
            n_results=min(top_k, max(1, self._collection.count())),
            include=["metadatas", "distances"],
        )
        output = []
        for i, meta in enumerate(results["metadatas"][0]):
            chunk_id = meta.get("chunk_id", "")
            distance = results["distances"][0][i]
            score = 1.0 - distance  # cosine distance → similarity
            output.append((chunk_id, score, meta))
        return output

    def add_to_index(self, new_chunks: list[DocumentChunk]):
        valid = [c for c in new_chunks if c.chunk_index >= 0 and c.content.strip()]
        for c in valid:
            self._chunks[c.id] = c
        self._upsert_batch(valid)
        logger.info(f"Added {len(valid)} chunks to index")

    def _safe_id(self, uid: str) -> str:
        return hashlib.md5(uid.encode()).hexdigest()

    def _load_chunk_store(self) -> dict:
        """
        Load chunk store từ ChromaDB về memory.
        Dùng khi search/chat mà không index lại từ đầu.
        """
        from .models import DocumentChunk, ChunkType

        results = self._collection.get(include=["documents", "metadatas"])
        chunk_store = {}

        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunk_id = meta.get("chunk_id", "")
            if not chunk_id:
                continue
            chunk = DocumentChunk(
                doc_id=meta.get("doc_id", ""),
                doc_path=meta.get("doc_path", ""),
                chunk_type=ChunkType(meta.get("chunk_type", "text")),
                content=doc,
                semantic_content=doc,
                page_num=meta.get("page_num", 0),
                chunk_index=meta.get("chunk_index", 0),
                parent_chunk_id=meta.get("parent_chunk_id") or None,
            )
            # Gán lại đúng id
            object.__setattr__(chunk, 'id', chunk_id) if hasattr(chunk, '__dataclass_fields__') else None
            chunk.__dict__['id'] = chunk_id
            chunk_store[chunk_id] = chunk

        logger.info(f"Loaded {len(chunk_store)} chunks from ChromaDB")
        return chunk_store


class DenseVectorStore:
    """Stub — ChromaDB đã xử lý dense store, class này giữ để pipeline không lỗi."""

    def __init__(self, config=None):
        self.config = config

    def upsert_chunks(self, chunks: list[DocumentChunk]):
        pass  # ColBERTIndexer đã index rồi

    def search(self, query_vector: list[float], top_k: int = 20) -> list:
        return []


class BM25Index:
    """BM25 sparse index dùng rank_bm25."""

    def __init__(self):
        self._bm25 = None
        self._chunks: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")

        self._chunks = [c for c in chunks if c.chunk_index >= 0]
        tokenized = [
            (c.semantic_content or c.content).lower().split()
            for c in self._chunks
        ]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(self._chunks)} docs")

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float, dict]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._chunks[i].id, float(scores[i]), {}) for i in top_idx]


class HybridRetriever:
    """ChromaDB dense + BM25 sparse → RRF fusion."""

    def __init__(self, config: RAGConfig, chunk_store: dict[str, DocumentChunk]):
        self.config = config
        self.chunk_store = chunk_store
        self.colbert: Optional[ColBERTIndexer] = None
        self.dense: Optional[DenseVectorStore] = None
        self.bm25: Optional[BM25Index] = None

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        all_rankings: dict[str, list[int]] = {}

        # Dense (ChromaDB)
        if self.colbert:
            try:
                for rank, (chunk_id, score, _) in enumerate(
                    self.colbert.search(query, top_k=top_k * 2)
                ):
                    all_rankings.setdefault(chunk_id, []).append(rank + 1)
            except Exception as e:
                logger.warning(f"Dense search failed: {e}")

        # BM25
        if self.bm25 and self.config.enable_hybrid_search:
            try:
                for rank, (chunk_id, score, _) in enumerate(
                    self.bm25.search(query, top_k=top_k * 2)
                ):
                    all_rankings.setdefault(chunk_id, []).append(rank + 1)
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

        rrf_scores = self._rrf(all_rankings)
        top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        results = []
        for rank, chunk_id in enumerate(top_ids):
            chunk = self.chunk_store.get(chunk_id)
            if chunk:
                results.append(RetrievalResult(
                    chunk=chunk,
                    rrf_score=rrf_scores[chunk_id],
                    rank=rank + 1,
                ))
        return results

    def _rrf(self, rankings: dict[str, list[int]]) -> dict[str, float]:
        k = self.config.rrf_k
        return {
            cid: sum(1.0 / (k + r) for r in ranks)
            for cid, ranks in rankings.items()
        }