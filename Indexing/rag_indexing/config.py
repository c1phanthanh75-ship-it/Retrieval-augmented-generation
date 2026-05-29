"""
Advanced RAG Indexing Configuration
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OCRConfig:
    use_angle_cls: bool = True
    lang: str = "en"
    layout: bool = True
    table: bool = True
    show_log: bool = False
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.6
    det_db_unclip_ratio: float = 1.5
    use_gpu: bool = True   # auto-detect, set False để force CPU


@dataclass
class LaTeXConfig:
    pix2tex_model: str = "default"
    temperature: float = 0.25
    max_dim: int = 2048
    enable_semantic_desc: bool = True  # LLM generates natural language description
    sympy_simplify: bool = True


@dataclass
class LlamaConfig:
    model: str = "gemma4:e4b"
    base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"   # fallback fast embed
    context_length: int = 8192
    temperature: float = 0.0
    num_gpu: int = -1  # -1 = auto


@dataclass
class ColBERTConfig:
    model_name: str = "colbert-ir/colbertv2.0"
    index_name: str = "rag_colbert_index"
    index_root: str = "./colbert_indexes"
    doc_maxlen: int = 256
    query_maxlen: int = 64
    nbits: int = 2           # compression bits
    kmeans_niters: int = 4
    ncells: int = 1          # PLAID cells for retrieval
    centroid_score_threshold: float = 0.45


@dataclass
class ChunkingConfig:
    chunk_size: int = 256          
    chunk_overlap: int = 32        
    min_chunk_size: int = 50
    isolate_formulas: bool = True
    isolate_tables: bool = True
    semantic_splitting: bool = True
    max_sentences_per_flush: int = 500   


@dataclass
class VectorStoreConfig:
    provider: str = "qdrant"         # qdrant | chroma | faiss
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "rag_dense"
    embed_dim: int = 768             # nomic-embed-text dim
    distance: str = "Cosine"


@dataclass
class RAGConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    latex: LaTeXConfig = field(default_factory=LaTeXConfig)
    llama: LlamaConfig = field(default_factory=LlamaConfig)
    colbert: ColBERTConfig = field(default_factory=ColBERTConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    output_dir: Path = Path("./rag_output")
    log_level: str = "INFO"
    max_concurrent_docs: int = 4
    enable_hybrid_search: bool = True
    rrf_k: int = 60
    # GPU settings
    use_gpu: bool = True           # auto-detect CUDA, set False để force CPU
    gpu_batch_size: int = 128      # batch size khi embed trên GPU (lớn hơn CPU)
    cpu_batch_size: int = 32       # batch size khi embed trên CPU