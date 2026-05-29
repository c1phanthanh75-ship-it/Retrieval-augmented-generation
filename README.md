# Advanced RAG Indexing Pipeline
**Gemma4  · PaddleOCR · LaTeX · ColBERT**

## Architecture

```
Documents (PDF / IMG)
        │
        ▼
┌──────────────────────┐
│  PaddleOCR PP-OCRv4  │  → Text, Tables, Formula regions
│  + PP-Structure      │  → Layout detection (title/header/figure)
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   LaTeX Parser       │  → Pix2Tex: formula image → LaTeX
│   (Pix2Tex + SymPy)  │  → SymPy: simplify, extract variables
│   + Llama 3 desc     │  → Llama 3: natural language description
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   Advanced Chunker   │  → Semantic splitting (512 tok, 64 overlap)
│   Hierarchical       │  → Isolated formula & table chunks
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│   Llama 3.2 Vision   │  → Dense embeddings (nomic-embed-text)
│   + VLM Enrichment   │  → Image/Table semantic descriptions
└──────────────────────┘
        │
    ┌───┴───────────┐
    ▼               ▼
ColBERT         Qdrant           BM25
(PLAID)      (dense 768d)     (sparse)
    │               │               │
    └───────────────┴───────────────┘
                    │
                 RRF Fusion
                    │
              Top-K Results
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama + pull models
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2-vision:11b
ollama pull nomic-embed-text
```

### 3. Start Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Install PaddlePaddle (CPU)
```bash
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
# Or GPU:
pip install paddlepaddle-gpu
```

## Usage

```bash
# Demo (no external services needed)
python main.py demo

# Index documents
python main.py index ./documents/

# Index single file
python main.py index ./paper.pdf

# Search
python main.py search "cross entropy loss optimization"

# Interactive chat
python main.py chat
```

## Python API

```python
from rag_indexing import AdvancedRAGIndexer, RAGConfig

config = RAGConfig()
indexer = AdvancedRAGIndexer(config)

# Index
indexer.index_directory("./documents")

# Search
results = indexer.retrieve("neural network loss function", top_k=10)
for r in results:
    print(f"[{r.rank}] {r.chunk.chunk_type.value}: {r.chunk.content[:100]}")

# RAG generation context
context = indexer.retrieve_with_context("What is cross entropy?", top_k=5)
```

## Project Structure

```
rag_indexing/
├── __init__.py          # Public API
├── config.py            # All configuration dataclasses
├── models.py            # Data models (Region, Chunk, Result...)
├── ocr_extractor.py     # PaddleOCR extraction module
├── latex_parser.py      # Pix2Tex + SymPy + LLM formula parsing
├── llama_client.py      # Ollama client + VLM enrichment
├── chunker.py           # Advanced chunking (semantic + hierarchical)
└── colbert_indexer.py   # ColBERT + Qdrant + BM25 + RRF fusion
main.py                  # CLI entry point
requirements.txt
```

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| OCR | PaddleOCR PP-OCRv4 | Best layout analysis, table recovery |
| Formula OCR | Pix2Tex | Dedicated math OCR model |
| Formula semantics | SymPy + Llama 3 | Symbolic simplification + NL description |
| Embeddings | nomic-embed-text | Fast, high quality, 768-dim |
| VLM | Llama 3.2-vision 11B | Local, multimodal, open source |
| Retrieval | ColBERT v2 PLAID | Token-level late interaction, high accuracy |
| Hybrid fusion | RRF (k=60) | Robust fusion of dense + sparse + ColBERT |
| Vector DB | Qdrant | Fast, local, production-ready |
