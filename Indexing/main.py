import os
# Fix: OpenMP conflict between torch / numpy / opencv on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Advanced RAG Indexing Pipeline - Entry Point & Examples

Usage:
    # Index a directory
    python main.py index ./documents

    # Search
    python main.py search "cross entropy loss function"

    # Interactive chat
    python main.py chat
"""
import argparse
import asyncio
import sys
from pathlib import Path

from rag_indexing.pipeline import AdvancedRAGIndexer, RAGConfig
from rag_indexing.config import (
    OCRConfig, LaTeXConfig, LlamaConfig,
    ColBERTConfig, ChunkingConfig, VectorStoreConfig
)


def build_config() -> RAGConfig:
    """
    Build production config.
    Adjust model names, paths, and server URLs here.
    """
    return RAGConfig(
        ocr=OCRConfig(
            use_angle_cls=True,
            lang="en",          # Change to "vi" for Vietnamese, "ch" for Chinese
            layout=True,
            table=True,
            show_log=False,
        ),
        latex=LaTeXConfig(
            enable_semantic_desc=True,
            sympy_simplify=True,
            temperature=0.2,
        ),
        llama=LlamaConfig(
            model="gemma4:e4b",                # or "llava:13b"
            base_url="http://localhost:11434",
            embed_model="nomic-embed-text",     # fast & accurate for RAG
            context_length=8192,
        ),
        colbert=ColBERTConfig(
            model_name="colbert-ir/colbertv2.0",
            index_name="my_rag_index",
            index_root=r"C:\Users\Admin\Desktop\Agent_Assistant\RAG\Indexing\colbert_indexes",
            doc_maxlen=256,
            query_maxlen=64,
        ),
        chunking=ChunkingConfig(
            chunk_size=512,
            chunk_overlap=64,
            isolate_formulas=True,
            isolate_tables=True,
            semantic_splitting=True,
        ),
        vector_store=VectorStoreConfig(
            provider="qdrant",
            host="localhost",
            port=6333,
            collection_name="rag_dense",
            embed_dim=768,      # nomic-embed-text dimension
        ),
        output_dir=Path("./rag_output"),
        enable_hybrid_search=True,
        rrf_k=60,
    )


def cmd_index(args):
    """Index a directory of documents."""
    config = build_config()
    indexer = AdvancedRAGIndexer(config)

    doc_path = Path(args.path)
    if not doc_path.exists():
        print(f"Error: path does not exist: {doc_path}")
        sys.exit(1)

    if doc_path.is_file():
        result = indexer.index_file(doc_path)
        print(f"\n✓ Indexed: {result.source_path}")
        print(f"  Pages: {result.total_pages}")
        print(f"  Chunks: {result.total_chunks}")
        print(f"  Formulas: {len(result.formulas)}")
        print(f"  Tables: {len(result.tables)}")
        print(f"  Time: {result.processing_time_s:.1f}s")

        # Build indexes after single file
        all_chunks = list(indexer.chunk_store.values())
        indexer.colbert.build_index(all_chunks, force_rebuild=args.force)
        indexer.bm25.build(all_chunks)
        indexer._save_index_manifest()
    else:
        indexer.index_directory(doc_path, force_rebuild=args.force)
    BASE = Path(r"C:\Users\Admin\Desktop\Agent_Assistant")
    manifest = config.BASE = Path(r"C:\Users\Admin\Desktop\Agent_Assistant\RAG\Indexing\output_dir") / "index_manifest.json"
    print(f"\n📂 Index manifest: {manifest}")


def cmd_search(args):
    """Run search — nếu không có query thì vào interactive mode."""
    from rag_indexing.query_translator import QueryStrategy

    config = build_config()
    print("⏳ Loading pipeline (one-time)...")
    indexer = AdvancedRAGIndexer(config)

    index_path = Path(config.colbert.index_root) / config.colbert.index_name
    if not index_path.exists() or indexer.colbert._collection.count() == 0:
        print("No index found. Run 'index' first.")
        sys.exit(1)

    indexer.chunk_store = indexer.colbert._load_chunk_store()
    strategy = QueryStrategy(args.strategy)
    print(f"✓ Ready. {indexer.colbert._collection.count()} vectors | strategy={strategy.value}\n")

    def do_search(query: str, top_k: int):
        if strategy == QueryStrategy.ORIGINAL:
            results = indexer.retrieve(query, top_k=top_k)
            translated_queries = [query]
        else:
            results, translated = indexer.retrieve_advanced(
                query, strategy=strategy, top_k=top_k
            )
            translated_queries = translated.queries
            # Hiển thị queries được sinh ra
            if len(translated_queries) > 1:
                print(f"  📝 Generated queries ({strategy.value}):")
                for i, q in enumerate(translated_queries, 1):
                    display = q[:120] + "..." if len(q) > 120 else q
                    print(f"     {i}. {display}")
                print()

        print(f"🔍 Query: {query}\n")
        if not results:
            print("  No results found.")
            return
        for r in results:
            chunk = r.chunk
            ctype = chunk.chunk_type.value.upper()
            print(f"  [{r.rank}] Score={r.rrf_score:.4f} | {ctype} | "
                  f"Page {chunk.page_num+1} | {Path(chunk.doc_path).name}")
            print(f"  {chunk.content.replace(chr(10), ' ')}")
            print()

    if args.query:
        do_search(args.query, args.top_k)
        return

    # Interactive mode
    print("💡 Interactive search mode. Type 'quit' to exit.")
    print(f"   Strategy: {strategy.value} | change with --strategy flag\n")
    while True:
        try:
            query = input("Search: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        do_search(query, args.top_k)


def cmd_chat(args):
    """Interactive RAG chat loop."""
    config = build_config()
    indexer = AdvancedRAGIndexer(config)

    # ChromaDB tự load từ disk
    index_path = Path(config.colbert.index_root) / config.colbert.index_name
    if index_path.exists() and indexer.colbert._collection.count() > 0:
        indexer.chunk_store = indexer.colbert._load_chunk_store()
        print(f"✓ ChromaDB index loaded ({indexer.colbert._collection.count()} vectors)")

    print("\n🤖 Advanced RAG Chat (Gemma 4 + ColBERT)")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        context = indexer.retrieve_with_context(query, top_k=5)
        system = (
            "You are a helpful assistant. Answer questions based ONLY on the provided context. "
            "If the answer is not in the context, say so. Be concise and accurate."
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        print("\nAssistant: ", end="", flush=True)
        try:
            answer = indexer.llama.generate(prompt, max_tokens=512, system=system)
            print(answer)
        except Exception as e:
            print(f"[Generation error: {e}]")
        print()


def cmd_adaptive(args):
    """
    Adaptive RAG chat:
    Router → Query Construction → Iterative Retrieval → CRAG → Answer
    """
    config = build_config()
    print("⏳ Loading pipeline...")
    indexer = AdvancedRAGIndexer(config)

    index_path = Path(config.colbert.index_root) / config.colbert.index_name
    if not index_path.exists() or indexer.colbert._collection.count() == 0:
        print("No index found. Run 'index' first.")
        sys.exit(1)

    indexer.chunk_store = indexer.colbert._load_chunk_store()
    print(f"✓ Ready. {indexer.colbert._collection.count()} vectors | Adaptive RAG ON\n")
    print("Modes: Router + HyDE + StepBack + MultiQuery + CRAG")
    print("Type 'quit' to exit, 'clear' to reset conversation.\n")

    # Single query mode
    if args.query:
        result = indexer.adaptive_query(args.query, top_k=10)
        _print_adaptive_result(result)
        return

    # Interactive mode
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "clear":
            indexer._get_adaptive_rag().clear_history()
            print("✓ Conversation cleared.\n")
            continue
        if not query:
            continue

        result = indexer.adaptive_query(query, top_k=10)
        _print_adaptive_result(result)


def _print_adaptive_result(result):
    """Pretty print AdaptiveResult."""
    print(f"\n{'─'*60}")
    print(f"Strategy : {result.strategy_used}")
    print(f"Iter     : {result.iterations} | CRAG: {'✓' if result.corrected else '✗'} | Self-RAG skip: {'✓' if result.self_rag_skipped else '✗'}")
    print(f"Chunks   : {len(result.final_chunks)} | Confidence: {result.confidence:.2f}")
    if result.sources:
        print(f"Sources  : {', '.join(Path(s).name for s in result.sources)}")
    print(f"{'─'*60}")
    print(f"\n🤖 {result.answer}\n")

    if result.final_chunks and not result.self_rag_skipped:
        print("📄 Top chunks:")
        for r in result.final_chunks[:3]:
            print(f"  [{r.rank}] Page {r.chunk.page_num+1} | {r.chunk.chunk_type.value} | score={r.rrf_score:.4f}")
            print(f"  {r.chunk.content.replace(chr(10), ' ')}")
            print()
    print()


def cmd_raw_rag(args):
    """
    Raw RAG — chỉ retrieve chunks, KHÔNG gọi Gemma 4 / VLM.
    Trả về chunks thô + metadata, dùng để:
    - Debug / kiểm tra chất lượng retrieval
    - Pipe output sang LLM khác
    - Xem nội dung PDF không qua generation
    """
    config = build_config()
    print("⏳ Loading pipeline (no LLM)...")

    # Tắt Ollama init để khởi động nhanh hơn
    indexer = AdvancedRAGIndexer(config)

    index_path = Path(config.colbert.index_root) / config.colbert.index_name
    if not index_path.exists() or indexer.colbert._collection.count() == 0:
        print("No index found. Run 'index' first.")
        sys.exit(1)

    indexer.chunk_store = indexer.colbert._load_chunk_store()
    strategy = getattr(args, 'strategy', 'auto')
    fmt = getattr(args, 'format', 'pretty')
    top_k = args.top_k

    print(f"✓ Ready. {indexer.colbert._collection.count()} vectors | strategy={strategy} | format={fmt}\n")

    def do_raw(query: str):
        results, translated_queries = indexer.translate_and_retrieve(query, strategy=strategy, top_k=top_k)

        # In translated queries nếu có nhiều hơn 1
        if len(translated_queries) > 1:
            print(f"  📝 Generated queries ({strategy}):")
            for i, q in enumerate(translated_queries, 1):
                display = q[:120] + "..." if len(q) > 120 else q
                print(f"     {i}. {display}")
            print()

        if fmt == 'json':
            import json
            output = {
                "query": query,
                "strategy": strategy,
                "total": len(results),
                "chunks": [
                    {
                        "rank": r.rank,
                        "score": round(r.rrf_score, 6),
                        "page": r.chunk.page_num + 1,
                        "type": r.chunk.chunk_type.value,
                        "doc": Path(r.chunk.doc_path).name,
                        "content": r.chunk.content,
                        "semantic_content": r.chunk.semantic_content or "",
                    }
                    for r in results
                ],
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))

        elif fmt == 'plain':
            for r in results:
                print(r.chunk.content)
                print()

        else:  # pretty (default)
            print(f"🔍 Query: {query}")
            print(f"   Strategy: {strategy} | {len(results)} chunks\n")
            for r in results:
                chunk = r.chunk
                ctype = chunk.chunk_type.value.upper()
                doc = Path(chunk.doc_path).name
                print(f"{'─'*60}")
                print(f"[{r.rank}] {ctype} | Page {chunk.page_num+1} | {doc} | score={r.rrf_score:.4f}")
                print()
                print(chunk.content)
                if chunk.semantic_content and chunk.semantic_content != chunk.content:
                    print(f"\n[semantic] {chunk.semantic_content}")
                print()

    # Single query
    if args.query:
        do_raw(args.query)
        return

    # Interactive
    print("💡 Raw RAG interactive. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        do_raw(query)


def cmd_demo(args):
    """Run a quick demo with synthetic data (no real files needed)."""
    print("🚀 Running demo with synthetic data...\n")
    config = build_config()

    # Skip Qdrant for demo
    config.enable_hybrid_search = True

    from rag_indexing.models import (
        DocumentChunk, ChunkType, FormulaResult, TableResult
    )
    from rag_indexing.colbert_indexer import BM25Index

    # Create synthetic chunks
    demo_chunks = [
        DocumentChunk(
            doc_id="demo_doc",
            doc_path="demo_paper.pdf",
            chunk_type=ChunkType.TEXT,
            content="Deep learning models are trained using gradient descent optimization algorithms. "
                    "The loss function quantifies prediction error on training data.",
            semantic_content="Deep learning optimization via gradient descent. Loss function measures error.",
            page_num=0,
            chunk_index=0,
        ),
        DocumentChunk(
            doc_id="demo_doc",
            doc_path="demo_paper.pdf",
            chunk_type=ChunkType.FORMULA,
            content=r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i | x_i; \theta)",
            semantic_content=(
                "Cross-entropy loss function: negative mean of log-probabilities over N samples. "
                "Variables: N (samples), theta (parameters), y_i (labels), x_i (inputs).\n"
                r"LaTeX: \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i | x_i; \theta)"
            ),
            page_num=1,
            chunk_index=1,
        ),
        DocumentChunk(
            doc_id="demo_doc",
            doc_path="demo_paper.pdf",
            chunk_type=ChunkType.TABLE,
            content="| Model | Params | Accuracy | Latency |\n|---|---|---|---|\n"
                    "| BERT-base | 110M | 91.2% | 25ms |\n"
                    "| RoBERTa | 125M | 93.1% | 28ms |\n"
                    "| ColBERT | 110M | 94.7% | 42ms |",
            semantic_content="Comparison table of transformer models. ColBERT achieves highest accuracy at 94.7% "
                             "with 110M parameters and 42ms latency.",
            page_num=2,
            chunk_index=2,
        ),
    ]

    # Build BM25 only (no Ollama/Qdrant needed for demo)
    bm25 = BM25Index()
    bm25.build(demo_chunks)

    # Demo BM25 search
    queries = [
        "cross entropy loss function",
        "model accuracy comparison",
        "gradient descent optimization",
    ]

    for q in queries:
        results = bm25.search(q, top_k=2)
        print(f"Query: '{q}'")
        for chunk_id, score, _ in results:
            chunk = next(c for c in demo_chunks if c.id == chunk_id)
            print(f"  → [{chunk.chunk_type.value}] score={score:.2f}: "
                  f"{chunk.content[:80]}...")
        print()

    print("✓ Demo complete. Set up Ollama + Qdrant to run full pipeline.")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced RAG Indexing Pipeline - Gemma4 + PaddleOCR + LaTeX + ColBERT"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = subparsers.add_parser("index", help="Index documents")
    p_index.add_argument("path", help="File or directory to index")
    p_index.add_argument("--force", action="store_true", help="Force rebuild index")

    # search
    p_search = subparsers.add_parser("search", help="Search với Query Translation")
    p_search.add_argument("query", nargs="?", default=None, help="Query (omit for interactive)")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_search.add_argument(
        "--strategy", default="auto",
        choices=["original", "multi_query", "rag_fusion", "decompose", "step_back", "hyde", "auto"],
        help=(
            "Query translation strategy:\n"
            "  original    : direct search (no transform)\n"
            "  multi_query : generate N variants → merge\n"
            "  rag_fusion  : multi_query + RRF re-ranking\n"
            "  decompose   : break into sub-questions\n"
            "  step_back   : abstract query → background context\n"
            "  hyde        : hypothetical document embedding\n"
            "  auto        : pick best strategy automatically (default)"
        )
    )

    # chat
    subparsers.add_parser("chat", help="Interactive RAG chat")

    # adaptive — Router + Query Construction + Adaptive RAG
    p_adaptive = subparsers.add_parser(
        "adaptive",
        help="Adaptive RAG: Router → HyDE + StepBack + MultiQuery → CRAG → Answer"
    )
    p_adaptive.add_argument("query", nargs="?", default=None, help="Query (omit for interactive)")
    p_adaptive.add_argument("--top-k", type=int, default=10, help="Number of chunks")
    p_adaptive.add_argument("--no-answer", action="store_true", help="Skip answer generation, show chunks only")

    # raw_rag — retrieve only, no LLM generation
    p_raw = subparsers.add_parser(
        "raw",
        help="Raw RAG: retrieve chunks KHÔNG gọi Gemma 4 / VLM"
    )
    p_raw.add_argument("query", nargs="?", default=None, help="Query (omit for interactive)")
    p_raw.add_argument("--top-k", type=int, default=5, help="Number of chunks (default: 5)")
    p_raw.add_argument(
        "--strategy", default="auto",
        choices=["original", "multi_query", "rag_fusion", "decompose", "step_back", "hyde", "auto"],
        help="Query translation strategy (default: auto)"
    )
    p_raw.add_argument(
        "--format", default="pretty",
        choices=["pretty", "json", "plain"],
        help="Output format: pretty (default), json, plain text only"
    )

    # demo
    subparsers.add_parser("demo", help="Quick demo (no external services needed)")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "adaptive":
        args.generate_answer = not args.no_answer
        cmd_adaptive(args)
    elif args.command == "raw":
        cmd_raw_rag(args)
    elif args.command == "demo":
        cmd_demo(args)


if __name__ == "__main__":
    main()