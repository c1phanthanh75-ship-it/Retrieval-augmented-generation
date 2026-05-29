"""
Adaptive RAG Module
Tự động điều chỉnh retrieval strategy dựa trên:
1. Query complexity assessment
2. Retrieval quality evaluation
3. Iterative retrieval (nếu kết quả chưa đủ)
4. Self-RAG: model tự đánh giá có cần retrieval không
5. CRAG: Corrective RAG - tự sửa khi kết quả kém
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import DocumentChunk, RetrievalResult
    from .query_router import RouteDecision
    from .query_constructor import ConstructedQuery

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    max_iterations: int = 3           # max retrieval rounds
    min_relevance_score: float = 0.005 # minimum acceptable RRF score
    min_results: int = 2              # minimum chunks needed
    use_self_rag: bool = True         # skip retrieval if LLM already knows
    use_crag: bool = True             # corrective retrieval if quality low
    use_iterative: bool = True        # retry with refined query if insufficient
    relevance_threshold: float = 0.5  # LLM relevance judge threshold


@dataclass
class AdaptiveResult:
    """Final result from adaptive RAG pipeline."""
    query: str
    final_chunks: list                  # list[RetrievalResult]
    iterations: int = 1
    strategy_used: str = ""
    self_rag_skipped: bool = False      # True = LLM answered without retrieval
    corrected: bool = False             # True = CRAG triggered
    answer: str = ""                    # final generated answer
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0


class AdaptiveRAG:
    """
    Adaptive RAG orchestrator.
    Connects Router → Constructor → Retriever → Evaluator → Generator.
    """

    def __init__(
        self,
        retriever,          # HybridRetriever
        llm_client,         # OllamaClient
        router=None,        # QueryRouter
        constructor=None,   # QueryConstructor
        config: AdaptiveConfig = None,
    ):
        self.retriever = retriever
        self.llm = llm_client
        self.router = router
        self.constructor = constructor
        self.config = config or AdaptiveConfig()
        self._conversation_history: list[dict] = []

    def query(
        self,
        question: str,
        top_k: int = 10,
        generate_answer: bool = True,
    ) -> AdaptiveResult:
        """
        Full adaptive RAG pipeline.
        """
        logger.info(f"Adaptive RAG: '{question[:60]}'")

        # --- Step 1: Self-RAG check ---
        if self.config.use_self_rag:
            if self._should_skip_retrieval(question):
                answer = self._generate_direct(question)
                return AdaptiveResult(
                    query=question,
                    final_chunks=[],
                    self_rag_skipped=True,
                    answer=answer,
                    strategy_used="self_rag_direct",
                    confidence=0.9,
                )

        # --- Step 2: Route query ---
        route = None
        if self.router:
            route = self.router.route(question, self._conversation_history)
            logger.info(f"  Route: {route.strategy.value} | intent: {route.intent.value}")

        # --- Step 3: Construct queries ---
        constructed = None
        queries_to_run = [question]
        if self.constructor:
            constructed = self.constructor.construct(
                question,
                strategy=route.strategy.value if route else "hybrid",
                use_hyde=True,
                use_stepback=True,
                use_multiquery=True,
            )
            queries_to_run = constructed.final_queries[:4]  # max 4 variants
            logger.info(f"  Constructed {len(queries_to_run)} query variants")

        # --- Step 4: Iterative retrieval ---
        all_chunks = []
        iteration = 0

        for iteration in range(1, self.config.max_iterations + 1):
            chunks = self._retrieve_all(
                queries_to_run,
                top_k=top_k,
                filters=route.filters if route else {},
            )

            # Deduplicate by chunk id
            seen = {c.chunk.id for c in all_chunks}
            new_chunks = [c for c in chunks if c.chunk.id not in seen]
            all_chunks.extend(new_chunks)

            logger.info(f"  Iteration {iteration}: {len(all_chunks)} total chunks")

            # Check if sufficient
            if self._is_sufficient(all_chunks, question):
                break

            # Refine query for next iteration
            if iteration < self.config.max_iterations:
                queries_to_run = self._refine_queries(question, all_chunks)
                logger.info(f"  Refining queries for iteration {iteration + 1}")

        # --- Step 5: CRAG - evaluate & correct ---
        corrected = False
        if self.config.use_crag and all_chunks:
            all_chunks, corrected = self._corrective_filter(question, all_chunks)

        # Sort by RRF score
        all_chunks.sort(key=lambda x: x.rrf_score, reverse=True)
        final_chunks = all_chunks[:top_k]

        # --- Step 6: Generate answer ---
        answer = ""
        if generate_answer and final_chunks:
            answer = self._generate_answer(question, final_chunks)

        # Update conversation history
        self._conversation_history.append({"role": "user", "content": question})
        if answer:
            self._conversation_history.append({"role": "assistant", "content": answer})
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        sources = list({c.chunk.doc_path for c in final_chunks})
        confidence = self._estimate_confidence(final_chunks)

        return AdaptiveResult(
            query=question,
            final_chunks=final_chunks,
            iterations=iteration,
            strategy_used=route.strategy.value if route else "hybrid",
            corrected=corrected,
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    # Self-RAG                                                             #
    # ------------------------------------------------------------------ #

    def _should_skip_retrieval(self, question: str) -> bool:
        return False

    def _generate_direct(self, question: str) -> str:
        """Generate answer without retrieval context."""
        try:
            return self.llm.generate(
                question,
                max_tokens=512,
                system="You are a helpful technical assistant. Answer concisely and accurately.",
            )
        except Exception as e:
            return f"Error generating answer: {e}"

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def _retrieve_all(
        self,
        queries: list[str],
        top_k: int,
        filters: dict,
    ) -> list:
        """Run retrieval for all query variants and merge results."""
        from sentence_transformers import SentenceTransformer
        all_results = []
        seen_ids = set()

        for q in queries:
            try:
                # Get query embedding
                query_emb = self.retriever.colbert._embedder.encode([q]).tolist()[0]
                results = self.retriever.retrieve(q, query_emb, top_k=top_k)

                # Apply filters
                if filters:
                    results = self._apply_filters(results, filters)

                for r in results:
                    if r.chunk.id not in seen_ids:
                        seen_ids.add(r.chunk.id)
                        all_results.append(r)

            except Exception as e:
                logger.warning(f"Retrieval failed for '{q[:40]}': {e}")

        return all_results

    def _apply_filters(self, results: list, filters: dict) -> list:
        """Filter results by chunk_type or other metadata."""
        chunk_type = filters.get("chunk_type")
        if chunk_type:
            filtered = [r for r in results if r.chunk.chunk_type.value == chunk_type]
            # If filter removes everything, return original
            return filtered if filtered else results
        return results

    # ------------------------------------------------------------------ #
    # Quality Evaluation                                                   #
    # ------------------------------------------------------------------ #

    def _is_sufficient(self, chunks: list, question: str) -> bool:
        """Check if retrieved chunks are sufficient to answer the question."""
        if len(chunks) < self.config.min_results:
            return False

        top_score = max(c.rrf_score for c in chunks) if chunks else 0
        if top_score < self.config.min_relevance_score:
            return False

        return True

    def _corrective_filter(self, question, chunks):
        return [
            c for c in chunks
            if c.rrf_score > self.config.min_relevance_score
        ], False
    def _refine_queries(self, original, chunks):
        return [original]

        context_preview = current_chunks[0].chunk.content[:200] if current_chunks else ""
        try:
            prompt = (
                f"Original question: {original}\n"
                f"Current best result (not fully relevant): {context_preview}\n\n"
                "Generate 2 alternative search queries to find better information. "
                "One per line:"
            )
            result = self.llm.generate(prompt, max_tokens=100)
            refined = [l.strip() for l in result.split("\n") if l.strip()]
            return refined[:2] if refined else [original]
        except Exception:
            return [original]

    # ------------------------------------------------------------------ #
    # Answer Generation                                                    #
    # ------------------------------------------------------------------ #

    def _generate_answer(self, question: str, chunks: list) -> str:
        """Generate final answer using retrieved context."""
        from pathlib import Path

        context_parts = []
        for i, r in enumerate(chunks[:6], 1):
            chunk = r.chunk
            source = f"[{i}] Page {chunk.page_num + 1}, {Path(chunk.doc_path).name}"
            context_parts.append(f"{source}\n{chunk.content}")

        context = "\n\n---\n".join(context_parts)

        system = (
            "You are a precise technical assistant. "
            "Answer based ONLY on the provided context. "
            "Cite source numbers like [1], [2] when referencing specific information. "
            "If the answer is not in the context, say: 'Not found in the document.'"
        )

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            return self.llm.generate(prompt, max_tokens=600, system=system)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return ""

    def _estimate_confidence(self, chunks: list) -> float:
        """Estimate answer confidence from retrieval scores."""
        if not chunks:
            return 0.0
        top_score = max(c.rrf_score for c in chunks)
        avg_score = sum(c.rrf_score for c in chunks) / len(chunks)
        # Normalize to 0-1
        return min(1.0, (top_score * 0.7 + avg_score * 0.3) * 50)

    def clear_history(self):
        """Reset conversation history."""
        self._conversation_history = []
        logger.info("Conversation history cleared")