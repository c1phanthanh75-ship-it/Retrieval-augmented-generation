"""
Query Translation Module
5 kỹ thuật nâng cao chất lượng retrieval:

1. Multi-Query    : Tạo nhiều biến thể query → retrieve nhiều → merge
2. RAG Fusion     : Multi-Query + Reciprocal Rank Fusion
3. Decomposition  : Tách query phức tạp → sub-questions → answer each → synthesize
4. Step Back      : Tạo câu hỏi tổng quát hơn → retrieve background → answer
5. HyDE           : Tạo câu trả lời giả → embed → retrieve similar
"""
import logging
import re
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class QueryStrategy(str, Enum):
    ORIGINAL    = "original"      # Không transform
    MULTI_QUERY = "multi_query"   # Nhiều biến thể
    RAG_FUSION  = "rag_fusion"    # Multi-Query + RRF
    DECOMPOSE   = "decompose"     # Tách sub-questions
    STEP_BACK   = "step_back"     # Câu hỏi tổng quát hơn
    HYDE        = "hyde"          # Hypothetical Document Embedding
    AUTO        = "auto"          # Tự chọn dựa trên query


@dataclass
class TranslatedQuery:
    original: str
    strategy: QueryStrategy
    queries: list[str] = field(default_factory=list)       # Các query để retrieve
    hyde_doc: Optional[str] = None                          # HyDE: hypothetical document
    sub_questions: list[str] = field(default_factory=list)  # Decompose: sub-questions
    step_back_query: Optional[str] = None                   # Step-back query
    metadata: dict = field(default_factory=dict)


class QueryTranslator:
    """
    Áp dụng các kỹ thuật query translation trước khi retrieve.
    Dùng Llama 3 (Ollama) để sinh query variants.
    """

    def __init__(self, llm_client, strategy: QueryStrategy = QueryStrategy.AUTO):
        self.llm = llm_client
        self.strategy = strategy

    # ------------------------------------------------------------------ #
    # Main entry point                                                      #
    # ------------------------------------------------------------------ #

    def translate(self, query: str) -> TranslatedQuery:
        """
        Translate query theo strategy đã chọn.
        Trả về TranslatedQuery với danh sách queries để retrieve.
        """
        strategy = self.strategy
        if strategy == QueryStrategy.AUTO:
            strategy = self._auto_select(query)
            logger.info(f"Auto-selected strategy: {strategy.value}")

        logger.info(f"Query translation [{strategy.value}]: {query[:60]}")

        if strategy == QueryStrategy.ORIGINAL:
            return TranslatedQuery(
                original=query,
                strategy=strategy,
                queries=[query],
            )
        elif strategy == QueryStrategy.MULTI_QUERY:
            return self._multi_query(query)
        elif strategy == QueryStrategy.RAG_FUSION:
            return self._rag_fusion(query)
        elif strategy == QueryStrategy.DECOMPOSE:
            return self._decompose(query)
        elif strategy == QueryStrategy.STEP_BACK:
            return self._step_back(query)
        elif strategy == QueryStrategy.HYDE:
            return self._hyde(query)
        else:
            return TranslatedQuery(original=query, strategy=strategy, queries=[query])

    # ------------------------------------------------------------------ #
    # 1. Multi-Query                                                        #
    # ------------------------------------------------------------------ #

    def _multi_query(self, query: str, n: int = 4) -> TranslatedQuery:
        """
        Tạo N biến thể của query từ các góc nhìn khác nhau.
        Mỗi biến thể retrieve độc lập, kết quả merge lại.

        Ví dụ:
          Query: "What is a Butterworth filter?"
          →  "Butterworth filter definition and properties"
          →  "How does Butterworth filter work mathematically"
          →  "Butterworth filter frequency response characteristics"
          →  "Applications of Butterworth filter in signal processing"
        """
        prompt = f"""Generate {n} search queries for the same information need. Output ONLY the queries.
No introduction, no explanation, no numbering. One query per line.

Original: {query}

Query 1:"""

        response = self._call_llm(prompt, max_tokens=300)
        variants = self._parse_list(response)

        # Luôn include query gốc
        all_queries = [query] + variants[:n]
        all_queries = list(dict.fromkeys(all_queries))  # dedup giữ thứ tự

        logger.info(f"Multi-Query: {len(all_queries)} queries generated")
        return TranslatedQuery(
            original=query,
            strategy=QueryStrategy.MULTI_QUERY,
            queries=all_queries,
            metadata={"n_variants": len(variants)},
        )

    # ------------------------------------------------------------------ #
    # 2. RAG Fusion                                                         #
    # ------------------------------------------------------------------ #

    def _rag_fusion(self, query: str, n: int = 4) -> TranslatedQuery:
        """
        Multi-Query + Reciprocal Rank Fusion.
        Retrieve với nhiều queries, fuse kết quả bằng RRF.
        (Retrieval step xử lý RRF trong retriever)

        Paper: https://arxiv.org/abs/2402.03367
        """
        prompt = f"""Generate {n} search queries for document retrieval. Output ONLY the queries, nothing else.
No introduction, no explanation, no numbering. One query per line.

Topic: {query}

Query 1:"""

        response = self._call_llm(prompt, max_tokens=300)
        variants = self._parse_list(response)
        all_queries = [query] + variants[:n]
        all_queries = list(dict.fromkeys(all_queries))

        logger.info(f"RAG Fusion: {len(all_queries)} queries for RRF")
        return TranslatedQuery(
            original=query,
            strategy=QueryStrategy.RAG_FUSION,
            queries=all_queries,
            metadata={"fusion": "RRF", "n_queries": len(all_queries)},
        )

    # ------------------------------------------------------------------ #
    # 3. Decomposition                                                      #
    # ------------------------------------------------------------------ #

    def _decompose(self, query: str) -> TranslatedQuery:
        """
        Tách query phức tạp thành các sub-questions đơn giản hơn.
        Trả lời từng sub-question rồi tổng hợp.

        Ví dụ:
          Query: "Compare Butterworth and Chebyshev filters for audio applications"
          → "What are the properties of Butterworth filters?"
          → "What are the properties of Chebyshev filters?"
          → "What are the requirements for audio filter design?"
          → "How do ripple and rolloff affect audio quality?"
        """
        prompt = f"""Break down this complex question into 3-5 simpler sub-questions.
Each sub-question should be self-contained and answerable independently.
Together they should cover all aspects needed to answer the original question.
Return ONLY sub-questions, one per line.

Complex question: {query}

Sub-questions:"""

        response = self._call_llm(prompt, max_tokens=400)
        sub_questions = self._parse_list(response)

        if not sub_questions:
            sub_questions = [query]

        # Retrieve với cả sub-questions lẫn query gốc
        all_queries = [query] + sub_questions

        logger.info(f"Decomposition: {len(sub_questions)} sub-questions")
        return TranslatedQuery(
            original=query,
            strategy=QueryStrategy.DECOMPOSE,
            queries=all_queries,
            sub_questions=sub_questions,
            metadata={"n_sub_questions": len(sub_questions)},
        )

    # ------------------------------------------------------------------ #
    # 4. Step Back                                                          #
    # ------------------------------------------------------------------ #

    def _step_back(self, query: str) -> TranslatedQuery:
        """
        Tạo câu hỏi tổng quát hơn (step back) để retrieve background knowledge.
        Sau đó dùng cả background + original để answer.

        Paper: https://arxiv.org/abs/2310.06117

        Ví dụ:
          Query: "What is the cutoff frequency of a 3rd order Butterworth filter?"
          Step-back: "What are the principles of Butterworth filter design?"
        """
        prompt = f"""Given a specific question, generate a more general "step-back" question
that asks about the underlying principles, concepts, or background knowledge needed
to answer the specific question.

Specific question: {query}

Step-back question (more general, about underlying principles):"""

        step_back = self._call_llm(prompt, max_tokens=150).strip()
        if not step_back or len(step_back) < 5:
            step_back = query

        # Retrieve với cả step-back và query gốc
        all_queries = [step_back, query]

        logger.info(f"Step-back: '{step_back[:60]}'")
        return TranslatedQuery(
            original=query,
            strategy=QueryStrategy.STEP_BACK,
            queries=all_queries,
            step_back_query=step_back,
            metadata={"step_back": step_back},
        )

    # ------------------------------------------------------------------ #
    # 5. HyDE — Hypothetical Document Embedding                            #
    # ------------------------------------------------------------------ #

    def _hyde(self, query: str) -> TranslatedQuery:
        """
        Tạo một đoạn văn giả (hypothetical document) như thể đây là câu trả lời.
        Embed đoạn văn đó → tìm chunks tương tự trong vector store.

        Paper: https://arxiv.org/abs/2212.10496

        Ví dụ:
          Query: "What is a Butterworth filter?"
          HyDE doc: "A Butterworth filter is a type of signal processing filter
                     designed to have a frequency response that is as flat as possible
                     in the passband. It was first described in 1930 by the British
                     engineer Stephen Butterworth..."
        """
        prompt = f"""Write a short, factual paragraph (3-5 sentences) that directly answers
the following question. Write as if you are an expert writing a textbook passage.
Be specific and use technical terminology. Do NOT say "I think" or "I believe".

Question: {query}

Expert answer paragraph:"""

        hyde_doc = self._call_llm(prompt, max_tokens=300).strip()

        if not hyde_doc or len(hyde_doc) < 20:
            hyde_doc = query

        # Retrieve với hypothetical doc (richer semantic signal)
        # + query gốc để không bỏ sót
        all_queries = [hyde_doc, query]

        logger.info(f"HyDE doc generated: {hyde_doc[:80]}...")
        return TranslatedQuery(
            original=query,
            strategy=QueryStrategy.HYDE,
            queries=all_queries,
            hyde_doc=hyde_doc,
            metadata={"hyde_length": len(hyde_doc)},
        )

    # ------------------------------------------------------------------ #
    # Auto strategy selection                                               #
    # ------------------------------------------------------------------ #

    def _auto_select(self, query: str) -> QueryStrategy:
        """Tự chọn strategy phù hợp dựa trên đặc điểm của query."""
        q = query.lower().strip()
        words = q.split()

        # Query quá ngắn/mơ hồ (1-2 words) → original, không tốn LLM
        if len(words) <= 2:
            return QueryStrategy.ORIGINAL

        # Câu hỏi so sánh → Decompose
        compare_keywords = {"compare", "difference", "vs", "versus", "better", "pros", "cons", "tradeoff"}
        if any(w in compare_keywords for w in words):
            return QueryStrategy.DECOMPOSE

        # Câu hỏi cụ thể/số liệu → Step Back
        specific_keywords = {"specific", "exactly", "value", "number", "calculate", "compute", "formula"}
        if any(w in specific_keywords for w in words) or re.search(r'\d', q):
            return QueryStrategy.STEP_BACK

        # Query ngắn có nghĩa (3-5 words) → HyDE (sinh hypothetical doc giúp embed tốt hơn)
        if len(words) <= 5:
            return QueryStrategy.HYDE

        # Query dài, phức tạp → RAG Fusion
        if len(words) >= 15 or q.count("and") >= 2 or q.count(",") >= 2:
            return QueryStrategy.RAG_FUSION

        # Mặc định → Multi-Query
        return QueryStrategy.MULTI_QUERY

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        """Gọi Llama 3 qua Ollama, fallback về empty string nếu lỗi."""
        try:
            return self.llm.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

    def _parse_list(self, text: str) -> list[str]:
        """Parse danh sách queries từ LLM output, lọc preamble."""
        if not text:
            return []

        # Các pattern của preamble cần bỏ
        PREAMBLE_PATTERNS = [
            r"^here are",
            r"^the following",
            r"^below are",
            r"^i (will|have|can)",
            r"^sure",
            r"^certainly",
            r"^of course",
            r"^these are",
        ]

        lines = []
        for line in text.strip().split("\n"):
            # Bỏ numbering: "1. ", "- ", "* ", "Query 1:"
            line = re.sub(r"^(query\s*\d+[\:\.]?\s*|[\d\.\-\*\)\s]+)", "", line, flags=re.IGNORECASE).strip()
            line = line.strip('"\'')

            if not line or len(line) < 8:
                continue

            # Bỏ preamble
            is_preamble = any(re.match(p, line, re.IGNORECASE) for p in PREAMBLE_PATTERNS)
            if is_preamble:
                continue

            # Bỏ dòng quá dài (>200 chars) — thường là explanation
            if len(line) > 200:
                continue

            lines.append(line)
        return lines


# ------------------------------------------------------------------ #
# Query Translation + Retrieval integration                            #
# ------------------------------------------------------------------ #

class TranslationAwareRetriever:
    """
    Kết hợp QueryTranslator với HybridRetriever.
    Retrieve nhiều queries, fuse kết quả bằng RRF.
    """

    def __init__(self, retriever, translator: QueryTranslator, rrf_k: int = 60):
        self.retriever = retriever        # HybridRetriever
        self.translator = translator
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        strategy: Optional[QueryStrategy] = None,
    ):
        from .models import RetrievalResult

        # Override strategy nếu được chỉ định
        if strategy:
            self.translator.strategy = strategy

        # Translate query
        translated = self.translator.translate(query)

        # Retrieve với từng query variant
        all_results: dict[str, list[int]] = {}   # chunk_id → list of ranks
        chunk_map = {}

        for q_idx, q in enumerate(translated.queries):
            try:
                # Embed query
                q_emb = self.retriever.colbert._embedder.encode([q]).tolist()[0]
                results = self.retriever.retrieve(q, q_emb, top_k=top_k * 2)

                for rank, result in enumerate(results):
                    cid = result.chunk.id
                    all_results.setdefault(cid, []).append(rank + 1)
                    chunk_map[cid] = result.chunk

            except Exception as e:
                logger.warning(f"Retrieve failed for query variant {q_idx}: {e}")

        # RRF fusion across all query variants
        rrf_scores = {
            cid: sum(1.0 / (self.rrf_k + r) for r in ranks)
            for cid, ranks in all_results.items()
        }

        top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        final_results = []
        for rank, cid in enumerate(top_ids):
            chunk = chunk_map.get(cid)
            if chunk:
                final_results.append(RetrievalResult(
                    chunk=chunk,
                    rrf_score=rrf_scores[cid],
                    rank=rank + 1,
                ))

        logger.info(
            f"[{translated.strategy.value}] {len(translated.queries)} queries "
            f"→ {len(final_results)} results"
        )
        return final_results, translated