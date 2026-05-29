"""
Query Router Module
Phân loại intent của query và route đến strategy phù hợp:
- FACTUAL     → dense vector search
- ANALYTICAL  → hybrid (dense + BM25)
- FORMULA     → formula index search
- TABLE       → table index search  
- COMPARATIVE → multi-query + fusion
- CONVERSATIONAL → context-aware search
"""
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FACTUAL = "factual"           # "What is X?" → dense search
    ANALYTICAL = "analytical"     # "How does X work?" → hybrid
    FORMULA = "formula"           # "equation for X" → formula index
    TABLE = "table"               # "compare X and Y" → table index
    COMPARATIVE = "comparative"   # "difference between X and Y" → multi-query
    PROCEDURAL = "procedural"     # "How to do X?" → BM25 + dense
    CONVERSATIONAL = "conversational"  # follow-up → context-aware


class RetrievalStrategy(str, Enum):
    DENSE_ONLY = "dense_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    FORMULA_FIRST = "formula_first"
    TABLE_FIRST = "table_first"
    MULTI_QUERY = "multi_query"
    CONTEXTUAL = "contextual"


@dataclass
class RouteDecision:
    intent: QueryIntent
    strategy: RetrievalStrategy
    confidence: float = 1.0
    top_k: int = 10
    sub_queries: list[str] = field(default_factory=list)
    reasoning: str = ""
    filters: dict = field(default_factory=dict)   # e.g. {"chunk_type": "formula"}


# ------------------------------------------------------------------ #
# Rule-based patterns (fast, no LLM needed)                           #
# ------------------------------------------------------------------ #

FORMULA_PATTERNS = re.compile(
    r'\b(equation|formula|expression|derive|integral|derivative|'
    r'theorem|proof|calculate|compute|laplace|fourier|transfer function|'
    r'eigenvalue|matrix|vector|gradient|loss function|softmax|sigmoid)\b',
    re.IGNORECASE
)

TABLE_PATTERNS = re.compile(
    r'\b(table|compare|comparison|versus|vs\.?|difference between|'
    r'list of|summary|overview|parameters|specifications|benchmark)\b',
    re.IGNORECASE
)

COMPARATIVE_PATTERNS = re.compile(
    r'\b(difference|compare|versus|vs\.?|better|worse|advantage|'
    r'disadvantage|pros|cons|trade.?off|which is)\b',
    re.IGNORECASE
)

PROCEDURAL_PATTERNS = re.compile(
    r'\b(how to|steps|procedure|process|algorithm|implement|'
    r'tutorial|guide|example|code|program)\b',
    re.IGNORECASE
)

ANALYTICAL_PATTERNS = re.compile(
    r'\b(explain|describe|analyze|why|what is|define|'
    r'concept|theory|principle|mechanism|overview)\b',
    re.IGNORECASE
)

MATH_SYMBOLS = re.compile(
    r'[∑∫∂∇αβγδθλμσπ]|\$[^$]+\$|\\[a-zA-Z]+\{|=\s*\d'
)


class QueryRouter:
    """
    Routes queries to optimal retrieval strategy.
    Uses LLM for complex cases, rule-based for simple ones.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._conversation_history: list[dict] = []

    def route(self, query: str, conversation_history: list[dict] = None) -> RouteDecision:
        """
        Main routing method.
        Returns RouteDecision with strategy and sub-queries.
        """
        # 1. Check if follow-up (conversational)
        if conversation_history and self._is_followup(query, conversation_history):
            return self._route_conversational(query, conversation_history)

        # 2. Fast rule-based classification
        intent, confidence = self._classify_intent(query)

        # 3. LLM refinement for low-confidence or complex queries
        if confidence < 0.7 and self.llm is not None:
            intent, confidence = self._llm_classify(query, intent)

        # 4. Build strategy from intent
        decision = self._build_decision(query, intent, confidence)

        logger.info(
            f"Route: '{query[:50]}' → {intent.value} "
            f"[{decision.strategy.value}] conf={confidence:.2f}"
        )
        return decision

    # ------------------------------------------------------------------ #
    # Intent Classification                                                #
    # ------------------------------------------------------------------ #

    def _classify_intent(self, query: str) -> tuple[QueryIntent, float]:
        """Rule-based fast classification."""
        q = query.strip().lower()
        scores: dict[QueryIntent, float] = {}

        # Math symbols → formula
        if MATH_SYMBOLS.search(query):
            scores[QueryIntent.FORMULA] = 0.95

        if FORMULA_PATTERNS.search(q):
            scores[QueryIntent.FORMULA] = scores.get(QueryIntent.FORMULA, 0) + 0.6

        if TABLE_PATTERNS.search(q):
            scores[QueryIntent.TABLE] = 0.75

        if COMPARATIVE_PATTERNS.search(q):
            scores[QueryIntent.COMPARATIVE] = 0.80

        if PROCEDURAL_PATTERNS.search(q):
            scores[QueryIntent.PROCEDURAL] = 0.75

        if ANALYTICAL_PATTERNS.search(q):
            scores[QueryIntent.ANALYTICAL] = 0.65

        # Short factual queries
        if len(q.split()) <= 6 and q.endswith("?"):
            scores[QueryIntent.FACTUAL] = scores.get(QueryIntent.FACTUAL, 0) + 0.5

        if not scores:
            return QueryIntent.FACTUAL, 0.5

        best = max(scores, key=scores.get)
        return best, min(scores[best], 1.0)

    def _llm_classify(
        self, query: str, fallback: QueryIntent
    ) -> tuple[QueryIntent, float]:
        """Use LLM to classify ambiguous queries."""
        prompt = (
            "Classify this search query into exactly ONE category:\n"
            "- factual: asking for a specific fact or definition\n"
            "- analytical: asking for explanation or analysis\n"
            "- formula: asking about math equations or formulas\n"
            "- table: asking for comparison or tabular data\n"
            "- comparative: comparing multiple things\n"
            "- procedural: asking how to do something\n\n"
            f"Query: {query}\n\n"
            "Reply with only the category name, nothing else."
        )
        try:
            result = self.llm.generate(prompt, max_tokens=10).strip().lower()
            for intent in QueryIntent:
                if intent.value in result:
                    return intent, 0.85
        except Exception as e:
            logger.warning(f"LLM classify failed: {e}")

        return fallback, 0.5

    # ------------------------------------------------------------------ #
    # Decision Building                                                    #
    # ------------------------------------------------------------------ #

    def _build_decision(
        self, query: str, intent: QueryIntent, confidence: float
    ) -> RouteDecision:
        """Map intent → retrieval strategy + parameters."""

        if intent == QueryIntent.FORMULA:
            return RouteDecision(
                intent=intent,
                strategy=RetrievalStrategy.FORMULA_FIRST,
                confidence=confidence,
                top_k=8,
                filters={"chunk_type": "formula"},
                reasoning="Query involves math/formulas → search formula index first",
            )

        elif intent == QueryIntent.TABLE:
            return RouteDecision(
                intent=intent,
                strategy=RetrievalStrategy.TABLE_FIRST,
                confidence=confidence,
                top_k=5,
                filters={"chunk_type": "table"},
                reasoning="Query asks for comparison/table data",
            )

        elif intent == QueryIntent.COMPARATIVE:
            sub_queries = self._generate_sub_queries(query)
            return RouteDecision(
                intent=intent,
                strategy=RetrievalStrategy.MULTI_QUERY,
                confidence=confidence,
                top_k=10,
                sub_queries=sub_queries,
                reasoning="Comparative query → decompose into sub-queries",
            )

        elif intent == QueryIntent.PROCEDURAL:
            return RouteDecision(
                intent=intent,
                strategy=RetrievalStrategy.HYBRID,
                confidence=confidence,
                top_k=8,
                reasoning="Procedural query → BM25 good for keyword matching",
            )

        elif intent == QueryIntent.ANALYTICAL:
            return RouteDecision(
                intent=intent,
                strategy=RetrievalStrategy.HYBRID,
                confidence=confidence,
                top_k=10,
                reasoning="Analytical query → hybrid for broad coverage",
            )

        else:  # FACTUAL
            return RouteDecision(
                intent=intent,
                strategy=RetrievalStrategy.DENSE_ONLY,
                confidence=confidence,
                top_k=5,
                reasoning="Factual query → dense vector for semantic match",
            )

    def _generate_sub_queries(self, query: str) -> list[str]:
        """
        Decompose comparative query into sub-queries.
        e.g. "difference between Butterworth and Chebyshev" →
             ["Butterworth filter properties", "Chebyshev filter properties"]
        """
        if self.llm:
            try:
                prompt = (
                    f"Decompose this query into 2-3 simpler sub-queries for document search.\n"
                    f"Query: {query}\n"
                    "Output one sub-query per line, no numbering."
                )
                result = self.llm.generate(prompt, max_tokens=100)
                subs = [l.strip() for l in result.split("\n") if l.strip()]
                if subs:
                    return subs[:3]
            except Exception:
                pass

        # Fallback: extract key terms
        words = query.lower().split()
        connectors = {"between", "and", "or", "vs", "versus", "compared"}
        parts = []
        current = []
        for w in words:
            if w in connectors and current:
                parts.append(" ".join(current))
                current = []
            else:
                current.append(w)
        if current:
            parts.append(" ".join(current))

        return [p for p in parts if len(p.split()) >= 2] or [query]

    # ------------------------------------------------------------------ #
    # Conversational                                                       #
    # ------------------------------------------------------------------ #

    def _is_followup(self, query: str, history: list[dict]) -> bool:
        """Detect follow-up queries that need conversation context."""
        followup_indicators = [
            r'\b(it|this|that|they|these|those|its|their)\b',
            r'^(and|also|what about|how about|tell me more)',
            r'^(why|when|where|who)\b(?!.*\b(is|are|was|were)\b.*\b(the|a|an)\b)',
        ]
        q = query.lower()
        for pattern in followup_indicators:
            if re.search(pattern, q):
                return True
        return len(query.split()) <= 4 and not query.endswith("?")

    def _route_conversational(
        self, query: str, history: list[dict]
    ) -> RouteDecision:
        """Build context-aware query from conversation history."""
        # Expand query with recent context
        recent = history[-3:] if len(history) >= 3 else history
        context_terms = []
        for turn in recent:
            if turn.get("role") == "user":
                # Extract key nouns from previous queries
                words = turn["content"].split()
                context_terms.extend([w for w in words if len(w) > 4])

        expanded = f"{query} {' '.join(context_terms[-5:])}"

        return RouteDecision(
            intent=QueryIntent.CONVERSATIONAL,
            strategy=RetrievalStrategy.CONTEXTUAL,
            confidence=0.8,
            top_k=8,
            sub_queries=[expanded, query],
            reasoning=f"Follow-up query, expanded with context: {expanded[:60]}",
        )
