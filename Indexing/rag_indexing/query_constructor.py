"""
Query Construction Module
Biến đổi query gốc thành dạng tối ưu cho retrieval:
1. Query Rewriting      - sửa lỗi, làm rõ nghĩa
2. HyDE                 - Hypothetical Document Embedding
3. Step-back Prompting  - abstraction cho câu hỏi cụ thể
4. Multi-query          - sinh nhiều query variants
5. Query Expansion      - thêm synonyms/related terms
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConstructedQuery:
    """Kết quả sau khi construct query."""
    original: str
    rewritten: str                          # cleaned/clarified version
    hyde_document: str = ""                 # hypothetical answer document
    step_back: str = ""                     # abstract/broader version
    variants: list[str] = field(default_factory=list)   # multiple phrasings
    expanded_terms: list[str] = field(default_factory=list)
    final_queries: list[str] = field(default_factory=list)  # all queries to run

    def all_queries(self) -> list[str]:
        """Return deduplicated list of all query variants."""
        seen = set()
        result = []
        for q in [self.rewritten, self.hyde_document, self.step_back] + self.variants:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                result.append(q)
        return result


class QueryConstructor:
    """
    Constructs optimal queries for RAG retrieval.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def construct(
        self,
        query: str,
        strategy: str = "hybrid",
        use_hyde: bool = True,
        use_stepback: bool = True,
        use_multiquery: bool = True,
    ) -> ConstructedQuery:
        """
        Main construction pipeline.
        """
        result = ConstructedQuery(
            original=query,
            rewritten=self._rewrite(query),
        )

        if use_hyde and self.llm:
            result.hyde_document = self._hyde(result.rewritten)

        if use_stepback and self.llm:
            result.step_back = self._step_back(result.rewritten)

        if use_multiquery and self.llm:
            result.variants = self._multi_query(result.rewritten)
        else:
            result.variants = self._rule_based_variants(result.rewritten)

        result.expanded_terms = self._expand_terms(result.rewritten)
        result.final_queries = result.all_queries()

        logger.info(
            f"Query construction: '{query[:40]}' → "
            f"{len(result.final_queries)} variants"
        )
        return result

    # ------------------------------------------------------------------ #
    # 1. Query Rewriting                                                   #
    # ------------------------------------------------------------------ #

    def _rewrite(self, query: str) -> str:
        """
        Clean and clarify query.
        Rule-based: no LLM needed.
        """
        q = query.strip()

        # Fix common abbreviations
        abbreviations = {
            r'\bDSP\b': 'digital signal processing',
            r'\bFIR\b': 'finite impulse response filter',
            r'\bIIR\b': 'infinite impulse response filter',
            r'\bFFT\b': 'fast fourier transform',
            r'\bDFT\b': 'discrete fourier transform',
            r'\bLTI\b': 'linear time invariant system',
            r'\bSNR\b': 'signal to noise ratio',
            r'\bBPF\b': 'band pass filter',
            r'\bLPF\b': 'low pass filter',
            r'\bHPF\b': 'high pass filter',
            r'\bML\b': 'machine learning',
            r'\bNN\b': 'neural network',
            r'\bCNN\b': 'convolutional neural network',
            r'\bRNN\b': 'recurrent neural network',
        }
        for pattern, replacement in abbreviations.items():
            q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)

        # Remove filler words at start
        fillers = re.compile(
            r'^(please |can you |could you |tell me |i want to know |'
            r'what is the |explain the )',
            re.IGNORECASE
        )
        q = fillers.sub('', q).strip()

        # Ensure question ends properly
        if q and not q[-1] in '.?!':
            q = q + '?'

        # LLM rewrite for complex queries
        if self.llm and len(q.split()) > 10:
            try:
                prompt = (
                    "Rewrite this search query to be clearer and more specific "
                    "for a document retrieval system. Keep it concise.\n"
                    f"Original: {q}\n"
                    "Rewritten (one line only):"
                )
                rewritten = self.llm.generate(prompt, max_tokens=60).strip()
                if rewritten and len(rewritten) > 5:
                    return rewritten
            except Exception:
                pass

        return q

    # ------------------------------------------------------------------ #
    # 2. HyDE — Hypothetical Document Embedding                           #
    # ------------------------------------------------------------------ #

    def _hyde(self, query: str) -> str:
        """
        Generate a hypothetical answer document.
        The embedding of this doc often retrieves better chunks than the query itself.
        """
        if not self.llm:
            return ""
        try:
            prompt = (
                "Write a short, dense technical paragraph (3-4 sentences) that would "
                "directly answer the following question. Write as if from a textbook.\n\n"
                f"Question: {query}\n\n"
                "Paragraph:"
            )
            doc = self.llm.generate(prompt, max_tokens=200)
            if doc and len(doc.strip()) > 20:
                logger.debug(f"HyDE doc: {doc[:80]}...")
                return doc.strip()
        except Exception as e:
            logger.warning(f"HyDE failed: {e}")
        return ""

    # ------------------------------------------------------------------ #
    # 3. Step-back Prompting                                               #
    # ------------------------------------------------------------------ #

    def _step_back(self, query: str) -> str:
        """
        Generate a more abstract/general version of the query.
        Helps retrieve background context for specific questions.
        e.g. "What is the cutoff frequency of a 3rd order Butterworth?" →
             "What are the properties of Butterworth filters?"
        """
        if not self.llm:
            return ""
        try:
            prompt = (
                "Given this specific question, generate a more general/abstract "
                "version that would help find background information.\n\n"
                f"Specific: {query}\n"
                "General (one line):"
            )
            abstract = self.llm.generate(prompt, max_tokens=60).strip()
            if abstract and abstract.lower() != query.lower():
                return abstract
        except Exception as e:
            logger.warning(f"Step-back failed: {e}")
        return ""

    # ------------------------------------------------------------------ #
    # 4. Multi-Query Generation                                            #
    # ------------------------------------------------------------------ #

    def _multi_query(self, query: str) -> list[str]:
        """
        Generate multiple phrasings of the same query.
        Different phrasings retrieve different relevant chunks.
        """
        if not self.llm:
            return self._rule_based_variants(query)
        try:
            prompt = (
                "Generate 3 different ways to phrase this search query "
                "for a technical document retrieval system. "
                "Each should capture a different aspect.\n\n"
                f"Query: {query}\n\n"
                "Output 3 variants, one per line, no numbering:"
            )
            result = self.llm.generate(prompt, max_tokens=150)
            variants = [
                l.strip().lstrip("•-123456789. ")
                for l in result.split("\n")
                if l.strip() and l.strip() != query
            ]
            return variants[:3]
        except Exception as e:
            logger.warning(f"Multi-query failed: {e}")
            return self._rule_based_variants(query)

    def _rule_based_variants(self, query: str) -> list[str]:
        """Fallback: simple rule-based query variants."""
        variants = []
        q = query.rstrip("?").strip()

        # Keyword-only version
        stopwords = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do',
                     'of', 'in', 'for', 'to', 'and', 'or', 'are', 'why',
                     'when', 'where', 'which', 'with', 'about'}
        keywords = [w for w in q.lower().split() if w not in stopwords]
        if keywords:
            variants.append(" ".join(keywords))

        # Add "definition of" prefix for factual queries
        if any(q.lower().startswith(w) for w in ["what is", "define", "explain"]):
            core = re.sub(r'^(what is|define|explain)\s+', '', q, flags=re.IGNORECASE)
            variants.append(f"definition of {core}")
            variants.append(core)

        return variants[:2]

    # ------------------------------------------------------------------ #
    # 5. Term Expansion                                                    #
    # ------------------------------------------------------------------ #

    def _expand_terms(self, query: str) -> list[str]:
        """
        Add related technical terms to improve BM25 recall.
        Rule-based domain knowledge.
        """
        domain_synonyms = {
            "filter": ["filtering", "frequency response", "transfer function"],
            "butterworth": ["maximally flat", "butterworth polynomial"],
            "chebyshev": ["equiripple", "chebyshev polynomial"],
            "fourier": ["frequency domain", "spectrum", "DFT", "FFT"],
            "convolution": ["convolution sum", "linear filtering", "impulse response"],
            "sampling": ["nyquist", "sampling theorem", "aliasing", "sample rate"],
            "signal": ["waveform", "time series", "sequence"],
            "noise": ["SNR", "noise reduction", "denoising"],
            "neural network": ["deep learning", "backpropagation", "gradient descent"],
            "loss": ["cost function", "objective function", "cross entropy"],
            "embedding": ["vector representation", "latent space", "encoding"],
        }

        expanded = []
        q_lower = query.lower()
        for term, synonyms in domain_synonyms.items():
            if term in q_lower:
                expanded.extend(synonyms)

        return list(set(expanded))[:5]
