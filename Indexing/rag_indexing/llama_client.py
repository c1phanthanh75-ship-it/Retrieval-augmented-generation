"""
Gemma 4 Vision Language Model Client
- Text generation (semantic descriptions, table summaries)
- Multi-modal embedding via Ollama
- Async batch processing
"""
import asyncio
import logging
from typing import Optional
import httpx

from .config import LlamaConfig
from .models import DocumentRegion, RegionType

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Sync/Async client for Ollama local server.
    Supports: text generation, vision inference, embeddings.
    """

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self._check_server()

    def _check_server(self):
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"Ollama server OK. Available models: {models}")
            if self.config.model not in models:
                logger.warning(
                    f"Model '{self.config.model}' not found. "
                    f"Pull it: ollama pull {self.config.model}"
                )
        except Exception as e:
            logger.warning(f"Ollama server check failed: {e}. Continuing anyway.")

    # ------------------------------------------------------------------ #
    # Text Generation                                                       #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        system: str = "",
        temperature: float = None,
    ) -> str:
        """Synchronous text generation."""
        import requests

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.config.context_length,
            },
        }
        if system:
            payload["system"] = system

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def generate_with_image(
        self,
        prompt: str,
        image_bytes: bytes,
        max_tokens: int = 512,
    ) -> str:
        """Vision inference: prompt + image → text."""
        import requests
        import base64

        img_b64 = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.config.context_length,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    # ------------------------------------------------------------------ #
    # Embeddings                                                            #
    # ------------------------------------------------------------------ #

    def embed_text(self, text: str) -> list[float]:
        """Get dense embedding for text using embed model."""
        import requests

        payload = {
            "model": self.config.embed_model,
            "prompt": text,
        }
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Ollama doesn't support true batch, so we loop."""
        embeddings = []
        for i, text in enumerate(texts):
            try:
                emb = self.embed_text(text)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Embedding failed for text {i}: {e}")
                embeddings.append([])
        return embeddings

    # ------------------------------------------------------------------ #
    # Async variants                                                        #
    # ------------------------------------------------------------------ #

    async def agenerate(self, prompt: str, max_tokens: int = 512) -> str:
        """Async text generation."""
        async with httpx.AsyncClient(timeout=180) as client:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.config.context_length,
                },
            }
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()

    async def aembed(self, text: str) -> list[float]:
        """Async embedding."""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.config.embed_model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def aembed_batch(
        self, texts: list[str], concurrency: int = 8
    ) -> list[list[float]]:
        """Async parallel batch embedding with semaphore."""
        sem = asyncio.Semaphore(concurrency)

        async def embed_one(text: str, idx: int) -> tuple[int, list[float]]:
            async with sem:
                try:
                    emb = await self.aembed(text)
                    return idx, emb
                except Exception as e:
                    logger.error(f"Async embed failed [{idx}]: {e}")
                    return idx, []

        tasks = [embed_one(t, i) for i, t in enumerate(texts)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]


class VLMEnricher:
    """
    Uses Gemma 4 Vision to generate rich semantic descriptions
    for images, tables, and complex regions.
    """

    SYSTEM_PROMPT = (
        "You are a document analysis assistant. "
        "Provide precise, information-dense descriptions suitable for semantic search. "
        "Be concise: 1-3 sentences max."
    )

    def __init__(self, client: OllamaClient):
        self.client = client

    def describe_image_region(self, region: DocumentRegion) -> str:
        """Generate description for a figure/image region."""
        if not region.image_data:
            return region.text or ""

        prompt = (
            "Describe what this image/figure shows. "
            "Focus on: type of visualization, key data points, trends, labels. "
            "Output: 2-3 sentences."
        )
        try:
            return self.client.generate_with_image(prompt, region.image_data)
        except Exception as e:
            logger.warning(f"VLM image description failed: {e}")
            return region.text or ""

    def describe_table(self, table_markdown: str) -> str:
        """Generate semantic description of a table."""
        prompt = (
            f"Summarize what this table contains and its key insights.\n\n"
            f"{table_markdown[:2000]}\n\n"
            "Output: 2-3 sentence summary."
        )
        try:
            return self.client.generate(prompt, max_tokens=200, system=self.SYSTEM_PROMPT)
        except Exception as e:
            logger.warning(f"Table description failed: {e}")
            return ""

    def enrich_chunk_context(self, chunk_text: str, doc_context: str = "") -> str:
        """
        Optionally generate a richer version of a chunk for embedding.
        Adds hypothetical questions the chunk answers (HyDE-style).
        """
        if not doc_context:
            return chunk_text

        prompt = (
            f"Given this document context: {doc_context[:500]}\n\n"
            f"Chunk: {chunk_text[:800]}\n\n"
            "Generate 2 questions this chunk answers (for search improvement). "
            "Return: questions only, one per line."
        )
        try:
            questions = self.client.generate(prompt, max_tokens=150)
            return f"{chunk_text}\n\nRelated questions:\n{questions}"
        except Exception:
            return chunk_text