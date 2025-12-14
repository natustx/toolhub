"""Embedder for generating vector embeddings from chunks.

Uses sentence-transformers with all-MiniLM-L6-v2 by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from toolhub.indexer.chunker import Chunk

# Lazy load the model to avoid slow import on every CLI invocation
_model = None
_model_name: str | None = None


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy load the sentence transformer model."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
        _model_name = model_name
    return _model


@dataclass
class EmbeddedChunk:
    """A chunk with its embedding vector."""

    content: str
    source_file: str
    heading: str | None
    heading_path: str
    is_code: bool
    embedding: NDArray[np.float32]

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "source_file": self.source_file,
            "heading": self.heading or "",
            "heading_path": self.heading_path,
            "is_code": self.is_code,
            "vector": self.embedding.tolist(),
        }


def embed_text(
    text: str, model_name: str = "all-MiniLM-L6-v2", timings: dict | None = None
) -> NDArray[np.float32]:
    """Generate embedding for a single text string.

    Args:
        text: Text to embed
        model_name: Sentence transformer model to use
        timings: Optional dict to store timing breakdown

    Returns:
        Embedding vector as numpy array
    """
    import time

    t0 = time.perf_counter()
    model = _get_model(model_name)
    if timings is not None:
        timings["get_model"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    embedding = model.encode(text, convert_to_numpy=True)
    if timings is not None:
        timings["encode"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = embedding.astype(np.float32)
    if timings is not None:
        timings["astype"] = time.perf_counter() - t0

    return result


def embed_texts(
    texts: list[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32
) -> NDArray[np.float32]:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        model_name: Sentence transformer model to use
        batch_size: Batch size for encoding

    Returns:
        Array of embedding vectors (shape: [n_texts, embedding_dim])
    """
    model = _get_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
    return embeddings.astype(np.float32)


def embed_chunks(
    chunks: list[Chunk],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> list[EmbeddedChunk]:
    """Generate embeddings for a list of chunks.

    Creates searchable text from chunk content and metadata,
    then generates embeddings.

    Args:
        chunks: List of Chunk objects to embed
        model_name: Sentence transformer model to use
        batch_size: Batch size for encoding

    Returns:
        List of EmbeddedChunk objects with embeddings
    """
    if not chunks:
        return []

    # Create searchable text for each chunk
    # Include heading context for better semantic matching
    texts = []
    for chunk in chunks:
        searchable = chunk.content
        if chunk.heading_path:
            searchable = f"{chunk.heading_path}\n\n{chunk.content}"
        texts.append(searchable)

    # Generate embeddings in batch
    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)

    # Create EmbeddedChunk objects
    embedded = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded.append(
            EmbeddedChunk(
                content=chunk.content,
                source_file=chunk.source_file,
                heading=chunk.heading,
                heading_path=chunk.heading_path,
                is_code=chunk.is_code,
                embedding=embedding,
            )
        )

    return embedded


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """Get the embedding dimension for a model.

    Args:
        model_name: Sentence transformer model name

    Returns:
        Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
    """
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()
