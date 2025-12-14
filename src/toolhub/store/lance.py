"""LanceDB vector store wrapper for per-tool document storage."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import lancedb
import pyarrow as pa

from toolhub.paths import get_lance_dir, get_tool_lance_path

if TYPE_CHECKING:
    from toolhub.indexer.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

# Schema for the chunks table
CHUNKS_SCHEMA = pa.schema(
    [
        pa.field("content", pa.utf8()),
        pa.field("source_file", pa.utf8()),
        pa.field("heading", pa.utf8()),
        pa.field("heading_path", pa.utf8()),
        pa.field("is_code", pa.bool_()),
        pa.field("vector", pa.list_(pa.float32(), 384)),  # all-MiniLM-L6-v2 dimension
    ]
)

TABLE_NAME = "chunks"


class VectorStore:
    """LanceDB-backed vector store for a single tool's documentation."""

    def __init__(self, tool_id: str, db_path: Path | None = None):
        """Initialize vector store for a tool.

        Args:
            tool_id: Unique identifier for the tool
            db_path: Optional custom path for the LanceDB directory
        """
        self.tool_id = tool_id
        self.db_path = db_path or get_tool_lance_path(tool_id)
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    def _ensure_db(self) -> lancedb.DBConnection:
        """Ensure database connection is open."""
        if self._db is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _ensure_table(self) -> lancedb.table.Table:
        """Ensure chunks table exists and is open."""
        if self._table is not None:
            return self._table

        db = self._ensure_db()

        if TABLE_NAME in db.table_names():
            self._table = db.open_table(TABLE_NAME)
        else:
            # Create empty table with schema
            self._table = db.create_table(TABLE_NAME, schema=CHUNKS_SCHEMA)

        return self._table

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> int:
        """Add embedded chunks to the store.

        Args:
            chunks: List of EmbeddedChunk objects with embeddings

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        table = self._ensure_table()

        # Convert to records
        records = []
        for chunk in chunks:
            records.append(
                {
                    "content": chunk.content,
                    "source_file": chunk.source_file,
                    "heading": chunk.heading or "",
                    "heading_path": chunk.heading_path,
                    "is_code": chunk.is_code,
                    "vector": chunk.embedding.tolist(),
                }
            )

        table.add(records)
        logger.info(f"Added {len(records)} chunks to {self.tool_id}")
        return len(records)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[dict]:
        """Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of result dicts with content, metadata, and distance
        """
        table = self._ensure_table()

        results = table.search(query_vector).metric("cosine").limit(limit).to_list()

        # Convert to clean dicts
        return [
            {
                "content": r["content"],
                "source_file": r["source_file"],
                "heading": r["heading"],
                "heading_path": r["heading_path"],
                "is_code": r["is_code"],
                "distance": r["_distance"],
            }
            for r in results
        ]

    def count(self) -> int:
        """Get number of chunks in the store."""
        try:
            table = self._ensure_table()
            return table.count_rows()
        except Exception:
            return 0

    def clear(self) -> None:
        """Remove all chunks from the store."""
        db = self._ensure_db()
        if TABLE_NAME in db.table_names():
            db.drop_table(TABLE_NAME)
        self._table = None
        logger.info(f"Cleared all chunks from {self.tool_id}")

    def delete(self) -> None:
        """Delete the entire store for this tool."""
        self._table = None
        self._db = None
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
        logger.info(f"Deleted store for {self.tool_id}")

    def exists(self) -> bool:
        """Check if the store exists and has data."""
        return self.db_path.exists() and self.count() > 0


def list_tool_stores() -> list[str]:
    """List all tool IDs that have vector stores.

    Returns:
        List of tool IDs with .lance directories
    """
    lance_dir = get_lance_dir()
    if not lance_dir.exists():
        return []

    tool_ids = []
    for path in lance_dir.iterdir():
        if path.is_dir() and path.suffix == ".lance":
            tool_ids.append(path.stem)

    return sorted(tool_ids)


def delete_tool_store(tool_id: str) -> bool:
    """Delete a tool's vector store.

    Args:
        tool_id: Tool identifier

    Returns:
        True if store was deleted, False if it didn't exist
    """
    store = VectorStore(tool_id)
    if store.exists():
        store.delete()
        return True
    return False
