"""KnowledgeStore: Unified Postgres + S3 storage backend.

Replaces per-tool LanceDB stores and SQLite operations DB with a single
Postgres database (pgvector for vectors, FTS for keyword search) and
S3/MinIO for artifact storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import boto3
import jsonschema
import psycopg
from botocore.client import Config as BotoConfig
from jsonschema import ValidationError as JsonSchemaValidationError
from pgvector.psycopg import register_vector

from toolhub.config import Config

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)


class SourceStatus(str, Enum):
    """Status of a documentation source."""

    PENDING = "pending"
    CRAWLING = "crawling"
    INDEXED = "indexed"
    FAILED = "failed"


class ArtifactKind(str, Enum):
    """Types of source artifacts stored in S3."""

    RAW = "raw"
    EXTRACTED = "extracted"
    MANIFEST = "manifest"


@dataclass
class Source:
    """A documentation source (repo, website, llms.txt)."""

    id: uuid.UUID
    canonical_url: str
    source_type: str
    collection: str
    tags: list[str]
    status: SourceStatus
    sha: str | None
    fetched_at: datetime | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: tuple) -> Source:
        """Create Source from database row."""
        return cls(
            id=row[0],
            canonical_url=row[1],
            source_type=row[2],
            collection=row[3],
            tags=row[4] or [],
            status=SourceStatus(row[5]),
            sha=row[6],
            fetched_at=row[7],
            created_at=row[8],
            updated_at=row[9],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": str(self.id),
            "canonical_url": self.canonical_url,
            "source_type": self.source_type,
            "collection": self.collection,
            "tags": self.tags,
            "status": self.status.value,
            "sha": self.sha,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class SourceArtifact:
    """Reference to an artifact stored in S3."""

    id: uuid.UUID
    source_id: uuid.UUID
    kind: ArtifactKind
    s3_key: str
    content_type: str | None
    size_bytes: int | None
    sha256: str | None
    created_at: datetime


@dataclass
class Chunk:
    """A chunk of documentation with embedding."""

    id: uuid.UUID
    source_id: uuid.UUID
    content: str
    heading: str | None
    heading_path: str | None
    source_file: str | None
    is_code: bool
    chunk_index: int
    model_id: str
    embedding: list[float] | None
    created_at: datetime


class EntityValidationError(Exception):
    """Raised when entity profile fails schema validation."""

    def __init__(self, message: str, field_path: str | None = None, schema_path: str | None = None):
        super().__init__(message)
        self.field_path = field_path
        self.schema_path = schema_path

    def __str__(self) -> str:
        parts = [str(self.args[0])]
        if self.field_path:
            parts.append(f"at {self.field_path}")
        return " ".join(parts)


@dataclass
class EntityType:
    """A registered entity type with JSON Schema."""

    id: uuid.UUID
    type_key: str
    json_schema: dict
    schema_version: int
    description: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: tuple) -> EntityType:
        """Create EntityType from database row."""
        return cls(
            id=row[0],
            type_key=row[1],
            json_schema=row[2],
            schema_version=row[3],
            description=row[4],
            created_at=row[5],
            updated_at=row[6],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": str(self.id),
            "type_key": self.type_key,
            "json_schema": self.json_schema,
            "schema_version": self.schema_version,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Entity:
    """An entity instance with typed profile."""

    id: uuid.UUID
    type_key: str
    name: str
    profile: dict
    tags: list[str]
    collection: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: tuple) -> Entity:
        """Create Entity from database row."""
        return cls(
            id=row[0],
            type_key=row[1],
            name=row[2],
            profile=row[3] or {},
            tags=row[4] or [],
            collection=row[5],
            created_at=row[6],
            updated_at=row[7],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": str(self.id),
            "type_key": self.type_key,
            "name": self.name,
            "profile": self.profile,
            "tags": self.tags,
            "collection": self.collection,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Evidence:
    """A link between an entity field and source material."""

    id: uuid.UUID
    entity_id: uuid.UUID
    field_path: str
    chunk_id: uuid.UUID | None
    source_id: uuid.UUID | None
    quote: str | None
    locator: str | None
    confidence: float | None
    created_at: datetime

    @classmethod
    def from_row(cls, row: tuple) -> Evidence:
        """Create Evidence from database row."""
        return cls(
            id=row[0],
            entity_id=row[1],
            field_path=row[2],
            chunk_id=row[3],
            source_id=row[4],
            quote=row[5],
            locator=row[6],
            confidence=row[7],
            created_at=row[8],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "id": str(self.id),
            "entity_id": str(self.entity_id),
            "field_path": self.field_path,
            "chunk_id": str(self.chunk_id) if self.chunk_id else None,
            "source_id": str(self.source_id) if self.source_id else None,
            "quote": self.quote,
            "locator": self.locator,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Citation:
    """A rendered citation for an entity field."""

    field_path: str
    quote: str | None
    source_url: str
    source_file: str | None
    heading_path: str | None
    chunk_content: str | None
    confidence: float | None

    def to_markdown(self) -> str:
        """Render citation as markdown."""
        parts = []
        if self.quote:
            parts.append(f'> "{self.quote}"')
        location = self.source_file or self.source_url
        if self.heading_path:
            location = f"{location} > {self.heading_path}"
        parts.append(f"— *{location}*")
        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "field_path": self.field_path,
            "quote": self.quote,
            "source_url": self.source_url,
            "source_file": self.source_file,
            "heading_path": self.heading_path,
            "chunk_content": self.chunk_content,
            "confidence": self.confidence,
        }


@dataclass
class SearchResult:
    """Result from hybrid search."""

    chunk_id: uuid.UUID
    source_id: uuid.UUID
    content: str
    heading: str | None
    heading_path: str | None
    source_file: str | None
    is_code: bool
    similarity: float
    canonical_url: str
    collection: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "chunk_id": str(self.chunk_id),
            "source_id": str(self.source_id),
            "content": self.content,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "source_file": self.source_file,
            "is_code": self.is_code,
            "similarity": round(self.similarity, 4),
            "canonical_url": self.canonical_url,
            "collection": self.collection,
        }

    def to_markdown(self) -> str:
        """Format as markdown for CLI output."""
        lines = []
        heading_info = f" > {self.heading_path}" if self.heading_path else ""
        source_info = self.source_file or self.canonical_url
        lines.append(f"### {source_info}{heading_info}")
        lines.append(f"*Similarity: {self.similarity:.2%}*")
        lines.append("")

        if self.is_code and not self.content.startswith("```"):
            lines.append("```")
            lines.append(self.content)
            lines.append("```")
        else:
            lines.append(self.content)

        lines.append("")
        return "\n".join(lines)


@dataclass
class SearchResponse:
    """Response from search operation."""

    query: str
    results: list[SearchResult]
    collection: str | None
    total_chunks_searched: int
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "query": self.query,
            "collection": self.collection,
            "result_count": len(self.results),
            "total_chunks_searched": self.total_chunks_searched,
            "results": [r.to_dict() for r in self.results],
            "timings": self.timings,
        }

    def to_markdown(self) -> str:
        """Format as markdown for CLI output."""
        lines = []
        lines.append(f"# Search: {self.query}")
        if self.collection:
            lines.append(f"*Collection: {self.collection}*")
        lines.append(f"*Found {len(self.results)} results*")
        lines.append("")

        for result in self.results:
            lines.append(result.to_markdown())

        return "\n".join(lines)


class KnowledgeStore:
    """Unified storage backend using Postgres + S3."""

    def __init__(self, config: Config, env: str = "dev"):
        """Initialize with config and environment prefix for S3 keys."""
        self.config = config
        self.env = env
        self._conn: psycopg.Connection | None = None
        self._s3: S3Client | None = None
        # In-process schema cache: type_key -> (schema_version, json_schema)
        self._schema_cache: dict[str, tuple[int, dict]] = {}

    def _get_conn(self) -> psycopg.Connection:
        """Get or create database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.config.postgres.url)
            register_vector(self._conn)
        return self._conn

    def _get_s3(self) -> S3Client:
        """Get or create S3 client."""
        if self._s3 is None:
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.config.s3.endpoint_url,
                aws_access_key_id=self.config.s3.access_key,
                aws_secret_access_key=self.config.s3.secret_key,
                config=BotoConfig(signature_version="s3v4"),
                region_name=self.config.s3.region,
            )
        return self._s3

    def close(self) -> None:
        """Close connections."""
        if self._conn and not self._conn.closed:
            self._conn.close()
        self._conn = None

    # === Source operations ===

    def add_source(
        self,
        canonical_url: str,
        source_type: str,
        collection: str = "default",
        tags: list[str] | None = None,
        status: SourceStatus = SourceStatus.PENDING,
    ) -> Source:
        """Insert or update a source record.

        Returns the created/updated Source with its assigned UUID.
        """
        conn = self._get_conn()
        tags = tags or []

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sources (canonical_url, source_type, collection, tags, status)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (canonical_url, collection)
                DO UPDATE SET
                    source_type = EXCLUDED.source_type,
                    tags = EXCLUDED.tags,
                    status = EXCLUDED.status,
                    updated_at = NOW()
                RETURNING id, canonical_url, source_type, collection, tags, status,
                          sha, fetched_at, created_at, updated_at
                """,
                (canonical_url, source_type, collection, tags, status.value),
            )
            row = cur.fetchone()
            conn.commit()

        logger.info("Added source: %s in collection %s", canonical_url, collection)
        return Source.from_row(row)

    def get_source(self, source_id: uuid.UUID) -> Source | None:
        """Get a source by ID."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, canonical_url, source_type, collection, tags, status,
                       sha, fetched_at, created_at, updated_at
                FROM sources WHERE id = %s
                """,
                (source_id,),
            )
            row = cur.fetchone()
        return Source.from_row(row) if row else None

    def update_source_status(
        self,
        source_id: uuid.UUID,
        status: SourceStatus,
        sha: str | None = None,
        fetched_at: datetime | None = None,
    ) -> None:
        """Update source status and metadata."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sources
                SET status = %s, sha = COALESCE(%s, sha),
                    fetched_at = COALESCE(%s, fetched_at), updated_at = NOW()
                WHERE id = %s
                """,
                (status.value, sha, fetched_at, source_id),
            )
            conn.commit()

    def list_sources(
        self,
        collection: str | None = None,
        tags: list[str] | None = None,
        status: SourceStatus | None = None,
    ) -> list[Source]:
        """List sources with optional filters."""
        conn = self._get_conn()
        conditions = []
        params: list = []

        if collection:
            conditions.append("collection = %s")
            params.append(collection)
        if tags:
            conditions.append("tags @> %s")
            params.append(tags)
        if status:
            conditions.append("status = %s")
            params.append(status.value)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, canonical_url, source_type, collection, tags, status,
                       sha, fetched_at, created_at, updated_at
                FROM sources
                WHERE {where_clause}
                ORDER BY updated_at DESC
                """,
                params,
            )
            rows = cur.fetchall()

        return [Source.from_row(row) for row in rows]

    def delete_source(self, source_id: uuid.UUID) -> bool:
        """Delete a source and all related data (cascades to artifacts/chunks)."""
        conn = self._get_conn()

        # Delete S3 artifacts first
        artifacts = self._list_artifacts(source_id)
        s3 = self._get_s3()
        for artifact in artifacts:
            try:
                s3.delete_object(Bucket=self.config.s3.bucket, Key=artifact.s3_key)
            except Exception as e:
                logger.warning("Failed to delete S3 object %s: %s", artifact.s3_key, e)

        with conn.cursor() as cur:
            cur.execute("DELETE FROM sources WHERE id = %s RETURNING id", (source_id,))
            deleted = cur.fetchone() is not None
            conn.commit()

        return deleted

    # === Artifact operations (S3) ===

    def _s3_key(self, source: Source, kind: ArtifactKind, extension: str = "") -> str:
        """Generate S3 key for an artifact."""
        return f"{self.env}/{source.collection}/{source.id}/{kind.value}{extension}"

    def store_artifact(
        self,
        source: Source,
        kind: ArtifactKind,
        content: bytes,
        content_type: str = "application/octet-stream",
        extension: str = "",
    ) -> SourceArtifact:
        """Upload artifact to S3 and record in database."""
        s3 = self._get_s3()
        conn = self._get_conn()

        s3_key = self._s3_key(source, kind, extension)
        sha256 = hashlib.sha256(content).hexdigest()

        # Upload to S3
        s3.put_object(
            Bucket=self.config.s3.bucket,
            Key=s3_key,
            Body=content,
            ContentType=content_type,
        )

        # Record in database
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO source_artifacts
                    (source_id, kind, s3_key, content_type, size_bytes, sha256)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_id, kind)
                DO UPDATE SET
                    s3_key = EXCLUDED.s3_key,
                    content_type = EXCLUDED.content_type,
                    size_bytes = EXCLUDED.size_bytes,
                    sha256 = EXCLUDED.sha256,
                    created_at = NOW()
                RETURNING id, source_id, kind, s3_key, content_type, size_bytes, sha256, created_at
                """,
                (source.id, kind.value, s3_key, content_type, len(content), sha256),
            )
            row = cur.fetchone()
            conn.commit()

        logger.debug("Stored artifact %s for source %s", kind.value, source.id)
        return SourceArtifact(
            id=row[0],
            source_id=row[1],
            kind=ArtifactKind(row[2]),
            s3_key=row[3],
            content_type=row[4],
            size_bytes=row[5],
            sha256=row[6],
            created_at=row[7],
        )

    def get_artifact(self, source: Source, kind: ArtifactKind) -> bytes | None:
        """Retrieve artifact content from S3."""
        conn = self._get_conn()
        s3 = self._get_s3()

        with conn.cursor() as cur:
            cur.execute(
                "SELECT s3_key FROM source_artifacts WHERE source_id = %s AND kind = %s",
                (source.id, kind.value),
            )
            row = cur.fetchone()

        if not row:
            return None

        try:
            response = s3.get_object(Bucket=self.config.s3.bucket, Key=row[0])
            return response["Body"].read()
        except s3.exceptions.NoSuchKey:
            logger.warning("Artifact %s not found in S3 for source %s", kind.value, source.id)
            return None

    def _list_artifacts(self, source_id: uuid.UUID) -> list[SourceArtifact]:
        """List all artifacts for a source."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source_id, kind, s3_key, content_type, size_bytes, sha256, created_at
                FROM source_artifacts WHERE source_id = %s
                """,
                (source_id,),
            )
            rows = cur.fetchall()

        return [
            SourceArtifact(
                id=row[0],
                source_id=row[1],
                kind=ArtifactKind(row[2]),
                s3_key=row[3],
                content_type=row[4],
                size_bytes=row[5],
                sha256=row[6],
                created_at=row[7],
            )
            for row in rows
        ]

    # === Chunk operations ===

    def add_chunk(
        self,
        source: Source,
        content: str,
        chunk_index: int,
        embedding: Sequence[float],
        model_id: str,
        heading: str | None = None,
        heading_path: str | None = None,
        source_file: str | None = None,
        is_code: bool = False,
    ) -> uuid.UUID:
        """Add a chunk with its embedding vector."""
        conn = self._get_conn()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks
                    (source_id, content, heading, heading_path, source_file,
                     is_code, chunk_index, model_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    source.id,
                    content,
                    heading,
                    heading_path,
                    source_file,
                    is_code,
                    chunk_index,
                    model_id,
                    list(embedding),
                ),
            )
            chunk_id = cur.fetchone()[0]
            conn.commit()

        return chunk_id

    def add_chunks_batch(
        self,
        source: Source,
        chunks: list[dict],
        model_id: str,
    ) -> list[uuid.UUID]:
        """Add multiple chunks in a single transaction for efficiency."""
        conn = self._get_conn()
        chunk_ids = []

        with conn.cursor() as cur:
            for i, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO chunks
                        (source_id, content, heading, heading_path, source_file,
                         is_code, chunk_index, model_id, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        source.id,
                        chunk["content"],
                        chunk.get("heading"),
                        chunk.get("heading_path"),
                        chunk.get("source_file"),
                        chunk.get("is_code", False),
                        i,
                        model_id,
                        list(chunk["embedding"]),
                    ),
                )
                chunk_ids.append(cur.fetchone()[0])
            conn.commit()

        logger.info("Added %d chunks for source %s", len(chunk_ids), source.id)
        return chunk_ids

    def delete_chunks(self, source_id: uuid.UUID) -> int:
        """Delete all chunks for a source."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
            deleted = cur.rowcount
            conn.commit()
        return deleted

    # === Search operations ===

    def search(
        self,
        query_embedding: Sequence[float],
        query_text: str | None = None,
        collection: str | None = None,
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_similarity: float = 0.0,
        semantic_weight: float = 0.7,
    ) -> SearchResponse:
        """Hybrid search combining vector similarity and FTS.

        Args:
            query_embedding: Query vector from embedding model
            query_text: Original query text for FTS (optional)
            collection: Filter to specific collection (exact match)
            collections: Filter to any of these collections (OR semantics)
            tags: Filter to sources with all these tags (AND semantics)
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0-1)
            semantic_weight: Weight for semantic vs keyword (0.7 = 70% semantic)

        Returns:
            SearchResponse with ranked results
        """
        import time

        conn = self._get_conn()
        timings: dict[str, float] = {}

        # Build filter conditions
        conditions = []
        params: list = [list(query_embedding)]

        if collection:
            conditions.append("s.collection = %s")
            params.append(collection)
        elif collections:
            # OR semantics: match any of the listed collections
            conditions.append("s.collection = ANY(%s)")
            params.append(collections)
        if tags:
            conditions.append("s.tags @> %s")
            params.append(tags)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        start = time.perf_counter()

        # Use hybrid search if we have query text, otherwise vector-only
        if query_text:
            # Hybrid: combine vector similarity with FTS
            keyword_weight = 1.0 - semantic_weight

            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH vector_results AS (
                        SELECT c.id, c.source_id, c.content, c.heading, c.heading_path,
                               c.source_file, c.is_code,
                               1 - (c.embedding <=> %s::vector) AS vec_similarity
                        FROM chunks c
                        JOIN sources s ON c.source_id = s.id
                        WHERE {where_clause}
                          AND c.embedding IS NOT NULL
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                    ),
                    fts_results AS (
                        SELECT c.id,
                            ts_rank(c.search_vector, plainto_tsquery('english', %s)) AS fts_rank
                        FROM chunks c
                        JOIN sources s ON c.source_id = s.id
                        WHERE {where_clause}
                          AND c.search_vector @@ plainto_tsquery('english', %s)
                    ),
                    combined AS (
                        SELECT
                            v.id, v.source_id, v.content, v.heading, v.heading_path,
                            v.source_file, v.is_code,
                            v.vec_similarity * %s + COALESCE(f.fts_rank, 0) * %s AS final_score,
                            s.canonical_url, s.collection
                        FROM vector_results v
                        JOIN sources s ON v.source_id = s.id
                        LEFT JOIN fts_results f ON v.id = f.id
                    )
                    SELECT * FROM combined
                    WHERE final_score >= %s
                    ORDER BY final_score DESC
                    LIMIT %s
                    """,
                    (
                        *params,  # embedding + filters for vector
                        query_embedding,  # ORDER BY
                        limit * 2,  # Over-fetch for hybrid merge
                        query_text,  # FTS query
                        *params[1:],  # filters for FTS (skip embedding)
                        query_text,  # FTS match
                        semantic_weight,
                        keyword_weight,
                        min_similarity,
                        limit,
                    ),
                )
                rows = cur.fetchall()
        else:
            # Vector-only search
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT c.id, c.source_id, c.content, c.heading, c.heading_path,
                           c.source_file, c.is_code,
                           1 - (c.embedding <=> %s::vector) AS similarity,
                           s.canonical_url, s.collection
                    FROM chunks c
                    JOIN sources s ON c.source_id = s.id
                    WHERE {where_clause}
                      AND c.embedding IS NOT NULL
                      AND 1 - (c.embedding <=> %s::vector) >= %s
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (*params, query_embedding, min_similarity, query_embedding, limit),
                )
                rows = cur.fetchall()

        timings["search"] = time.perf_counter() - start

        # Get total chunks count
        with conn.cursor() as cur:
            if collection:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM chunks c
                    JOIN sources s ON c.source_id = s.id
                    WHERE s.collection = %s
                    """,
                    (collection,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cur.fetchone()[0]

        results = [
            SearchResult(
                chunk_id=row[0],
                source_id=row[1],
                content=row[2],
                heading=row[3],
                heading_path=row[4],
                source_file=row[5],
                is_code=row[6],
                similarity=row[7],
                canonical_url=row[8],
                collection=row[9],
            )
            for row in rows
        ]

        return SearchResponse(
            query=query_text or "(vector search)",
            results=results,
            collection=collection,
            total_chunks_searched=total_chunks,
            timings=timings,
        )

    def get_chunk_count(self, source_id: uuid.UUID | None = None) -> int:
        """Get total chunk count, optionally filtered by source."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            if source_id:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE source_id = %s", (source_id,))
            else:
                cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]

    # === Entity Type operations ===

    def register_entity_type(
        self,
        type_key: str,
        json_schema: dict,
        description: str | None = None,
    ) -> EntityType:
        """Register or update an entity type schema.

        If the type_key exists and schema differs, increments schema_version.
        Invalidates cache on any change.

        Args:
            type_key: Unique identifier like 'competitor', 'wisdom'
            json_schema: JSON Schema dict for validating entity profiles
            description: Optional human-readable description

        Returns:
            The created or updated EntityType
        """
        conn = self._get_conn()

        # Check if type exists and if schema changed
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._ENTITY_TYPE_COLS} FROM entity_types WHERE type_key = %s",
                (type_key,),
            )
            existing_row = cur.fetchone()

        if existing_row:
            existing = EntityType.from_row(existing_row)
            # Compare schemas to decide if version bump needed
            existing_json = json.dumps(existing.json_schema, sort_keys=True)
            new_json = json.dumps(json_schema, sort_keys=True)

            if existing_json != new_json:
                # Schema changed - bump version
                new_version = existing.schema_version + 1
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        UPDATE entity_types
                        SET json_schema = %s, schema_version = %s,
                            description = COALESCE(%s, description)
                        WHERE type_key = %s
                        RETURNING {self._ENTITY_TYPE_COLS}
                        """,
                        (json.dumps(json_schema), new_version, description, type_key),
                    )
                    row = cur.fetchone()
                    conn.commit()
                logger.info("Updated entity type %s to version %d", type_key, new_version)
            else:
                # Schema unchanged - maybe update description only
                if description and description != existing.description:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"""
                            UPDATE entity_types SET description = %s WHERE type_key = %s
                            RETURNING {self._ENTITY_TYPE_COLS}
                            """,
                            (description, type_key),
                        )
                        row = cur.fetchone()
                        conn.commit()
                else:
                    row = existing_row
        else:
            # New type
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO entity_types (type_key, json_schema, description)
                    VALUES (%s, %s, %s)
                    RETURNING {self._ENTITY_TYPE_COLS}
                    """,
                    (type_key, json.dumps(json_schema), description),
                )
                row = cur.fetchone()
                conn.commit()
            logger.info("Created entity type %s", type_key)

        entity_type = EntityType.from_row(row)
        # Invalidate cache for this type
        self._schema_cache.pop(type_key, None)
        return entity_type

    _ENTITY_TYPE_COLS = (
        "id, type_key, json_schema, schema_version, description, created_at, updated_at"
    )

    def get_entity_type(self, type_key: str) -> EntityType | None:
        """Get an entity type by key."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._ENTITY_TYPE_COLS} FROM entity_types WHERE type_key = %s",
                (type_key,),
            )
            row = cur.fetchone()
        return EntityType.from_row(row) if row else None

    def list_entity_types(self) -> list[EntityType]:
        """List all registered entity types."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT {self._ENTITY_TYPE_COLS} FROM entity_types ORDER BY type_key")
            rows = cur.fetchall()
        return [EntityType.from_row(row) for row in rows]

    def _get_cached_schema(self, type_key: str) -> dict:
        """Get schema from cache or database.

        Raises KeyError if type_key doesn't exist.
        """
        # Check cache first
        if type_key in self._schema_cache:
            return self._schema_cache[type_key][1]

        # Load from database
        entity_type = self.get_entity_type(type_key)
        if entity_type is None:
            raise KeyError(f"Entity type '{type_key}' not registered")

        # Cache it
        self._schema_cache[type_key] = (entity_type.schema_version, entity_type.json_schema)
        return entity_type.json_schema

    def _validate_profile(self, type_key: str, profile: dict) -> None:
        """Validate a profile against its entity type schema.

        Raises EntityValidationError with field path on failure.
        """
        schema = self._get_cached_schema(type_key)

        try:
            jsonschema.validate(instance=profile, schema=schema)
        except JsonSchemaValidationError as e:
            # Convert to our error with field path
            field_path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "$"
            raise EntityValidationError(
                message=e.message,
                field_path=field_path,
                schema_path=".".join(str(p) for p in e.schema_path) if e.schema_path else None,
            ) from e

    # === Entity operations ===

    def create_entity(
        self,
        type_key: str,
        name: str,
        profile: dict | None = None,
        tags: list[str] | None = None,
        collection: str = "default",
    ) -> Entity:
        """Create a new entity with validated profile.

        Args:
            type_key: The entity type (must be registered)
            name: Entity name (unique within type_key + collection)
            profile: JSONB profile data (validated against schema)
            tags: Optional tags for filtering
            collection: Collection for multi-tenancy

        Returns:
            The created Entity

        Raises:
            EntityValidationError: If profile doesn't match schema
            KeyError: If type_key not registered
        """
        profile = profile or {}
        tags = tags or []

        # Validate profile against schema
        self._validate_profile(type_key, profile)

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO entities (type_key, name, profile, tags, collection)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, type_key, name, profile, tags, collection, created_at, updated_at
                """,
                (type_key, name, json.dumps(profile), tags, collection),
            )
            row = cur.fetchone()
            conn.commit()

        logger.debug("Created entity %s of type %s", name, type_key)
        return Entity.from_row(row)

    def get_entity(self, entity_id: uuid.UUID) -> Entity | None:
        """Get an entity by ID."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type_key, name, profile, tags, collection, created_at, updated_at
                FROM entities WHERE id = %s
                """,
                (entity_id,),
            )
            row = cur.fetchone()
        return Entity.from_row(row) if row else None

    def get_entity_by_name(
        self,
        type_key: str,
        name: str,
        collection: str = "default",
    ) -> Entity | None:
        """Get an entity by type_key + name + collection."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, type_key, name, profile, tags, collection, created_at, updated_at
                FROM entities WHERE type_key = %s AND name = %s AND collection = %s
                """,
                (type_key, name, collection),
            )
            row = cur.fetchone()
        return Entity.from_row(row) if row else None

    def update_entity(
        self,
        entity_id: uuid.UUID,
        profile: dict | None = None,
        tags: list[str] | None = None,
    ) -> Entity:
        """Update an entity's profile and/or tags.

        Args:
            entity_id: The entity to update
            profile: New profile (validated against schema, replaces existing)
            tags: New tags (replaces existing)

        Returns:
            The updated Entity

        Raises:
            EntityValidationError: If profile doesn't match schema
            KeyError: If entity doesn't exist
        """
        conn = self._get_conn()

        # Get existing entity to know type_key
        existing = self.get_entity(entity_id)
        if existing is None:
            raise KeyError(f"Entity {entity_id} not found")

        # Validate new profile if provided
        new_profile = profile if profile is not None else existing.profile
        self._validate_profile(existing.type_key, new_profile)

        new_tags = tags if tags is not None else existing.tags

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE entities
                SET profile = %s, tags = %s
                WHERE id = %s
                RETURNING id, type_key, name, profile, tags, collection, created_at, updated_at
                """,
                (json.dumps(new_profile), new_tags, entity_id),
            )
            row = cur.fetchone()
            conn.commit()

        return Entity.from_row(row)

    def delete_entity(self, entity_id: uuid.UUID) -> bool:
        """Delete an entity and its evidence (cascades)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM entities WHERE id = %s RETURNING id", (entity_id,))
            deleted = cur.fetchone() is not None
            conn.commit()
        return deleted

    def list_entities(
        self,
        type_key: str | None = None,
        collection: str | None = None,
        tags: list[str] | None = None,
    ) -> list[Entity]:
        """List entities with optional filters."""
        conn = self._get_conn()
        conditions = []
        params: list = []

        if type_key:
            conditions.append("type_key = %s")
            params.append(type_key)
        if collection:
            conditions.append("collection = %s")
            params.append(collection)
        if tags:
            conditions.append("tags @> %s")
            params.append(tags)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, type_key, name, profile, tags, collection, created_at, updated_at
                FROM entities
                WHERE {where_clause}
                ORDER BY name
                """,
                params,
            )
            rows = cur.fetchall()

        return [Entity.from_row(row) for row in rows]

    # === Evidence operations ===

    _EVIDENCE_COLS = (
        "id, entity_id, field_path, chunk_id, source_id, quote, locator, confidence, created_at"
    )

    def add_evidence(
        self,
        entity_id: uuid.UUID,
        field_path: str,
        chunk_id: uuid.UUID | None = None,
        source_id: uuid.UUID | None = None,
        quote: str | None = None,
        locator: str | None = None,
        confidence: float | None = None,
    ) -> Evidence:
        """Add evidence linking an entity field to source material.

        Args:
            entity_id: The entity this evidence supports
            field_path: JSON path to the field (e.g., 'funding.total', 'features[0]')
            chunk_id: The chunk containing the evidence (preferred)
            source_id: The source containing the evidence (fallback)
            quote: Exact quote from the source
            locator: Additional location info (line number, selector)
            confidence: Extraction confidence (0.0-1.0)

        Returns:
            The created Evidence

        Raises:
            ValueError: If neither chunk_id nor source_id is provided
        """
        if chunk_id is None and source_id is None:
            raise ValueError("Evidence must reference either chunk_id or source_id")

        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO evidence
                    (entity_id, field_path, chunk_id, source_id, quote, locator, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING {self._EVIDENCE_COLS}
                """,
                (entity_id, field_path, chunk_id, source_id, quote, locator, confidence),
            )
            row = cur.fetchone()
            conn.commit()

        return Evidence.from_row(row)

    def get_evidence(self, evidence_id: uuid.UUID) -> Evidence | None:
        """Get evidence by ID."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {self._EVIDENCE_COLS} FROM evidence WHERE id = %s",
                (evidence_id,),
            )
            row = cur.fetchone()
        return Evidence.from_row(row) if row else None

    def list_evidence(
        self,
        entity_id: uuid.UUID | None = None,
        field_path: str | None = None,
        chunk_id: uuid.UUID | None = None,
        source_id: uuid.UUID | None = None,
    ) -> list[Evidence]:
        """List evidence with optional filters."""
        conn = self._get_conn()
        conditions = []
        params: list = []

        if entity_id:
            conditions.append("entity_id = %s")
            params.append(entity_id)
        if field_path:
            conditions.append("field_path = %s")
            params.append(field_path)
        if chunk_id:
            conditions.append("chunk_id = %s")
            params.append(chunk_id)
        if source_id:
            conditions.append("source_id = %s")
            params.append(source_id)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {self._EVIDENCE_COLS} FROM evidence
                WHERE {where_clause}
                ORDER BY field_path, created_at
                """,
                params,
            )
            rows = cur.fetchall()

        return [Evidence.from_row(row) for row in rows]

    def delete_evidence(self, evidence_id: uuid.UUID) -> bool:
        """Delete evidence by ID."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM evidence WHERE id = %s RETURNING id", (evidence_id,))
            deleted = cur.fetchone() is not None
            conn.commit()
        return deleted

    def delete_evidence_for_entity(self, entity_id: uuid.UUID) -> int:
        """Delete all evidence for an entity."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM evidence WHERE entity_id = %s", (entity_id,))
            count = cur.rowcount
            conn.commit()
        return count

    def update_evidence_field_path(
        self, evidence_id: uuid.UUID, new_field_path: str
    ) -> bool:
        """Update the field_path of an evidence record.

        Used during feature key migrations to update references.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE evidence SET field_path = %s WHERE id = %s RETURNING id",
                (new_field_path, evidence_id),
            )
            updated = cur.fetchone() is not None
            conn.commit()
        return updated

    def get_evidence_with_citations(
        self,
        entity_id: uuid.UUID,
    ) -> dict[str, list[Citation]]:
        """Get evidence for an entity, grouped by field_path with full citations.

        Returns a dictionary mapping field_path to a list of Citations,
        each containing the source context needed for rendering.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            # Join evidence with chunks and sources to get full citation context
            cur.execute(
                """
                SELECT
                    e.field_path,
                    e.quote,
                    s.canonical_url,
                    c.source_file,
                    c.heading_path,
                    c.content,
                    e.confidence
                FROM evidence e
                LEFT JOIN chunks c ON e.chunk_id = c.id
                LEFT JOIN sources s ON COALESCE(c.source_id, e.source_id) = s.id
                WHERE e.entity_id = %s
                ORDER BY e.field_path, e.created_at
                """,
                (entity_id,),
            )
            rows = cur.fetchall()

        # Group by field_path
        citations_by_field: dict[str, list[Citation]] = {}
        for row in rows:
            field_path = row[0]
            citation = Citation(
                field_path=field_path,
                quote=row[1],
                source_url=row[2] or "",
                source_file=row[3],
                heading_path=row[4],
                chunk_content=row[5],
                confidence=row[6],
            )
            if field_path not in citations_by_field:
                citations_by_field[field_path] = []
            citations_by_field[field_path].append(citation)

        return citations_by_field

    def get_entity_with_citations(
        self,
        entity_id: uuid.UUID,
    ) -> tuple[Entity | None, dict[str, list[Citation]]]:
        """Get an entity with all its citations grouped by field.

        Returns tuple of (entity, citations_by_field).
        If entity doesn't exist, returns (None, {}).
        """
        entity = self.get_entity(entity_id)
        if entity is None:
            return None, {}

        citations = self.get_evidence_with_citations(entity_id)
        return entity, citations

    # === Text-based search convenience method ===

    def search_text(
        self,
        query: str,
        collection: str | None = None,
        collections: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
        min_similarity: float = 0.0,
        model_name: str = "all-MiniLM-L6-v2",
        timings: dict[str, float] | None = None,
    ) -> SearchResponse:
        """Search using a text query (handles embedding internally).

        This is a convenience wrapper around search() that embeds the query text
        before performing the vector search.

        Args:
            query: Natural language search query
            collection: Filter to specific collection (exact match)
            collections: Filter to any of these collections (OR semantics)
            tags: Filter to sources with all these tags (AND semantics)
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0-1)
            model_name: Embedding model to use
            timings: Optional dict to store timing breakdown

        Returns:
            SearchResponse with ranked results
        """
        import time

        from toolhub.indexer.embedder import embed_text

        # Embed query
        t0 = time.perf_counter()
        embed_timings: dict[str, float] = {}
        query_embedding = embed_text(query, model_name=model_name, timings=embed_timings)
        if timings is not None:
            timings["embed_query"] = time.perf_counter() - t0
            for k, v in embed_timings.items():
                timings[f"embed_{k}"] = v

        # Perform search
        response = self.search(
            query_embedding=list(query_embedding),
            query_text=query,
            collection=collection,
            collections=collections,
            tags=tags,
            limit=limit,
            min_similarity=min_similarity,
        )

        # Merge timings
        if timings is not None:
            for k, v in response.timings.items():
                timings[k] = v

        return response

    # === Context manager support ===

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
