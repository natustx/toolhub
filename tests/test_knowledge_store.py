"""Integration tests for KnowledgeStore.

Tests the full ingest and query pipeline:
source → S3 artifacts → Postgres chunks → embeddings → hybrid search
"""

from __future__ import annotations

import uuid

import pytest

from toolhub.config import Config, PostgresConfig, S3Config
from toolhub.store.knowledge import (
    ArtifactKind,
    KnowledgeStore,
    SearchResponse,
    Source,
    SourceStatus,
)

# Test constants (match docker-compose.test.yml)
TEST_POSTGRES_URL = "postgresql://toolhub:toolhub@localhost:5433/toolhub_test_main"
TEST_S3_ENDPOINT = "http://localhost:9010"
TEST_S3_BUCKET = "toolhub-test"

pytestmark = pytest.mark.integration


@pytest.fixture
def test_config() -> Config:
    """Create test configuration pointing to test containers."""
    config = Config()
    config.postgres = PostgresConfig(url=TEST_POSTGRES_URL)
    config.s3 = S3Config(
        endpoint_url=TEST_S3_ENDPOINT,
        bucket=TEST_S3_BUCKET,
        access_key="minioadmin",
        secret_key="minioadmin",
    )
    return config


@pytest.fixture
def store(test_config: Config, test_database_url: str, s3_cleanup) -> KnowledgeStore:
    """Create KnowledgeStore using test fixtures.

    Uses test_database_url for connection and s3_cleanup prefix for S3 isolation.
    Each test gets its own KnowledgeStore instance with fresh connection.
    Cleans up all data after test completes.
    """
    # Override config with test database URL
    test_config.postgres.url = test_database_url
    ks = KnowledgeStore(test_config, env=s3_cleanup)
    yield ks

    # Cleanup: delete all data created during test
    conn = ks._get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM evidence")
        cur.execute("DELETE FROM entities")
        cur.execute("DELETE FROM entity_types")
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM source_artifacts")
        cur.execute("DELETE FROM sources")
        conn.commit()
    ks.close()


class TestSourceOperations:
    """Test source CRUD operations."""

    def test_add_source_creates_record(self, store: KnowledgeStore):
        """Adding a source creates a record with assigned UUID."""
        source = store.add_source(
            canonical_url="https://github.com/test/repo",
            source_type="github",
            collection="docs",
            tags=["python", "api"],
        )

        assert source.id is not None
        assert source.canonical_url == "https://github.com/test/repo"
        assert source.source_type == "github"
        assert source.collection == "docs"
        assert source.tags == ["python", "api"]
        assert source.status == SourceStatus.PENDING

    def test_add_source_upserts_on_conflict(self, store: KnowledgeStore):
        """Adding same URL+collection updates existing record."""
        source1 = store.add_source(
            canonical_url="https://example.com",
            source_type="website",
            collection="test",
            tags=["v1"],
        )

        source2 = store.add_source(
            canonical_url="https://example.com",
            source_type="website",
            collection="test",
            tags=["v2"],
        )

        # Same ID (upserted, not duplicated)
        assert source1.id == source2.id
        # Tags updated
        assert source2.tags == ["v2"]

    def test_get_source_returns_existing(self, store: KnowledgeStore):
        """get_source returns the source by ID."""
        created = store.add_source(
            canonical_url="https://test.com",
            source_type="llmstxt",
        )

        fetched = store.get_source(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.canonical_url == "https://test.com"

    def test_get_source_returns_none_for_missing(self, store: KnowledgeStore):
        """get_source returns None for non-existent ID."""
        result = store.get_source(uuid.uuid4())
        assert result is None

    def test_update_source_status(self, store: KnowledgeStore):
        """update_source_status changes status and metadata."""
        source = store.add_source(
            canonical_url="https://test.com",
            source_type="website",
        )
        assert source.status == SourceStatus.PENDING

        store.update_source_status(source.id, SourceStatus.INDEXED, sha="abc123")

        updated = store.get_source(source.id)
        assert updated.status == SourceStatus.INDEXED
        assert updated.sha == "abc123"

    def test_list_sources_with_filters(self, store: KnowledgeStore):
        """list_sources filters by collection, tags, and status."""
        store.add_source("https://a.com", "website", collection="docs", tags=["python"])
        store.add_source("https://b.com", "website", collection="docs", tags=["go"])
        store.add_source("https://c.com", "website", collection="tools", tags=["python"])

        # Filter by collection
        docs = store.list_sources(collection="docs")
        assert len(docs) == 2

        # Filter by tags
        python = store.list_sources(tags=["python"])
        assert len(python) == 2

        # Combined filters
        docs_python = store.list_sources(collection="docs", tags=["python"])
        assert len(docs_python) == 1
        assert docs_python[0].canonical_url == "https://a.com"

    def test_delete_source_removes_record(self, store: KnowledgeStore):
        """delete_source removes the source."""
        source = store.add_source("https://to-delete.com", "website")

        deleted = store.delete_source(source.id)

        assert deleted is True
        assert store.get_source(source.id) is None


class TestArtifactOperations:
    """Test S3 artifact storage."""

    def test_store_and_get_artifact(self, store: KnowledgeStore):
        """Storing an artifact uploads to S3 and records metadata."""
        source = store.add_source("https://test.com", "website")

        # Store raw HTML artifact
        artifact = store.store_artifact(
            source,
            ArtifactKind.RAW,
            content=b"<html><body>Test</body></html>",
            content_type="text/html",
            extension=".html",
        )

        assert artifact.source_id == source.id
        assert artifact.kind == ArtifactKind.RAW
        assert artifact.size_bytes == 30
        assert artifact.sha256 is not None

        # Retrieve it
        content = store.get_artifact(source, ArtifactKind.RAW)
        assert content == b"<html><body>Test</body></html>"

    def test_get_artifact_returns_none_for_missing(self, store: KnowledgeStore):
        """get_artifact returns None when artifact doesn't exist."""
        source = store.add_source("https://test.com", "website")

        content = store.get_artifact(source, ArtifactKind.EXTRACTED)

        assert content is None

    def test_store_artifact_updates_on_conflict(self, store: KnowledgeStore):
        """Storing same artifact kind updates existing record."""
        source = store.add_source("https://test.com", "website")

        store.store_artifact(source, ArtifactKind.EXTRACTED, b"v1", "text/markdown")
        artifact2 = store.store_artifact(source, ArtifactKind.EXTRACTED, b"v2", "text/markdown")

        content = store.get_artifact(source, ArtifactKind.EXTRACTED)
        assert content == b"v2"
        assert artifact2.size_bytes == 2


class TestChunkOperations:
    """Test chunk storage with embeddings."""

    def test_add_chunk_stores_embedding(self, store: KnowledgeStore):
        """add_chunk stores content and embedding vector."""
        source = store.add_source("https://test.com", "website")

        # Create a fake embedding (384 dimensions for all-MiniLM-L6-v2)
        embedding = [0.1] * 384

        chunk_id = store.add_chunk(
            source=source,
            content="Test content about authentication",
            chunk_index=0,
            embedding=embedding,
            model_id="all-MiniLM-L6-v2",
            heading="Authentication",
            heading_path="Security > Authentication",
            source_file="docs/security.md",
            is_code=False,
        )

        assert chunk_id is not None
        assert store.get_chunk_count(source.id) == 1

    def test_add_chunks_batch(self, store: KnowledgeStore):
        """add_chunks_batch stores multiple chunks efficiently."""
        source = store.add_source("https://test.com", "website")

        chunks = [
            {"content": f"Chunk {i}", "embedding": [0.1 * i] * 384, "heading": f"Heading {i}"}
            for i in range(5)
        ]

        chunk_ids = store.add_chunks_batch(source, chunks, "all-MiniLM-L6-v2")

        assert len(chunk_ids) == 5
        assert store.get_chunk_count(source.id) == 5

    def test_delete_chunks_removes_all_for_source(self, store: KnowledgeStore):
        """delete_chunks removes all chunks for a source."""
        source = store.add_source("https://test.com", "website")
        chunks = [{"content": f"Chunk {i}", "embedding": [0.1] * 384} for i in range(3)]
        store.add_chunks_batch(source, chunks, "test-model")

        deleted = store.delete_chunks(source.id)

        assert deleted == 3
        assert store.get_chunk_count(source.id) == 0


class TestSearchOperations:
    """Test hybrid search functionality."""

    @pytest.fixture
    def indexed_source(self, store: KnowledgeStore) -> Source:
        """Create a source with indexed chunks for search tests."""
        source = store.add_source(
            "https://docs.example.com",
            "website",
            collection="docs",
            tags=["api"],
        )

        # Create chunks with varied content and embeddings
        chunks = [
            {
                "content": "Authentication using OAuth2 allows secure access to APIs.",
                "heading": "OAuth2",
                "heading_path": "Security > OAuth2",
                "source_file": "security.md",
                "embedding": [0.9, 0.1] + [0.0] * 382,  # High on first dimension
            },
            {
                "content": "Rate limiting prevents abuse of API endpoints.",
                "heading": "Rate Limiting",
                "heading_path": "Security > Rate Limiting",
                "source_file": "security.md",
                "embedding": [0.1, 0.9] + [0.0] * 382,  # High on second dimension
            },
            {
                "content": "Database connections should use connection pooling.",
                "heading": "Pooling",
                "heading_path": "Database > Pooling",
                "source_file": "database.md",
                "embedding": [0.5, 0.5] + [0.0] * 382,  # Balanced
            },
        ]
        store.add_chunks_batch(source, chunks, "test-model")

        return source

    def test_vector_search_returns_similar_chunks(
        self, store: KnowledgeStore, indexed_source: Source
    ):
        """Vector search returns chunks ordered by similarity."""
        # Query embedding similar to first chunk (OAuth2)
        query_embedding = [0.85, 0.15] + [0.0] * 382

        response = store.search(query_embedding, limit=3)

        assert isinstance(response, SearchResponse)
        assert len(response.results) == 3
        # First result should be most similar to query (OAuth2 chunk)
        first_content = response.results[0].content
        assert "OAuth2" in first_content or "Authentication" in first_content

    def test_search_filters_by_collection(self, store: KnowledgeStore):
        """Search respects collection filter."""
        # Create sources in different collections
        source1 = store.add_source("https://a.com", "website", collection="docs")
        source2 = store.add_source("https://b.com", "website", collection="tools")

        store.add_chunk(source1, "Docs content", 0, [0.5] * 384, "test-model")
        store.add_chunk(source2, "Tools content", 0, [0.5] * 384, "test-model")

        # Search only docs collection
        response = store.search([0.5] * 384, collection="docs")

        assert len(response.results) == 1
        assert response.results[0].collection == "docs"

    def test_search_response_includes_citations(
        self, store: KnowledgeStore, indexed_source: Source
    ):
        """Search results include source information for citations."""
        query_embedding = [0.5, 0.5] + [0.0] * 382

        response = store.search(query_embedding, limit=1)

        result = response.results[0]
        assert result.canonical_url == "https://docs.example.com"
        assert result.source_file is not None
        assert result.heading_path is not None

    def test_search_to_markdown(self, store: KnowledgeStore, indexed_source: Source):
        """SearchResponse renders to markdown."""
        response = store.search([0.5] * 384, query_text="test query", limit=1)

        markdown = response.to_markdown()

        assert "# Search: test query" in markdown
        assert "Similarity:" in markdown

    def test_search_to_dict(self, store: KnowledgeStore, indexed_source: Source):
        """SearchResponse renders to dict for JSON output."""
        response = store.search([0.5] * 384, query_text="test", limit=1)

        data = response.to_dict()

        assert data["query"] == "test"
        assert "results" in data
        assert "timings" in data


class TestFullPipeline:
    """End-to-end tests for the complete ingest → search pipeline."""

    def test_ingest_and_search_pipeline(self, store: KnowledgeStore):
        """Full pipeline: add source → store artifacts → add chunks → search."""
        # 1. Add source
        source = store.add_source(
            canonical_url="https://fastapi.tiangolo.com",
            source_type="website",
            collection="frameworks",
            tags=["python", "web"],
        )

        # 2. Store raw artifact
        raw_html = b"<html><body><h1>FastAPI</h1><p>Modern API framework.</p></body></html>"
        store.store_artifact(source, ArtifactKind.RAW, raw_html, "text/html", ".html")

        # 3. Store extracted markdown
        extracted_md = b"# FastAPI\n\nModern API framework for Python."
        store.store_artifact(source, ArtifactKind.EXTRACTED, extracted_md, "text/markdown", ".md")

        # 4. Update status to indicate crawling done
        store.update_source_status(source.id, SourceStatus.CRAWLING)

        # 5. Add chunks with embeddings
        chunks = [
            {
                "content": "FastAPI is a modern, fast web framework for building APIs with Python.",
                "heading": "Introduction",
                "heading_path": "FastAPI > Introduction",
                "source_file": "index.md",
                "embedding": [0.8, 0.2] + [0.0] * 382,
            },
            {
                "content": "Automatic API documentation with Swagger UI and ReDoc.",
                "heading": "Features",
                "heading_path": "FastAPI > Features",
                "source_file": "features.md",
                "embedding": [0.3, 0.7] + [0.0] * 382,
            },
        ]
        store.add_chunks_batch(source, chunks, "all-MiniLM-L6-v2")

        # 6. Mark as indexed
        store.update_source_status(source.id, SourceStatus.INDEXED)

        # 7. Search
        query_embedding = [0.75, 0.25] + [0.0] * 382  # Similar to first chunk
        response = store.search(
            query_embedding,
            query_text="fast api framework python",
            collection="frameworks",
            limit=2,
        )

        # Verify results
        assert len(response.results) == 2
        assert response.collection == "frameworks"
        assert response.total_chunks_searched == 2

        # First result should be the introduction chunk
        top_result = response.results[0]
        assert "FastAPI" in top_result.content
        assert top_result.canonical_url == "https://fastapi.tiangolo.com"

        # 8. Verify artifacts can be retrieved for rebuild
        raw_content = store.get_artifact(source, ArtifactKind.RAW)
        assert raw_content == raw_html


class TestSourceToDict:
    """Test Source.to_dict() serialization."""

    def test_source_to_dict_includes_all_fields(self, store: KnowledgeStore):
        """to_dict includes all source fields for JSON serialization."""
        source = store.add_source(
            canonical_url="https://test.example.com/docs",
            source_type="help_docs",
            collection="research",
            tags=["nonprofit", "crm"],
        )

        result = source.to_dict()

        assert result["id"] == str(source.id)
        assert result["canonical_url"] == "https://test.example.com/docs"
        assert result["source_type"] == "help_docs"
        assert result["collection"] == "research"
        assert result["tags"] == ["nonprofit", "crm"]
        assert result["status"] == "pending"
        assert result["sha"] is None
        assert result["fetched_at"] is None
        assert "created_at" in result
        assert "updated_at" in result

    def test_source_to_dict_status_as_string(self, store: KnowledgeStore):
        """to_dict converts SourceStatus enum to string value."""
        source = store.add_source("https://test.com", "website")
        store.update_source_status(source.id, SourceStatus.INDEXED)

        # Refresh source from database
        refreshed = store.get_source(source.id)
        result = refreshed.to_dict()

        assert result["status"] == "indexed"

    def test_source_to_dict_dates_as_iso(self, store: KnowledgeStore):
        """to_dict converts datetime fields to ISO format strings."""
        source = store.add_source("https://test.com", "website")
        result = source.to_dict()

        # ISO format should be parseable
        from datetime import datetime

        created = datetime.fromisoformat(result["created_at"])
        assert created is not None
