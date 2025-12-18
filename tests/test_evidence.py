"""Integration tests for Evidence model and retrieval.

Tests cover:
- Evidence CRUD operations
- Evidence linkage (field_path → chunk_id with quote)
- Retrieval grouping by entity
- Citation rendering (chunk excerpt + heading_path + canonical_url)
"""

from __future__ import annotations

import uuid

import pytest

from toolhub.config import Config, PostgresConfig, S3Config
from toolhub.store.knowledge import (
    Citation,
    KnowledgeStore,
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
    """Create KnowledgeStore using test fixtures."""
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


# Sample schema for testing
COMPETITOR_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}},
        "funding": {
            "type": "object",
            "properties": {
                "total": {"type": "number"},
                "rounds": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "required": ["name"],
}


class TestEvidenceCRUD:
    """Test evidence create/read/update/delete operations."""

    @pytest.fixture
    def setup_entity_with_source(self, store: KnowledgeStore):
        """Set up an entity and source with chunks for evidence tests."""
        # Register entity type and create entity
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)
        entity = store.create_entity(
            type_key="competitor",
            name="Acme Corp",
            profile={
                "name": "Acme Corp",
                "description": "A competitor",
                "features": ["feature1", "feature2"],
                "funding": {"total": 10000000, "rounds": ["Series A"]},
            },
        )

        # Create source with chunks
        source = store.add_source(
            canonical_url="https://techcrunch.com/acme",
            source_type="website",
        )
        chunk_id = store.add_chunk(
            source=source,
            content="Acme Corp raised $10M in Series A funding.",
            chunk_index=0,
            embedding=[0.1] * 384,
            model_id="test-model",
            heading="Funding",
            heading_path="News > Funding",
            source_file="article.html",
        )

        return {"entity": entity, "source": source, "chunk_id": chunk_id}

    def test_add_evidence_with_chunk(self, store: KnowledgeStore, setup_entity_with_source):
        """Adding evidence with chunk_id succeeds."""
        data = setup_entity_with_source
        evidence = store.add_evidence(
            entity_id=data["entity"].id,
            field_path="funding.total",
            chunk_id=data["chunk_id"],
            quote="raised $10M in Series A",
            confidence=0.95,
        )

        assert evidence.id is not None
        assert evidence.entity_id == data["entity"].id
        assert evidence.field_path == "funding.total"
        assert evidence.chunk_id == data["chunk_id"]
        assert evidence.quote == "raised $10M in Series A"
        assert evidence.confidence == 0.95

    def test_add_evidence_with_source_fallback(
        self, store: KnowledgeStore, setup_entity_with_source
    ):
        """Adding evidence with source_id (no chunk) succeeds."""
        data = setup_entity_with_source
        evidence = store.add_evidence(
            entity_id=data["entity"].id,
            field_path="description",
            source_id=data["source"].id,
            quote="A leading competitor",
            locator="paragraph 2",
        )

        assert evidence.source_id == data["source"].id
        assert evidence.chunk_id is None
        assert evidence.locator == "paragraph 2"

    def test_add_evidence_requires_reference(self, store: KnowledgeStore, setup_entity_with_source):
        """Adding evidence without chunk_id or source_id raises ValueError."""
        data = setup_entity_with_source
        with pytest.raises(ValueError) as exc_info:
            store.add_evidence(
                entity_id=data["entity"].id,
                field_path="name",
                # Neither chunk_id nor source_id provided
            )

        assert "must reference" in str(exc_info.value)

    def test_get_evidence_by_id(self, store: KnowledgeStore, setup_entity_with_source):
        """get_evidence retrieves evidence by ID."""
        data = setup_entity_with_source
        created = store.add_evidence(
            entity_id=data["entity"].id,
            field_path="funding.total",
            chunk_id=data["chunk_id"],
        )

        fetched = store.get_evidence(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.field_path == "funding.total"

    def test_get_evidence_returns_none_for_missing(self, store: KnowledgeStore):
        """get_evidence returns None for non-existent ID."""
        result = store.get_evidence(uuid.uuid4())
        assert result is None

    def test_list_evidence_by_entity(self, store: KnowledgeStore, setup_entity_with_source):
        """list_evidence filters by entity_id."""
        data = setup_entity_with_source
        store.add_evidence(
            entity_id=data["entity"].id,
            field_path="funding.total",
            chunk_id=data["chunk_id"],
        )
        store.add_evidence(
            entity_id=data["entity"].id,
            field_path="features",
            source_id=data["source"].id,
        )

        evidence_list = store.list_evidence(entity_id=data["entity"].id)

        assert len(evidence_list) == 2

    def test_list_evidence_by_field_path(self, store: KnowledgeStore, setup_entity_with_source):
        """list_evidence filters by field_path."""
        data = setup_entity_with_source
        store.add_evidence(
            entity_id=data["entity"].id,
            field_path="funding.total",
            chunk_id=data["chunk_id"],
        )
        store.add_evidence(
            entity_id=data["entity"].id,
            field_path="features",
            source_id=data["source"].id,
        )

        evidence_list = store.list_evidence(field_path="funding.total")

        assert len(evidence_list) == 1
        assert evidence_list[0].field_path == "funding.total"

    def test_delete_evidence(self, store: KnowledgeStore, setup_entity_with_source):
        """delete_evidence removes the evidence."""
        data = setup_entity_with_source
        evidence = store.add_evidence(
            entity_id=data["entity"].id,
            field_path="funding.total",
            chunk_id=data["chunk_id"],
        )

        deleted = store.delete_evidence(evidence.id)

        assert deleted is True
        assert store.get_evidence(evidence.id) is None

    def test_delete_evidence_for_entity(self, store: KnowledgeStore, setup_entity_with_source):
        """delete_evidence_for_entity removes all evidence for entity."""
        data = setup_entity_with_source
        store.add_evidence(
            entity_id=data["entity"].id,
            field_path="funding.total",
            chunk_id=data["chunk_id"],
        )
        store.add_evidence(
            entity_id=data["entity"].id,
            field_path="features",
            source_id=data["source"].id,
        )

        count = store.delete_evidence_for_entity(data["entity"].id)

        assert count == 2
        assert store.list_evidence(entity_id=data["entity"].id) == []


class TestEvidenceRetrieval:
    """Test evidence retrieval grouped by entity/field."""

    @pytest.fixture
    def setup_entity_with_multiple_evidence(self, store: KnowledgeStore):
        """Set up an entity with multiple pieces of evidence."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)
        entity = store.create_entity(
            type_key="competitor",
            name="MultiEvidence Corp",
            profile={
                "name": "MultiEvidence Corp",
                "features": ["ai", "ml"],
                "funding": {"total": 50000000},
            },
        )

        # Create two sources with chunks
        source1 = store.add_source("https://techcrunch.com/multi", "website")
        source2 = store.add_source("https://bloomberg.com/multi", "website")

        chunk1 = store.add_chunk(
            source=source1,
            content="MultiEvidence raised $50M in funding.",
            chunk_index=0,
            embedding=[0.1] * 384,
            model_id="test",
            heading="Funding News",
            heading_path="Startups > Funding",
            source_file="funding.html",
        )

        chunk2 = store.add_chunk(
            source=source2,
            content="They offer AI and ML features.",
            chunk_index=0,
            embedding=[0.2] * 384,
            model_id="test",
            heading="Product Features",
            heading_path="Tech > Products",
            source_file="products.html",
        )

        # Add evidence for multiple fields
        store.add_evidence(
            entity_id=entity.id,
            field_path="funding.total",
            chunk_id=chunk1,
            quote="raised $50M",
            confidence=0.9,
        )
        store.add_evidence(
            entity_id=entity.id,
            field_path="features",
            chunk_id=chunk2,
            quote="AI and ML features",
            confidence=0.85,
        )
        # Second evidence for same field (features)
        store.add_evidence(
            entity_id=entity.id,
            field_path="features",
            source_id=source1.id,
            quote="advanced ML capabilities",
            confidence=0.7,
        )

        return {
            "entity": entity,
            "source1": source1,
            "source2": source2,
            "chunk1": chunk1,
            "chunk2": chunk2,
        }

    def test_get_evidence_with_citations_groups_by_field(
        self, store: KnowledgeStore, setup_entity_with_multiple_evidence
    ):
        """Evidence is grouped by field_path."""
        data = setup_entity_with_multiple_evidence
        citations_by_field = store.get_evidence_with_citations(data["entity"].id)

        assert "funding.total" in citations_by_field
        assert "features" in citations_by_field
        assert len(citations_by_field["funding.total"]) == 1
        assert len(citations_by_field["features"]) == 2

    def test_citations_include_source_context(
        self, store: KnowledgeStore, setup_entity_with_multiple_evidence
    ):
        """Citations include full source context for rendering."""
        data = setup_entity_with_multiple_evidence
        citations_by_field = store.get_evidence_with_citations(data["entity"].id)

        funding_citation = citations_by_field["funding.total"][0]
        assert funding_citation.quote == "raised $50M"
        assert funding_citation.source_url == "https://techcrunch.com/multi"
        assert funding_citation.source_file == "funding.html"
        assert funding_citation.heading_path == "Startups > Funding"
        assert "MultiEvidence raised $50M" in funding_citation.chunk_content
        assert funding_citation.confidence == 0.9

    def test_get_entity_with_citations(
        self, store: KnowledgeStore, setup_entity_with_multiple_evidence
    ):
        """get_entity_with_citations returns entity + citations tuple."""
        data = setup_entity_with_multiple_evidence
        entity, citations = store.get_entity_with_citations(data["entity"].id)

        assert entity is not None
        assert entity.name == "MultiEvidence Corp"
        assert "funding.total" in citations
        assert "features" in citations

    def test_get_entity_with_citations_missing_entity(self, store: KnowledgeStore):
        """get_entity_with_citations returns None, {} for missing entity."""
        entity, citations = store.get_entity_with_citations(uuid.uuid4())

        assert entity is None
        assert citations == {}


class TestCitationRendering:
    """Test citation rendering to markdown and dict."""

    def test_citation_to_markdown_with_quote(self):
        """Citation renders with quote as blockquote."""
        citation = Citation(
            field_path="funding.total",
            quote="raised $50M in Series B",
            source_url="https://techcrunch.com/article",
            source_file="article.html",
            heading_path="Startups > Funding",
            chunk_content="The company raised $50M in Series B funding.",
            confidence=0.95,
        )

        md = citation.to_markdown()

        assert '> "raised $50M in Series B"' in md
        assert "article.html > Startups > Funding" in md

    def test_citation_to_markdown_without_quote(self):
        """Citation renders without quote if not provided."""
        citation = Citation(
            field_path="name",
            quote=None,
            source_url="https://example.com",
            source_file=None,
            heading_path=None,
            chunk_content="Some content",
            confidence=0.8,
        )

        md = citation.to_markdown()

        assert ">" not in md  # No blockquote
        assert "https://example.com" in md

    def test_citation_to_dict(self):
        """Citation.to_dict returns all fields."""
        citation = Citation(
            field_path="features[0]",
            quote="AI-powered",
            source_url="https://docs.example.com",
            source_file="features.md",
            heading_path="Overview > Features",
            chunk_content="AI-powered automation",
            confidence=0.92,
        )

        data = citation.to_dict()

        assert data["field_path"] == "features[0]"
        assert data["quote"] == "AI-powered"
        assert data["source_url"] == "https://docs.example.com"
        assert data["source_file"] == "features.md"
        assert data["heading_path"] == "Overview > Features"
        assert data["chunk_content"] == "AI-powered automation"
        assert data["confidence"] == 0.92


class TestEvidenceToDict:
    """Test Evidence serialization."""

    @pytest.fixture
    def setup_evidence(self, store: KnowledgeStore):
        """Create evidence for serialization tests."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)
        entity = store.create_entity("competitor", "Serialize Corp", {"name": "Serialize Corp"})
        source = store.add_source("https://example.com", "website")
        return store.add_evidence(
            entity_id=entity.id,
            field_path="name",
            source_id=source.id,
            quote="Serialize Corp",
            confidence=1.0,
        )

    def test_evidence_to_dict(self, store: KnowledgeStore, setup_evidence):
        """Evidence.to_dict produces JSON-serializable output."""
        evidence = setup_evidence
        data = evidence.to_dict()

        assert data["id"] == str(evidence.id)
        assert data["entity_id"] == str(evidence.entity_id)
        assert data["field_path"] == "name"
        assert data["source_id"] == str(evidence.source_id)
        assert data["chunk_id"] is None
        assert data["quote"] == "Serialize Corp"
        assert data["confidence"] == 1.0
        assert "created_at" in data


class TestEvidenceCascadeDelete:
    """Test that evidence is deleted when entity is deleted."""

    def test_evidence_deleted_with_entity(self, store: KnowledgeStore):
        """Evidence is cascade deleted when entity is deleted."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)
        entity = store.create_entity("competitor", "Cascade Corp", {"name": "Cascade Corp"})
        source = store.add_source("https://cascade.com", "website")

        store.add_evidence(
            entity_id=entity.id,
            field_path="name",
            source_id=source.id,
        )

        # Verify evidence exists
        evidence_list = store.list_evidence(entity_id=entity.id)
        assert len(evidence_list) == 1

        # Delete entity
        store.delete_entity(entity.id)

        # Evidence should be gone (cascade delete)
        evidence_list = store.list_evidence(entity_id=entity.id)
        assert len(evidence_list) == 0
