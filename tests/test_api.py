"""Integration tests for API endpoints.

Tests cover:
- Entity CRUD via API
- Evidence CRUD via API
- Report generation via API
- Knowledge query via API
- Error handling for invalid inputs
"""

from __future__ import annotations

import uuid

import pytest
from starlette.testclient import TestClient

from toolhub.config import Config, PostgresConfig, S3Config
from toolhub.daemon import app

# Test constants (match docker-compose.test.yml)
TEST_POSTGRES_URL = "postgresql://toolhub:toolhub@localhost:5433/toolhub_test_main"
TEST_S3_ENDPOINT = "http://localhost:9010"
TEST_S3_BUCKET = "toolhub-test"

pytestmark = pytest.mark.integration


# Sample schemas for testing
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


@pytest.fixture
def test_config(test_database_url: str) -> Config:
    """Create test configuration pointing to test containers."""
    config = Config()
    config.postgres = PostgresConfig(url=test_database_url)
    config.s3 = S3Config(
        endpoint_url=TEST_S3_ENDPOINT,
        bucket=TEST_S3_BUCKET,
        access_key="minioadmin",
        secret_key="minioadmin",
    )
    return config


@pytest.fixture
def client(test_config: Config, test_database_url: str, s3_cleanup, monkeypatch) -> TestClient:
    """Create test client with mocked config."""

    # Monkeypatch load_config to return our test config
    def mock_load_config():
        test_config.postgres.url = test_database_url
        return test_config

    monkeypatch.setattr("toolhub.daemon.load_config", mock_load_config)
    monkeypatch.setattr("toolhub.store.knowledge.Config", lambda: test_config)

    with TestClient(app) as client:
        yield client

    # Cleanup: delete all data created during test
    from toolhub.store.knowledge import KnowledgeStore

    test_config.postgres.url = test_database_url
    ks = KnowledgeStore(test_config, env=s3_cleanup)
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


class TestEntityTypeAPI:
    """Test entity type API endpoints."""

    def test_register_entity_type(self, client: TestClient):
        """POST /api/v1/entity-types creates new entity type."""
        response = client.post(
            "/api/v1/entity-types",
            json={
                "type_key": "competitor",
                "json_schema": COMPETITOR_SCHEMA,
                "description": "Competitor entity",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type_key"] == "competitor"
        assert data["schema_version"] == 1
        assert data["description"] == "Competitor entity"

    def test_list_entity_types(self, client: TestClient):
        """GET /api/v1/entity-types returns all registered types."""
        # Register a type first
        client.post(
            "/api/v1/entity-types",
            json={"type_key": "competitor", "json_schema": COMPETITOR_SCHEMA},
        )

        response = client.get("/api/v1/entity-types")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(t["type_key"] == "competitor" for t in data)

    def test_get_entity_type(self, client: TestClient):
        """GET /api/v1/entity-types/{type_key} returns specific type."""
        client.post(
            "/api/v1/entity-types",
            json={"type_key": "competitor", "json_schema": COMPETITOR_SCHEMA},
        )

        response = client.get("/api/v1/entity-types/competitor")

        assert response.status_code == 200
        data = response.json()
        assert data["type_key"] == "competitor"

    def test_get_entity_type_not_found(self, client: TestClient):
        """GET /api/v1/entity-types/{type_key} returns 404 for missing type."""
        response = client.get("/api/v1/entity-types/nonexistent")

        assert response.status_code == 404


class TestEntityAPI:
    """Test entity CRUD API endpoints."""

    @pytest.fixture
    def setup_entity_type(self, client: TestClient):
        """Register entity type for entity tests."""
        client.post(
            "/api/v1/entity-types",
            json={"type_key": "competitor", "json_schema": COMPETITOR_SCHEMA},
        )

    def test_create_entity(self, client: TestClient, setup_entity_type):
        """POST /api/v1/entities creates a new entity."""
        response = client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Test Corp",
                "profile": {"name": "Test Corp", "features": ["api"]},
                "tags": ["saas"],
                "collection": "research",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Corp"
        assert data["type_key"] == "competitor"
        assert data["collection"] == "research"
        assert "id" in data

    def test_create_entity_validation_error(self, client: TestClient, setup_entity_type):
        """POST /api/v1/entities returns 422 for invalid profile."""
        response = client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Bad Corp",
                "profile": {"features": "not an array"},  # Should be array
            },
        )

        assert response.status_code == 422

    def test_create_entity_unknown_type(self, client: TestClient):
        """POST /api/v1/entities returns 400 for unknown type."""
        response = client.post(
            "/api/v1/entities",
            json={"type_key": "unknown", "name": "Test"},
        )

        assert response.status_code == 400

    def test_list_entities(self, client: TestClient, setup_entity_type):
        """GET /api/v1/entities returns entities with filters."""
        # Create entities
        client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Corp A",
                "profile": {"name": "Corp A"},
                "collection": "research",
            },
        )
        client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Corp B",
                "profile": {"name": "Corp B"},
                "collection": "other",
            },
        )

        # List all
        response = client.get("/api/v1/entities")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2

        # Filter by collection
        response = client.get("/api/v1/entities?collection=research")
        data = response.json()
        assert data["count"] == 1
        assert data["entities"][0]["name"] == "Corp A"

    def test_get_entity(self, client: TestClient, setup_entity_type):
        """GET /api/v1/entities/{id} returns specific entity."""
        create_response = client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Get Corp",
                "profile": {"name": "Get Corp"},
            },
        )
        entity_id = create_response.json()["id"]

        response = client.get(f"/api/v1/entities/{entity_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Get Corp"

    def test_get_entity_not_found(self, client: TestClient):
        """GET /api/v1/entities/{id} returns 404 for missing entity."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/entities/{fake_id}")

        assert response.status_code == 404

    def test_get_entity_invalid_id(self, client: TestClient):
        """GET /api/v1/entities/{id} returns 400 for invalid UUID."""
        response = client.get("/api/v1/entities/not-a-uuid")

        assert response.status_code == 400

    def test_update_entity(self, client: TestClient, setup_entity_type):
        """PUT /api/v1/entities/{id} updates entity profile and tags."""
        create_response = client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Update Corp",
                "profile": {"name": "Update Corp"},
            },
        )
        entity_id = create_response.json()["id"]

        response = client.put(
            f"/api/v1/entities/{entity_id}",
            json={"tags": ["new-tag"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "new-tag" in data["tags"]

    def test_delete_entity(self, client: TestClient, setup_entity_type):
        """DELETE /api/v1/entities/{id} removes entity."""
        create_response = client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Delete Corp",
                "profile": {"name": "Delete Corp"},
            },
        )
        entity_id = create_response.json()["id"]

        response = client.delete(f"/api/v1/entities/{entity_id}")

        assert response.status_code == 200
        assert "Deleted" in response.json()["message"]

        # Verify deleted
        get_response = client.get(f"/api/v1/entities/{entity_id}")
        assert get_response.status_code == 404


class TestEvidenceAPI:
    """Test evidence CRUD API endpoints."""

    @pytest.fixture
    def setup_entity_and_source(
        self, client: TestClient, test_config: Config, test_database_url: str, s3_cleanup
    ):
        """Set up entity and source for evidence tests."""
        from toolhub.store.knowledge import KnowledgeStore

        # Register entity type and create entity
        client.post(
            "/api/v1/entity-types",
            json={"type_key": "competitor", "json_schema": COMPETITOR_SCHEMA},
        )
        entity_response = client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Evidence Corp",
                "profile": {"name": "Evidence Corp", "funding": {"total": 1000000}},
            },
        )
        entity_id = entity_response.json()["id"]

        # Create source with chunk directly via store
        test_config.postgres.url = test_database_url
        store = KnowledgeStore(test_config, env=s3_cleanup)
        source = store.add_source("https://example.com/api/v1/evidence", "website")
        chunk_id = store.add_chunk(
            source=source,
            content="Evidence Corp raised $1M.",
            chunk_index=0,
            embedding=[0.1] * 384,
            model_id="test",
        )
        store.close()

        return {"entity_id": entity_id, "source_id": str(source.id), "chunk_id": str(chunk_id)}

    def test_add_evidence(self, client: TestClient, setup_entity_and_source):
        """POST /api/v1/evidence adds evidence."""
        data = setup_entity_and_source

        response = client.post(
            "/api/v1/evidence",
            json={
                "entity_id": data["entity_id"],
                "field_path": "funding.total",
                "chunk_id": data["chunk_id"],
                "quote": "raised $1M",
                "confidence": 0.9,
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["field_path"] == "funding.total"
        assert result["quote"] == "raised $1M"
        assert result["confidence"] == 0.9

    def test_add_evidence_missing_reference(self, client: TestClient, setup_entity_and_source):
        """POST /api/v1/evidence returns 400 when no chunk or source provided."""
        data = setup_entity_and_source

        response = client.post(
            "/api/v1/evidence",
            json={
                "entity_id": data["entity_id"],
                "field_path": "name",
                # No chunk_id or source_id
            },
        )

        assert response.status_code == 400

    def test_list_evidence(self, client: TestClient, setup_entity_and_source):
        """GET /api/v1/evidence returns evidence with filters."""
        data = setup_entity_and_source

        # Add evidence
        client.post(
            "/api/v1/evidence",
            json={
                "entity_id": data["entity_id"],
                "field_path": "funding.total",
                "chunk_id": data["chunk_id"],
            },
        )

        # List by entity
        response = client.get(f"/api/v1/evidence?entity_id={data['entity_id']}")

        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 1
        assert result["evidence"][0]["field_path"] == "funding.total"

    def test_get_evidence(self, client: TestClient, setup_entity_and_source):
        """GET /api/v1/evidence/{id} returns specific evidence."""
        data = setup_entity_and_source

        create_response = client.post(
            "/api/v1/evidence",
            json={
                "entity_id": data["entity_id"],
                "field_path": "name",
                "source_id": data["source_id"],
            },
        )
        evidence_id = create_response.json()["id"]

        response = client.get(f"/api/v1/evidence/{evidence_id}")

        assert response.status_code == 200
        assert response.json()["field_path"] == "name"

    def test_delete_evidence(self, client: TestClient, setup_entity_and_source):
        """DELETE /api/v1/evidence/{id} removes evidence."""
        data = setup_entity_and_source

        create_response = client.post(
            "/api/v1/evidence",
            json={
                "entity_id": data["entity_id"],
                "field_path": "name",
                "source_id": data["source_id"],
            },
        )
        evidence_id = create_response.json()["id"]

        response = client.delete(f"/api/v1/evidence/{evidence_id}")

        assert response.status_code == 200

        # Verify deleted
        get_response = client.get(f"/api/v1/evidence/{evidence_id}")
        assert get_response.status_code == 404


class TestReportAPI:
    """Test report API endpoints."""

    def test_list_report_types(self, client: TestClient):
        """GET /api/v1/reports returns available report types."""
        response = client.get("/api/v1/reports")

        assert response.status_code == 200
        data = response.json()
        assert "competitor-feature-matrix" in data
        assert "fundraising-timeline" in data
        assert "wisdom-digest" in data

    @pytest.fixture
    def setup_report_data(self, client: TestClient):
        """Set up data for report tests."""
        # Register entity type
        client.post(
            "/api/v1/entity-types",
            json={"type_key": "competitor", "json_schema": COMPETITOR_SCHEMA},
        )

        # Create competitors
        client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Report Corp A",
                "profile": {
                    "name": "Report Corp A",
                    "features": ["api", "auth"],
                    "funding": {"total": 5000000},
                },
                "collection": "report-test",
            },
        )
        client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Report Corp B",
                "profile": {
                    "name": "Report Corp B",
                    "features": ["api", "dashboard"],
                },
                "collection": "report-test",
            },
        )

    def test_generate_feature_matrix_report(self, client: TestClient, setup_report_data):
        """POST /api/v1/reports/{type} generates feature matrix."""
        response = client.post(
            "/api/v1/reports/competitor-feature-matrix",
            json={"collection": "report-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Competitor Feature Matrix"
        assert data["entities_count"] == 2
        # New structure uses groups with features nested inside
        assert "groups" in data["data"]
        assert "competitors" in data["data"]
        assert "markdown" in data

    def test_generate_fundraising_timeline(self, client: TestClient, setup_report_data):
        """POST /api/v1/reports/{type} generates fundraising timeline."""
        response = client.post(
            "/api/v1/reports/fundraising-timeline",
            json={"collection": "report-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Fundraising Timeline"
        assert "funding" in data["data"]

    def test_generate_report_unknown_type(self, client: TestClient):
        """POST /api/v1/reports/{type} returns 400 for unknown type."""
        response = client.post(
            "/api/v1/reports/unknown-report-type",
            json={},
        )

        assert response.status_code == 400


class TestEntityWithCitationsAPI:
    """Test entity with citations endpoint."""

    @pytest.fixture
    def setup_entity_with_evidence(
        self, client: TestClient, test_config: Config, test_database_url: str, s3_cleanup
    ):
        """Set up entity with evidence for citations tests."""
        from toolhub.store.knowledge import KnowledgeStore

        # Register entity type and create entity
        client.post(
            "/api/v1/entity-types",
            json={"type_key": "competitor", "json_schema": COMPETITOR_SCHEMA},
        )
        entity_response = client.post(
            "/api/v1/entities",
            json={
                "type_key": "competitor",
                "name": "Citation Corp",
                "profile": {"name": "Citation Corp", "features": ["api"]},
            },
        )
        entity_id = entity_response.json()["id"]

        # Add source and evidence via store
        test_config.postgres.url = test_database_url
        store = KnowledgeStore(test_config, env=s3_cleanup)

        source = store.add_source("https://example.com/citation", "website")
        chunk_id = store.add_chunk(
            source=source,
            content="Citation Corp offers API services.",
            chunk_index=0,
            embedding=[0.1] * 384,
            model_id="test",
            heading="Features",
            heading_path="Products > Features",
            source_file="features.html",
        )

        store.add_evidence(
            entity_id=uuid.UUID(entity_id),
            field_path="features",
            chunk_id=chunk_id,
            quote="offers API services",
            confidence=0.95,
        )
        store.close()

        return {"entity_id": entity_id}

    def test_get_entity_with_citations(self, client: TestClient, setup_entity_with_evidence):
        """GET /api/v1/entities/{id}/citations returns entity with citations."""
        entity_id = setup_entity_with_evidence["entity_id"]

        response = client.get(f"/api/v1/entities/{entity_id}/citations")

        assert response.status_code == 200
        data = response.json()
        assert data["entity"]["name"] == "Citation Corp"
        assert "features" in data["citations"]
        assert len(data["citations"]["features"]) == 1
        citation = data["citations"]["features"][0]
        assert citation["quote"] == "offers API services"
        assert citation["confidence"] == 0.95
