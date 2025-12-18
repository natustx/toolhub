"""Integration tests for Entity Schema Registry and Entity CRUD.

Tests cover:
- Entity type registration with schema versioning
- Entity CRUD with profile validation
- Schema caching for performance
- Validation error field paths
"""

from __future__ import annotations

import uuid

import pytest

from toolhub.config import Config, PostgresConfig, S3Config
from toolhub.store.knowledge import (
    EntityType,
    EntityValidationError,
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


# Sample JSON schemas for testing
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
        "team_size": {"type": "integer"},
    },
    "required": ["name"],
}

WISDOM_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "links": {"type": "array", "items": {"type": "string", "format": "uri"}},
    },
    "required": ["topic", "summary"],
}


class TestEntityTypeRegistration:
    """Test entity type registry operations."""

    def test_register_new_entity_type(self, store: KnowledgeStore):
        """Registering a new type creates it with version 1."""
        entity_type = store.register_entity_type(
            type_key="competitor",
            json_schema=COMPETITOR_SCHEMA,
            description="Competitor profile",
        )

        assert entity_type.type_key == "competitor"
        assert entity_type.schema_version == 1
        assert entity_type.description == "Competitor profile"
        assert entity_type.json_schema == COMPETITOR_SCHEMA

    def test_register_same_schema_keeps_version(self, store: KnowledgeStore):
        """Registering identical schema doesn't bump version."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        # Register again with same schema
        entity_type = store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        assert entity_type.schema_version == 1

    def test_register_changed_schema_bumps_version(self, store: KnowledgeStore):
        """Changing schema increments version."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        # Change schema (add required field)
        changed_schema = {
            **COMPETITOR_SCHEMA,
            "required": ["name", "description"],
        }
        entity_type = store.register_entity_type("competitor", changed_schema)

        assert entity_type.schema_version == 2
        assert entity_type.json_schema["required"] == ["name", "description"]

    def test_register_updates_description_without_version_bump(self, store: KnowledgeStore):
        """Updating description alone doesn't bump version."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA, description="v1")

        entity_type = store.register_entity_type(
            "competitor", COMPETITOR_SCHEMA, description="v2 - updated"
        )

        assert entity_type.schema_version == 1
        assert entity_type.description == "v2 - updated"

    def test_get_entity_type_returns_existing(self, store: KnowledgeStore):
        """get_entity_type retrieves registered type."""
        store.register_entity_type("wisdom", WISDOM_SCHEMA)

        entity_type = store.get_entity_type("wisdom")

        assert entity_type is not None
        assert entity_type.type_key == "wisdom"
        assert entity_type.json_schema == WISDOM_SCHEMA

    def test_get_entity_type_returns_none_for_missing(self, store: KnowledgeStore):
        """get_entity_type returns None for unregistered type."""
        result = store.get_entity_type("nonexistent")
        assert result is None

    def test_list_entity_types(self, store: KnowledgeStore):
        """list_entity_types returns all registered types."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)
        store.register_entity_type("wisdom", WISDOM_SCHEMA)

        types = store.list_entity_types()

        assert len(types) == 2
        type_keys = {t.type_key for t in types}
        assert type_keys == {"competitor", "wisdom"}


class TestEntityCRUD:
    """Test entity create/read/update/delete with validation."""

    @pytest.fixture
    def competitor_type(self, store: KnowledgeStore) -> EntityType:
        """Register competitor type for entity tests."""
        return store.register_entity_type("competitor", COMPETITOR_SCHEMA)

    def test_create_entity_with_valid_profile(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """Creating entity with valid profile succeeds."""
        entity = store.create_entity(
            type_key="competitor",
            name="Acme Corp",
            profile={
                "name": "Acme Corp",
                "description": "A competitor company",
                "features": ["feature1", "feature2"],
                "team_size": 50,
            },
            tags=["saas", "enterprise"],
            collection="market-research",
        )

        assert entity.id is not None
        assert entity.type_key == "competitor"
        assert entity.name == "Acme Corp"
        assert entity.profile["team_size"] == 50
        assert entity.tags == ["saas", "enterprise"]
        assert entity.collection == "market-research"

    def test_create_entity_rejects_invalid_profile(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """Creating entity with invalid profile raises EntityValidationError."""
        with pytest.raises(EntityValidationError) as exc_info:
            store.create_entity(
                type_key="competitor",
                name="Bad Corp",
                profile={
                    # Missing required 'name' field
                    "team_size": "not a number",  # Wrong type
                },
            )

        error = exc_info.value
        # Should indicate the validation failure
        assert "team_size" in str(error) or "name" in str(error)

    def test_create_entity_rejects_missing_required_field(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """Missing required field raises EntityValidationError with field path."""
        with pytest.raises(EntityValidationError) as exc_info:
            store.create_entity(
                type_key="competitor",
                name="Bad Corp",
                profile={
                    # Missing required 'name' field
                    "description": "Some description",
                },
            )

        error = exc_info.value
        assert "'name' is a required property" in str(error)

    def test_create_entity_rejects_unregistered_type(self, store: KnowledgeStore):
        """Creating entity with unregistered type raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            store.create_entity(type_key="nonexistent", name="Test")

        assert "not registered" in str(exc_info.value)

    def test_get_entity_by_id(self, store: KnowledgeStore, competitor_type: EntityType):
        """get_entity retrieves entity by ID."""
        created = store.create_entity(
            type_key="competitor",
            name="Test Corp",
            profile={"name": "Test Corp"},
        )

        fetched = store.get_entity(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Test Corp"

    def test_get_entity_returns_none_for_missing(self, store: KnowledgeStore):
        """get_entity returns None for non-existent ID."""
        result = store.get_entity(uuid.uuid4())
        assert result is None

    def test_get_entity_by_name(self, store: KnowledgeStore, competitor_type: EntityType):
        """get_entity_by_name finds entity by type + name + collection."""
        store.create_entity(
            type_key="competitor",
            name="Unique Corp",
            profile={"name": "Unique Corp"},
            collection="research",
        )

        entity = store.get_entity_by_name("competitor", "Unique Corp", "research")

        assert entity is not None
        assert entity.name == "Unique Corp"

    def test_update_entity_with_valid_profile(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """update_entity with valid profile succeeds."""
        entity = store.create_entity(
            type_key="competitor",
            name="Update Corp",
            profile={"name": "Update Corp", "team_size": 10},
        )

        updated = store.update_entity(
            entity.id,
            profile={"name": "Update Corp", "team_size": 100},
        )

        assert updated.profile["team_size"] == 100

    def test_update_entity_rejects_invalid_profile(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """update_entity with invalid profile raises EntityValidationError."""
        entity = store.create_entity(
            type_key="competitor",
            name="Validate Corp",
            profile={"name": "Validate Corp"},
        )

        with pytest.raises(EntityValidationError):
            store.update_entity(
                entity.id,
                profile={"name": "Validate Corp", "team_size": "not a number"},
            )

    def test_update_entity_raises_for_missing_id(self, store: KnowledgeStore):
        """update_entity raises KeyError for non-existent entity."""
        with pytest.raises(KeyError):
            store.update_entity(uuid.uuid4(), profile={})

    def test_update_entity_tags_only(self, store: KnowledgeStore, competitor_type: EntityType):
        """update_entity can update tags without changing profile."""
        entity = store.create_entity(
            type_key="competitor",
            name="Tag Corp",
            profile={"name": "Tag Corp"},
            tags=["old"],
        )

        updated = store.update_entity(entity.id, tags=["new", "tags"])

        assert updated.tags == ["new", "tags"]
        assert updated.profile == {"name": "Tag Corp"}

    def test_delete_entity(self, store: KnowledgeStore, competitor_type: EntityType):
        """delete_entity removes the entity."""
        entity = store.create_entity(
            type_key="competitor",
            name="Delete Corp",
            profile={"name": "Delete Corp"},
        )

        deleted = store.delete_entity(entity.id)

        assert deleted is True
        assert store.get_entity(entity.id) is None

    def test_delete_entity_returns_false_for_missing(self, store: KnowledgeStore):
        """delete_entity returns False for non-existent ID."""
        result = store.delete_entity(uuid.uuid4())
        assert result is False

    def test_list_entities_with_filters(self, store: KnowledgeStore, competitor_type: EntityType):
        """list_entities filters by type_key, collection, and tags."""
        # Register wisdom type too
        store.register_entity_type("wisdom", WISDOM_SCHEMA)

        store.create_entity(
            "competitor", "Corp A", {"name": "Corp A"}, tags=["saas"], collection="research"
        )
        store.create_entity(
            "competitor", "Corp B", {"name": "Corp B"}, tags=["enterprise"], collection="research"
        )
        store.create_entity(
            "competitor", "Corp C", {"name": "Corp C"}, tags=["saas"], collection="other"
        )
        store.create_entity(
            "wisdom", "Insight 1", {"topic": "t", "summary": "s"}, collection="research"
        )

        # Filter by type
        competitors = store.list_entities(type_key="competitor")
        assert len(competitors) == 3

        # Filter by collection
        research = store.list_entities(collection="research")
        assert len(research) == 3

        # Filter by tags
        saas = store.list_entities(tags=["saas"])
        assert len(saas) == 2

        # Combined filters
        competitor_saas_research = store.list_entities(
            type_key="competitor", collection="research", tags=["saas"]
        )
        assert len(competitor_saas_research) == 1
        assert competitor_saas_research[0].name == "Corp A"


class TestSchemaCaching:
    """Test in-process schema caching behavior."""

    def test_schema_cached_after_first_access(self, store: KnowledgeStore):
        """Schema is cached after first entity creation."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        # Cache should be empty
        assert "competitor" not in store._schema_cache

        # Create entity (triggers schema lookup and cache)
        store.create_entity("competitor", "Cache Test", {"name": "Cache Test"})

        # Now cached
        assert "competitor" in store._schema_cache
        assert store._schema_cache["competitor"][0] == 1  # version
        assert store._schema_cache["competitor"][1] == COMPETITOR_SCHEMA

    def test_cache_invalidated_on_schema_update(self, store: KnowledgeStore):
        """Cache is invalidated when schema is updated."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)
        store.create_entity("competitor", "Cache Test", {"name": "Cache Test"})

        # Cache populated
        assert "competitor" in store._schema_cache

        # Update schema
        new_schema = {**COMPETITOR_SCHEMA, "required": ["name", "description"]}
        store.register_entity_type("competitor", new_schema)

        # Cache invalidated
        assert "competitor" not in store._schema_cache


class TestEntityValidationErrorDetails:
    """Test that validation errors include helpful field path information."""

    @pytest.fixture
    def nested_schema_type(self, store: KnowledgeStore) -> EntityType:
        """Register type with nested schema for detailed error testing."""
        nested_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "config": {
                    "type": "object",
                    "properties": {
                        "settings": {
                            "type": "object",
                            "properties": {
                                "enabled": {"type": "boolean"},
                            },
                        },
                    },
                },
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["name"],
        }
        return store.register_entity_type("nested", nested_schema)

    def test_error_includes_nested_field_path(
        self, store: KnowledgeStore, nested_schema_type: EntityType
    ):
        """Validation error includes path to nested field."""
        with pytest.raises(EntityValidationError) as exc_info:
            store.create_entity(
                type_key="nested",
                name="Nested Test",
                profile={
                    "name": "Test",
                    "config": {
                        "settings": {
                            "enabled": "not a boolean",  # Should be boolean
                        }
                    },
                },
            )

        error = exc_info.value
        # Field path should indicate nested location
        assert error.field_path is not None
        path = error.field_path
        assert "config" in path or "settings" in path or "enabled" in path

    def test_error_includes_array_index_in_path(
        self, store: KnowledgeStore, nested_schema_type: EntityType
    ):
        """Validation error includes array index in field path."""
        with pytest.raises(EntityValidationError) as exc_info:
            store.create_entity(
                type_key="nested",
                name="Array Test",
                profile={
                    "name": "Test",
                    "items": [1, 2, "not an integer", 4],  # Index 2 is wrong
                },
            )

        error = exc_info.value
        assert error.field_path is not None
        # Should indicate items.2 or items[2]
        assert "items" in error.field_path
        assert "2" in error.field_path


class TestEntityToDict:
    """Test Entity serialization."""

    def test_entity_to_dict(self, store: KnowledgeStore):
        """Entity.to_dict produces JSON-serializable output."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)
        entity = store.create_entity(
            type_key="competitor",
            name="Dict Corp",
            profile={"name": "Dict Corp", "features": ["a", "b"]},
            tags=["test"],
        )

        data = entity.to_dict()

        assert data["id"] == str(entity.id)
        assert data["type_key"] == "competitor"
        assert data["name"] == "Dict Corp"
        assert data["profile"]["features"] == ["a", "b"]
        assert data["tags"] == ["test"]
        assert "created_at" in data
        assert "updated_at" in data
