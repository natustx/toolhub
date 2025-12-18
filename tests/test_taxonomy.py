"""Integration tests for Feature Taxonomy entity type and CLI.

Tests cover:
- Feature taxonomy entity type registration
- Taxonomy CRUD operations via KnowledgeStore
- Updated competitor schema with features as source counts
- Schema validation for taxonomy profiles
"""

from __future__ import annotations

import json

import pytest

from toolhub.config import Config, PostgresConfig, S3Config
from toolhub.store.knowledge import (
    EntityType,
    EntityValidationError,
    KnowledgeStore,
)
from toolhub.store.schemas import (
    COMPETITOR_SCHEMA,
    FEATURE_TAXONOMY_SCHEMA,
    register_builtin_schemas,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def test_config() -> Config:
    """Create test configuration."""
    config = Config()
    config.postgres = PostgresConfig(url="postgresql://toolhub:toolhub@localhost:5433/toolhub_test_main")
    config.s3 = S3Config(
        endpoint_url="http://localhost:9010",
        bucket="toolhub-test",
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


class TestFeatureTaxonomySchema:
    """Test the feature_taxonomy JSON schema definition."""

    def test_schema_has_required_fields(self):
        """Schema requires domain and groups fields."""
        assert "required" in FEATURE_TAXONOMY_SCHEMA
        assert "domain" in FEATURE_TAXONOMY_SCHEMA["required"]
        assert "groups" in FEATURE_TAXONOMY_SCHEMA["required"]

    def test_schema_groups_structure(self):
        """Groups are arrays of objects with key, label, features."""
        groups_schema = FEATURE_TAXONOMY_SCHEMA["properties"]["groups"]
        assert groups_schema["type"] == "array"

        group_item = groups_schema["items"]
        assert "key" in group_item["properties"]
        assert "label" in group_item["properties"]
        assert "features" in group_item["properties"]

    def test_schema_feature_structure(self):
        """Features have key and label with pattern validation."""
        groups_schema = FEATURE_TAXONOMY_SCHEMA["properties"]["groups"]
        feature_schema = groups_schema["items"]["properties"]["features"]["items"]

        assert feature_schema["properties"]["key"]["pattern"] == "^[a-z0-9-]+$"
        assert "label" in feature_schema["properties"]


class TestFeatureTaxonomyRegistration:
    """Test feature_taxonomy entity type registration."""

    def test_register_feature_taxonomy_type(self, store: KnowledgeStore):
        """Can register feature_taxonomy entity type."""
        entity_type = store.register_entity_type(
            type_key="feature_taxonomy",
            json_schema=FEATURE_TAXONOMY_SCHEMA,
            description="Feature taxonomy for grouping features",
        )

        assert entity_type.type_key == "feature_taxonomy"
        assert entity_type.schema_version == 1
        assert entity_type.json_schema == FEATURE_TAXONOMY_SCHEMA

    def test_register_builtin_schemas_helper(self, store: KnowledgeStore):
        """register_builtin_schemas registers all built-in types."""
        registered = register_builtin_schemas(store)

        assert "feature_taxonomy" in registered
        assert "competitor" in registered

        # Verify they exist
        ft = store.get_entity_type("feature_taxonomy")
        assert ft is not None

        comp = store.get_entity_type("competitor")
        assert comp is not None


class TestFeatureTaxonomyCRUD:
    """Test feature taxonomy entity CRUD operations."""

    @pytest.fixture
    def taxonomy_type(self, store: KnowledgeStore) -> EntityType:
        """Register feature_taxonomy type for tests."""
        return store.register_entity_type(
            "feature_taxonomy",
            FEATURE_TAXONOMY_SCHEMA,
            description="Feature taxonomy",
        )

    def test_create_empty_taxonomy(self, store: KnowledgeStore, taxonomy_type: EntityType):
        """Can create a taxonomy with empty groups."""
        taxonomy = store.create_entity(
            type_key="feature_taxonomy",
            name="k12-fundraising",
            profile={
                "domain": "K-12 Fundraising",
                "groups": [],
            },
            collection="market-research",
        )

        assert taxonomy.id is not None
        assert taxonomy.name == "k12-fundraising"
        assert taxonomy.profile["domain"] == "K-12 Fundraising"
        assert taxonomy.profile["groups"] == []

    def test_create_taxonomy_with_groups(self, store: KnowledgeStore, taxonomy_type: EntityType):
        """Can create a taxonomy with populated groups and features."""
        taxonomy = store.create_entity(
            type_key="feature_taxonomy",
            name="k12-fundraising",
            profile={
                "domain": "K-12 Fundraising",
                "groups": [
                    {
                        "key": "donor-management",
                        "label": "Donor Management",
                        "features": [
                            {"key": "donor-profiles", "label": "Donor Profiles"},
                            {"key": "giving-history", "label": "Giving History"},
                        ],
                    },
                    {
                        "key": "online-giving",
                        "label": "Online Giving",
                        "features": [
                            {"key": "donation-forms", "label": "Donation Forms"},
                        ],
                    },
                ],
            },
        )

        assert len(taxonomy.profile["groups"]) == 2
        assert taxonomy.profile["groups"][0]["key"] == "donor-management"
        assert len(taxonomy.profile["groups"][0]["features"]) == 2

    def test_taxonomy_rejects_invalid_feature_key(
        self, store: KnowledgeStore, taxonomy_type: EntityType
    ):
        """Feature keys must match pattern ^[a-z0-9-]+$."""
        with pytest.raises(EntityValidationError):
            store.create_entity(
                type_key="feature_taxonomy",
                name="bad-taxonomy",
                profile={
                    "domain": "Test",
                    "groups": [
                        {
                            "key": "test-group",
                            "label": "Test Group",
                            "features": [
                                {"key": "Invalid_Key!", "label": "Bad"},  # Invalid key
                            ],
                        },
                    ],
                },
            )

    def test_taxonomy_rejects_missing_domain(
        self, store: KnowledgeStore, taxonomy_type: EntityType
    ):
        """Domain is required."""
        with pytest.raises(EntityValidationError):
            store.create_entity(
                type_key="feature_taxonomy",
                name="bad-taxonomy",
                profile={
                    # Missing 'domain'
                    "groups": [],
                },
            )

    def test_update_taxonomy_profile(self, store: KnowledgeStore, taxonomy_type: EntityType):
        """Can update taxonomy to add groups."""
        taxonomy = store.create_entity(
            type_key="feature_taxonomy",
            name="update-test",
            profile={"domain": "Test", "groups": []},
        )

        updated = store.update_entity(
            taxonomy.id,
            profile={
                "domain": "Test",
                "groups": [
                    {
                        "key": "new-group",
                        "label": "New Group",
                        "features": [],
                    }
                ],
            },
        )

        assert len(updated.profile["groups"]) == 1
        assert updated.profile["groups"][0]["key"] == "new-group"

    def test_get_taxonomy_by_name(self, store: KnowledgeStore, taxonomy_type: EntityType):
        """Can retrieve taxonomy by name and collection."""
        store.create_entity(
            type_key="feature_taxonomy",
            name="named-taxonomy",
            profile={"domain": "Test", "groups": []},
            collection="test-collection",
        )

        found = store.get_entity_by_name(
            "feature_taxonomy", "named-taxonomy", "test-collection"
        )

        assert found is not None
        assert found.name == "named-taxonomy"

    def test_list_taxonomies(self, store: KnowledgeStore, taxonomy_type: EntityType):
        """Can list all taxonomies."""
        store.create_entity(
            "feature_taxonomy", "tax-1", {"domain": "One", "groups": []}, collection="a"
        )
        store.create_entity(
            "feature_taxonomy", "tax-2", {"domain": "Two", "groups": []}, collection="a"
        )
        store.create_entity(
            "feature_taxonomy", "tax-3", {"domain": "Three", "groups": []}, collection="b"
        )

        # List all
        all_taxonomies = store.list_entities(type_key="feature_taxonomy")
        assert len(all_taxonomies) == 3

        # Filter by collection
        collection_a = store.list_entities(type_key="feature_taxonomy", collection="a")
        assert len(collection_a) == 2


class TestCompetitorSchemaUpdate:
    """Test the updated competitor schema with features as source counts."""

    @pytest.fixture
    def competitor_type(self, store: KnowledgeStore) -> EntityType:
        """Register updated competitor type."""
        return store.register_entity_type(
            "competitor",
            COMPETITOR_SCHEMA,
            description="Competitor with feature source counts",
        )

    def test_competitor_features_as_source_counts(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """Competitor features are stored as feature-key -> {sources: N}."""
        competitor = store.create_entity(
            type_key="competitor",
            name="Bloomerang",
            profile={
                "description": "Donor management platform",
                "website": "https://bloomerang.com",
                "features": {
                    "donor-profiles": {"sources": 3},
                    "giving-history": {"sources": 2},
                    "recurring-gifts": {"sources": 1},
                },
            },
        )

        assert competitor.profile["features"]["donor-profiles"]["sources"] == 3
        assert competitor.profile["features"]["giving-history"]["sources"] == 2

    def test_competitor_features_requires_sources_count(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """Each feature entry requires sources count."""
        with pytest.raises(EntityValidationError):
            store.create_entity(
                type_key="competitor",
                name="Bad Competitor",
                profile={
                    "features": {
                        "some-feature": {},  # Missing 'sources'
                    },
                },
            )

    def test_competitor_funding_format(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """Competitor funding accepts total and rounds."""
        competitor = store.create_entity(
            type_key="competitor",
            name="Funded Corp",
            profile={
                "funding": {
                    "total": 50000000,
                    "rounds": ["Series A", "Series B"],
                },
            },
        )

        assert competitor.profile["funding"]["total"] == 50000000
        assert "Series A" in competitor.profile["funding"]["rounds"]

    def test_competitor_with_all_fields(
        self, store: KnowledgeStore, competitor_type: EntityType
    ):
        """Competitor with all fields validates correctly."""
        competitor = store.create_entity(
            type_key="competitor",
            name="Complete Competitor",
            profile={
                "description": "Full featured competitor",
                "website": "https://example.com",
                "features": {
                    "feature-a": {"sources": 2},
                    "feature-b": {"sources": 1},
                },
                "funding": {
                    "total": 10000000,
                    "rounds": ["Seed"],
                },
                "team_size": 50,
            },
            tags=["saas", "enterprise"],
            collection="market-research",
        )

        assert competitor.profile["team_size"] == 50
        assert competitor.tags == ["saas", "enterprise"]


class TestEntityTypeToDict:
    """Test EntityType.to_dict serialization."""

    def test_entity_type_to_dict(self, store: KnowledgeStore):
        """EntityType.to_dict produces JSON-serializable output."""
        entity_type = store.register_entity_type(
            "test_type",
            {"type": "object"},
            description="Test type",
        )

        data = entity_type.to_dict()

        assert data["type_key"] == "test_type"
        assert data["schema_version"] == 1
        assert data["description"] == "Test type"
        assert data["json_schema"] == {"type": "object"}
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

        # Should be JSON serializable
        json_str = json.dumps(data)
        assert "test_type" in json_str


class TestTaxonomyManagement:
    """Test taxonomy manipulation operations (add/rename/move)."""

    @pytest.fixture
    def setup_taxonomy(self, store: KnowledgeStore):
        """Set up taxonomy and competitor types for management tests."""
        store.register_entity_type("feature_taxonomy", FEATURE_TAXONOMY_SCHEMA)
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        # Create taxonomy with groups and features
        taxonomy = store.create_entity(
            type_key="feature_taxonomy",
            name="test-taxonomy",
            profile={
                "domain": "Test Domain",
                "groups": [
                    {
                        "key": "group-a",
                        "label": "Group A",
                        "features": [
                            {"key": "feature-1", "label": "Feature 1"},
                            {"key": "feature-2", "label": "Feature 2"},
                        ],
                    },
                    {
                        "key": "group-b",
                        "label": "Group B",
                        "features": [
                            {"key": "feature-3", "label": "Feature 3"},
                        ],
                    },
                ],
            },
        )
        return taxonomy

    def test_add_group_to_taxonomy(self, store: KnowledgeStore, setup_taxonomy):
        """Can add a new group to an existing taxonomy."""
        taxonomy = setup_taxonomy

        # Add new group via profile update
        groups = taxonomy.profile["groups"]
        groups.append({
            "key": "group-c",
            "label": "Group C",
            "features": [],
        })
        updated = store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        assert len(updated.profile["groups"]) == 3
        assert updated.profile["groups"][2]["key"] == "group-c"

    def test_add_feature_to_group(self, store: KnowledgeStore, setup_taxonomy):
        """Can add a feature to an existing group."""
        taxonomy = setup_taxonomy

        groups = taxonomy.profile["groups"]
        groups[0]["features"].append({
            "key": "feature-new",
            "label": "New Feature",
        })
        updated = store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        assert len(updated.profile["groups"][0]["features"]) == 3
        assert updated.profile["groups"][0]["features"][2]["key"] == "feature-new"

    def test_rename_group(self, store: KnowledgeStore, setup_taxonomy):
        """Can rename a group key and label."""
        taxonomy = setup_taxonomy

        groups = taxonomy.profile["groups"]
        groups[0]["key"] = "group-alpha"
        groups[0]["label"] = "Group Alpha"
        updated = store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        assert updated.profile["groups"][0]["key"] == "group-alpha"
        assert updated.profile["groups"][0]["label"] == "Group Alpha"

    def test_move_feature_between_groups(self, store: KnowledgeStore, setup_taxonomy):
        """Can move a feature from one group to another."""
        taxonomy = setup_taxonomy

        groups = taxonomy.profile["groups"]
        # Move feature-2 from group-a to group-b
        feature = groups[0]["features"].pop(1)  # Remove from group-a
        groups[1]["features"].append(feature)  # Add to group-b

        updated = store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        assert len(updated.profile["groups"][0]["features"]) == 1
        assert len(updated.profile["groups"][1]["features"]) == 2
        assert updated.profile["groups"][1]["features"][1]["key"] == "feature-2"


class TestFeatureKeyMigration:
    """Test feature rename with migration of competitors and evidence."""

    @pytest.fixture
    def setup_migration_data(self, store: KnowledgeStore):
        """Set up taxonomy, competitors, and evidence for migration tests."""
        store.register_entity_type("feature_taxonomy", FEATURE_TAXONOMY_SCHEMA)
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        # Create a source for evidence linking
        source = store.add_source(
            canonical_url="https://test.example.com/docs",
            source_type="website",
            collection="test",
        )

        # Create taxonomy
        taxonomy = store.create_entity(
            type_key="feature_taxonomy",
            name="migration-test",
            profile={
                "domain": "Migration Test",
                "groups": [
                    {
                        "key": "features",
                        "label": "Features",
                        "features": [
                            {"key": "old-feature", "label": "Old Feature"},
                        ],
                    },
                ],
            },
        )

        # Create competitor with the feature
        competitor = store.create_entity(
            type_key="competitor",
            name="Test Competitor",
            profile={
                "features": {
                    "old-feature": {"sources": 2},
                    "other-feature": {"sources": 1},
                },
            },
        )

        # Create evidence for the feature (requires source_id)
        evidence = store.add_evidence(
            entity_id=competitor.id,
            field_path="features.old-feature",
            source_id=source.id,
            quote="This is evidence for the old feature",
            confidence=0.9,
        )

        return {"taxonomy": taxonomy, "competitor": competitor, "evidence": evidence}

    def test_competitor_feature_key_migration(
        self, store: KnowledgeStore, setup_migration_data
    ):
        """Renaming feature key in competitor profile works correctly."""
        competitor = setup_migration_data["competitor"]

        # Simulate migration: rename old-feature to new-feature
        features = competitor.profile["features"]
        features["new-feature"] = features.pop("old-feature")

        updated = store.update_entity(
            competitor.id,
            profile={**competitor.profile, "features": features},
        )

        assert "new-feature" in updated.profile["features"]
        assert "old-feature" not in updated.profile["features"]
        assert updated.profile["features"]["new-feature"]["sources"] == 2
        assert updated.profile["features"]["other-feature"]["sources"] == 1

    def test_evidence_field_path_migration(
        self, store: KnowledgeStore, setup_migration_data
    ):
        """Evidence field_path can be updated during migration."""
        evidence = setup_migration_data["evidence"]

        # Update field_path
        result = store.update_evidence_field_path(evidence.id, "features.new-feature")
        assert result is True

        # Verify the update
        updated = store.get_evidence(evidence.id)
        assert updated.field_path == "features.new-feature"

    def test_list_evidence_by_field_path(
        self, store: KnowledgeStore, setup_migration_data
    ):
        """Can list evidence filtered by field_path for migration."""
        evidence_list = store.list_evidence(field_path="features.old-feature")

        assert len(evidence_list) == 1
        assert evidence_list[0].field_path == "features.old-feature"

    def test_full_feature_migration_workflow(
        self, store: KnowledgeStore, setup_migration_data
    ):
        """Complete workflow: rename feature in taxonomy, competitors, evidence."""
        taxonomy = setup_migration_data["taxonomy"]
        competitor = setup_migration_data["competitor"]
        evidence = setup_migration_data["evidence"]

        # 1. Update taxonomy
        groups = taxonomy.profile["groups"]
        groups[0]["features"][0]["key"] = "new-feature"
        groups[0]["features"][0]["label"] = "New Feature"
        store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        # 2. Update competitor
        features = competitor.profile["features"]
        features["new-feature"] = features.pop("old-feature")
        store.update_entity(
            competitor.id,
            profile={**competitor.profile, "features": features},
        )

        # 3. Update evidence
        store.update_evidence_field_path(evidence.id, "features.new-feature")

        # Verify all changes
        updated_taxonomy = store.get_entity(taxonomy.id)
        assert updated_taxonomy.profile["groups"][0]["features"][0]["key"] == "new-feature"

        updated_competitor = store.get_entity(competitor.id)
        assert "new-feature" in updated_competitor.profile["features"]
        assert "old-feature" not in updated_competitor.profile["features"]

        updated_evidence = store.get_evidence(evidence.id)
        assert updated_evidence.field_path == "features.new-feature"
