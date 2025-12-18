"""Integration tests for Report generation framework.

Tests cover:
- ReportResult serialization (to_json, to_markdown, to_dict)
- CompetitorFeatureMatrixReport generation and output
- FundraisingTimelineReport generation and output
- WisdomDigestReport generation and output
- Deterministic output (sorted, stable)
- Citation inclusion when evidence exists
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from toolhub.config import Config, PostgresConfig, S3Config
from toolhub.store.knowledge import KnowledgeStore
from toolhub.store.reports import (
    REPORT_TYPES,
    CompetitorFeatureMatrixReport,
    FundraisingTimelineReport,
    ReportResult,
    WisdomDigestReport,
    get_report,
)

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

WISDOM_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "links": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topic", "summary"],
}


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


class TestReportResult:
    """Test ReportResult serialization."""

    def test_to_dict(self):
        """ReportResult.to_dict returns all fields."""
        result = ReportResult(
            title="Test Report",
            report_type="test-type",
            generated_at=datetime(2024, 1, 15, 10, 30, 0),
            data={"key": "value"},
            entities_count=5,
            has_citations=True,
            metadata={"version": "1.0"},
        )

        data = result.to_dict()

        assert data["title"] == "Test Report"
        assert data["report_type"] == "test-type"
        assert data["generated_at"] == "2024-01-15T10:30:00"
        assert data["data"] == {"key": "value"}
        assert data["entities_count"] == 5
        assert data["has_citations"] is True
        assert data["metadata"] == {"version": "1.0"}

    def test_to_json(self):
        """ReportResult.to_json produces valid JSON string."""
        result = ReportResult(
            title="JSON Test",
            report_type="test",
            generated_at=datetime(2024, 1, 15),
            data={"items": [1, 2, 3]},
            entities_count=3,
            has_citations=False,
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["title"] == "JSON Test"
        assert parsed["data"]["items"] == [1, 2, 3]

    def test_to_markdown_default(self):
        """ReportResult.to_markdown produces formatted output."""
        result = ReportResult(
            title="Markdown Test",
            report_type="test",
            generated_at=datetime(2024, 1, 15, 12, 0, 0),
            data={"sample": "data"},
            entities_count=2,
            has_citations=True,
        )

        md = result.to_markdown()

        assert "# Markdown Test" in md
        assert "*Generated: 2024-01-15T12:00:00*" in md
        assert "*Entities: 2*" in md
        assert "```json" in md
        assert '"sample": "data"' in md


class TestReportRegistry:
    """Test report type registry and factory."""

    def test_registry_contains_all_reports(self):
        """REPORT_TYPES contains all v1 report types."""
        assert "competitor-feature-matrix" in REPORT_TYPES
        assert "fundraising-timeline" in REPORT_TYPES
        assert "wisdom-digest" in REPORT_TYPES

    def test_get_report_returns_correct_type(self, store: KnowledgeStore):
        """get_report returns correct report instance."""
        report = get_report("competitor-feature-matrix", store)
        assert isinstance(report, CompetitorFeatureMatrixReport)

        report = get_report("fundraising-timeline", store)
        assert isinstance(report, FundraisingTimelineReport)

        report = get_report("wisdom-digest", store)
        assert isinstance(report, WisdomDigestReport)

    def test_get_report_raises_for_unknown_type(self, store: KnowledgeStore):
        """get_report raises KeyError for unknown type."""
        with pytest.raises(KeyError) as exc_info:
            get_report("nonexistent-report", store)

        assert "Unknown report type" in str(exc_info.value)
        assert "nonexistent-report" in str(exc_info.value)


class TestCompetitorFeatureMatrixReport:
    """Test competitor feature matrix report generation."""

    @pytest.fixture
    def setup_competitors(self, store: KnowledgeStore):
        """Create competitors with varying features for matrix tests."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        # Competitor A: has features 1, 2, 3
        entity_a = store.create_entity(
            type_key="competitor",
            name="Alpha Corp",
            profile={
                "name": "Alpha Corp",
                "features": ["auth", "api", "dashboard"],
            },
            collection="research",
        )

        # Competitor B: has features 2, 3, 4
        entity_b = store.create_entity(
            type_key="competitor",
            name="Beta Inc",
            profile={
                "name": "Beta Inc",
                "features": ["api", "dashboard", "mobile"],
            },
            collection="research",
        )

        # Competitor C: has feature 1 only
        entity_c = store.create_entity(
            type_key="competitor",
            name="Gamma LLC",
            profile={
                "name": "Gamma LLC",
                "features": ["auth"],
            },
            collection="research",
        )

        return {"a": entity_a, "b": entity_b, "c": entity_c}

    def test_feature_union_across_competitors(self, store: KnowledgeStore, setup_competitors):
        """Matrix includes union of all features from all competitors."""
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research")

        # Extract feature keys from the groups structure
        groups = result.data["groups"]
        all_features = []
        for group in groups:
            for feat in group.get("features", []):
                all_features.append(feat["key"])

        # Should contain union: auth, api, dashboard, mobile
        assert set(all_features) == {"auth", "api", "dashboard", "mobile"}

    def test_matrix_rows_show_feature_support(self, store: KnowledgeStore, setup_competitors):
        """Matrix rows correctly show which competitors have each feature."""
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research")

        # Get features from the single group (flat matrix)
        features = result.data["groups"][0]["features"]

        # Find auth feature
        auth_feat = next(f for f in features if f["key"] == "auth")
        assert auth_feat["competitors"]["Alpha Corp"]["present"] is True
        assert auth_feat["competitors"]["Beta Inc"]["present"] is False
        assert auth_feat["competitors"]["Gamma LLC"]["present"] is True

        # Find mobile feature
        mobile_feat = next(f for f in features if f["key"] == "mobile")
        assert mobile_feat["competitors"]["Alpha Corp"]["present"] is False
        assert mobile_feat["competitors"]["Beta Inc"]["present"] is True
        assert mobile_feat["competitors"]["Gamma LLC"]["present"] is False

    def test_deterministic_output_sorted(self, store: KnowledgeStore, setup_competitors):
        """Output is deterministic - features and entities sorted alphabetically."""
        report = CompetitorFeatureMatrixReport(store)

        # Generate twice
        result1 = report.generate(collection="research")
        result2 = report.generate(collection="research")

        # Extract feature keys
        features1 = [f["key"] for f in result1.data["groups"][0]["features"]]
        assert features1 == sorted(features1)

        # Competitor names should be sorted
        competitors = result1.data["competitors"]
        assert competitors == sorted(competitors)

        # Results should be identical
        assert result1.data == result2.data

    def test_filter_by_entity_ids(self, store: KnowledgeStore, setup_competitors):
        """Report can filter to specific entity IDs."""
        data = setup_competitors
        report = CompetitorFeatureMatrixReport(store)

        result = report.generate(entity_ids=[str(data["a"].id), str(data["b"].id)])

        assert result.entities_count == 2
        assert "Alpha Corp" in result.data["competitors"]
        assert "Beta Inc" in result.data["competitors"]
        assert "Gamma LLC" not in result.data["competitors"]

    def test_empty_result_when_no_competitors(self, store: KnowledgeStore):
        """Report handles empty result gracefully."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)
        report = CompetitorFeatureMatrixReport(store)

        result = report.generate(collection="nonexistent")

        assert result.entities_count == 0
        assert result.data["competitors"] == []
        assert result.data["groups"] == [{"key": "_all", "label": "All Features", "features": []}]

    def test_to_markdown_renders_table(self, store: KnowledgeStore, setup_competitors):
        """to_markdown renders a proper markdown table."""
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research")

        md = report.to_markdown(result)

        # Should have header
        assert "# Competitor Feature Matrix" in md
        assert "| Feature |" in md
        assert "Alpha Corp" in md
        assert "Beta Inc" in md
        assert "Gamma LLC" in md

        # Should have checkmarks and dashes
        assert "✓" in md
        assert "—" in md

    def test_citations_included_when_evidence_exists(
        self, store: KnowledgeStore, setup_competitors
    ):
        """Citations are included when evidence exists for features."""
        data = setup_competitors

        # Add source and evidence for Alpha's features
        source = store.add_source("https://alpha.com/features", "website")
        chunk_id = store.add_chunk(
            source=source,
            content="Alpha Corp offers authentication and API features.",
            chunk_index=0,
            embedding=[0.1] * 384,
            model_id="test",
            heading="Features",
            heading_path="Products > Features",
            source_file="features.html",
        )
        # Use the new field_path format: features.<feature-key>
        store.add_evidence(
            entity_id=data["a"].id,
            field_path="features.auth",
            chunk_id=chunk_id,
            quote="authentication and API features",
            confidence=0.95,
        )

        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research")

        assert result.has_citations is True
        assert "Alpha Corp" in result.data["citations"]
        assert len(result.data["citations"]["Alpha Corp"]) == 1
        # Verify feature_key is extracted
        assert result.data["citations"]["Alpha Corp"][0]["feature_key"] == "auth"


class TestCompetitorFeatureMatrixWithTaxonomy:
    """Test feature matrix report with taxonomy grouping and new schema."""

    @pytest.fixture
    def setup_taxonomy_and_competitors(self, store: KnowledgeStore):
        """Create taxonomy and competitors using new features schema."""
        from toolhub.store.schemas import COMPETITOR_SCHEMA, FEATURE_TAXONOMY_SCHEMA

        store.register_entity_type("feature_taxonomy", FEATURE_TAXONOMY_SCHEMA)
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        # Create taxonomy with groups
        taxonomy = store.create_entity(
            type_key="feature_taxonomy",
            name="test-domain",
            profile={
                "domain": "Test Domain",
                "groups": [
                    {
                        "key": "core",
                        "label": "Core Features",
                        "features": [
                            {"key": "auth", "label": "Authentication"},
                            {"key": "api", "label": "API Access"},
                        ],
                    },
                    {
                        "key": "advanced",
                        "label": "Advanced Features",
                        "features": [
                            {"key": "analytics", "label": "Analytics"},
                        ],
                    },
                ],
            },
        )

        # Create competitors with new features schema (source counts)
        comp_a = store.create_entity(
            type_key="competitor",
            name="Verified Corp",
            profile={
                "description": "Has verified features",
                "features": {
                    "auth": {"sources": 3},  # Verified (3 sources)
                    "api": {"sources": 2},   # Verified (2 sources)
                    "analytics": {"sources": 1},  # Unverified (1 source)
                },
            },
            collection="research",
        )

        comp_b = store.create_entity(
            type_key="competitor",
            name="Sparse Corp",
            profile={
                "description": "Has fewer features",
                "features": {
                    "auth": {"sources": 1},  # Unverified
                    "extra-feature": {"sources": 2},  # Not in taxonomy
                },
            },
            collection="research",
        )

        return {"taxonomy": taxonomy, "comp_a": comp_a, "comp_b": comp_b}

    def test_taxonomy_grouping(self, store: KnowledgeStore, setup_taxonomy_and_competitors):
        """Report groups features by taxonomy structure."""
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research", taxonomy="test-domain")

        groups = result.data["groups"]

        # Should have 3 groups: core, advanced, and _ungrouped
        group_keys = [g["key"] for g in groups]
        assert "core" in group_keys
        assert "advanced" in group_keys
        assert "_ungrouped" in group_keys

        # Core should have auth and api
        core_group = next(g for g in groups if g["key"] == "core")
        core_feature_keys = [f["key"] for f in core_group["features"]]
        assert "auth" in core_feature_keys
        assert "api" in core_feature_keys

        # Ungrouped should have extra-feature
        ungrouped = next(g for g in groups if g["key"] == "_ungrouped")
        ungrouped_keys = [f["key"] for f in ungrouped["features"]]
        assert "extra-feature" in ungrouped_keys

    def test_verification_status(self, store: KnowledgeStore, setup_taxonomy_and_competitors):
        """Features show correct verification status based on source counts."""
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research", taxonomy="test-domain")

        # Find auth feature in core group
        core_group = next(g for g in result.data["groups"] if g["key"] == "core")
        auth_feat = next(f for f in core_group["features"] if f["key"] == "auth")

        # Verified Corp has 3 sources for auth (verified)
        assert auth_feat["competitors"]["Verified Corp"]["sources"] == 3
        assert auth_feat["competitors"]["Verified Corp"]["verified"] is True

        # Sparse Corp has 1 source for auth (unverified)
        assert auth_feat["competitors"]["Sparse Corp"]["sources"] == 1
        assert auth_feat["competitors"]["Sparse Corp"]["verified"] is False

    def test_metadata_includes_taxonomy_name(
        self, store: KnowledgeStore, setup_taxonomy_and_competitors
    ):
        """Report metadata includes taxonomy name when specified."""
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research", taxonomy="test-domain")

        assert result.metadata["taxonomy"] == "test-domain"
        assert result.metadata["total_features"] > 0

    def test_markdown_shows_verification_symbols(
        self, store: KnowledgeStore, setup_taxonomy_and_competitors
    ):
        """Markdown output uses ✓✓ for verified and ✓ for unverified."""
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research", taxonomy="test-domain")
        md = report.to_markdown(result)

        # Should have double-check for verified features
        assert "✓✓" in md
        # Should have single check for unverified
        assert "✓" in md
        # Should have legend
        assert "Verified (2+ sources)" in md

    def test_without_taxonomy_flat_structure(
        self, store: KnowledgeStore, setup_taxonomy_and_competitors
    ):
        """Without taxonomy, report uses flat structure with _all group."""
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate(collection="research")  # No taxonomy

        groups = result.data["groups"]
        assert len(groups) == 1
        assert groups[0]["key"] == "_all"
        assert groups[0]["label"] == "All Features"

        # All features should be in the single group
        feature_keys = [f["key"] for f in groups[0]["features"]]
        assert "auth" in feature_keys
        assert "api" in feature_keys
        assert "analytics" in feature_keys
        assert "extra-feature" in feature_keys


class TestFundraisingTimelineReport:
    """Test fundraising timeline report generation."""

    @pytest.fixture
    def setup_funding_data(self, store: KnowledgeStore):
        """Create competitors with funding data."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        # High funding
        entity_a = store.create_entity(
            type_key="competitor",
            name="Well Funded Co",
            profile={
                "name": "Well Funded Co",
                "funding": {
                    "total": 100000000,
                    "rounds": ["Seed", "Series A", "Series B"],
                },
            },
        )

        # Lower funding
        entity_b = store.create_entity(
            type_key="competitor",
            name="Startup Inc",
            profile={
                "name": "Startup Inc",
                "funding": {
                    "total": 5000000,
                    "rounds": ["Seed"],
                },
            },
        )

        # No funding data
        entity_c = store.create_entity(
            type_key="competitor",
            name="Bootstrapped LLC",
            profile={
                "name": "Bootstrapped LLC",
            },
        )

        return {"a": entity_a, "b": entity_b, "c": entity_c}

    def test_funding_sorted_by_total_descending(self, store: KnowledgeStore, setup_funding_data):
        """Funding data sorted by total funding, descending."""
        report = FundraisingTimelineReport(store)
        result = report.generate()

        funding = result.data["funding"]

        # Well Funded first ($100M), then Startup ($5M), then Bootstrapped (None)
        assert funding[0]["name"] == "Well Funded Co"
        assert funding[0]["total"] == 100000000
        assert funding[1]["name"] == "Startup Inc"
        assert funding[1]["total"] == 5000000
        assert funding[2]["name"] == "Bootstrapped LLC"
        assert funding[2]["total"] is None

    def test_rounds_included(self, store: KnowledgeStore, setup_funding_data):
        """Funding rounds are included in output."""
        report = FundraisingTimelineReport(store)
        result = report.generate()

        funding = result.data["funding"]
        well_funded = next(f for f in funding if f["name"] == "Well Funded Co")

        assert well_funded["rounds"] == ["Seed", "Series A", "Series B"]

    def test_to_markdown_format(self, store: KnowledgeStore, setup_funding_data):
        """to_markdown produces readable funding report."""
        report = FundraisingTimelineReport(store)
        result = report.generate()

        md = report.to_markdown(result)

        assert "# Fundraising Timeline" in md
        assert "## Well Funded Co" in md
        assert "**Total Funding:** $100,000,000" in md
        assert "**Rounds:** Seed, Series A, Series B" in md
        assert "Not disclosed" in md  # For Bootstrapped LLC

    def test_citations_for_funding(self, store: KnowledgeStore, setup_funding_data):
        """Citations included when evidence exists for funding."""
        data = setup_funding_data

        source = store.add_source("https://techcrunch.com/funding", "website")
        chunk_id = store.add_chunk(
            source=source,
            content="Well Funded Co raised $100M in Series B.",
            chunk_index=0,
            embedding=[0.1] * 384,
            model_id="test",
        )
        store.add_evidence(
            entity_id=data["a"].id,
            field_path="funding.total",
            chunk_id=chunk_id,
            quote="raised $100M",
            confidence=0.9,
        )

        report = FundraisingTimelineReport(store)
        result = report.generate()

        assert result.has_citations is True
        well_funded = next(f for f in result.data["funding"] if f["name"] == "Well Funded Co")
        assert len(well_funded["citations"]) == 1


class TestWisdomDigestReport:
    """Test wisdom digest report generation."""

    @pytest.fixture
    def setup_wisdom_entries(self, store: KnowledgeStore):
        """Create wisdom entries for digest tests."""
        store.register_entity_type("wisdom", WISDOM_SCHEMA)

        entity_a = store.create_entity(
            type_key="wisdom",
            name="API Design Best Practices",
            profile={
                "topic": "API Design",
                "summary": "Guidelines for designing RESTful APIs.",
                "key_points": [
                    "Use proper HTTP methods",
                    "Version your APIs",
                    "Return appropriate status codes",
                ],
                "links": [
                    "https://restfulapi.net",
                    "https://api-guide.example.com",
                ],
            },
            collection="engineering",
        )

        entity_b = store.create_entity(
            type_key="wisdom",
            name="Database Optimization",
            profile={
                "topic": "Database Performance",
                "summary": "Tips for optimizing database queries.",
                "key_points": [
                    "Use indexes appropriately",
                    "Avoid N+1 queries",
                ],
            },
            collection="engineering",
        )

        return {"a": entity_a, "b": entity_b}

    def test_digest_groups_by_topic(self, store: KnowledgeStore, setup_wisdom_entries):
        """Digest entries organized by topic."""
        report = WisdomDigestReport(store)
        result = report.generate(collection="engineering")

        entries = result.data["entries"]

        assert len(entries) == 2
        topics = [e["topic"] for e in entries]
        assert "API Design" in topics
        assert "Database Performance" in topics

    def test_key_points_included(self, store: KnowledgeStore, setup_wisdom_entries):
        """Key points included in digest entries."""
        report = WisdomDigestReport(store)
        result = report.generate(collection="engineering")

        entries = result.data["entries"]
        api_entry = next(e for e in entries if e["topic"] == "API Design")

        assert len(api_entry["key_points"]) == 3
        assert "Use proper HTTP methods" in api_entry["key_points"]

    def test_links_included(self, store: KnowledgeStore, setup_wisdom_entries):
        """Links included in digest entries."""
        report = WisdomDigestReport(store)
        result = report.generate(collection="engineering")

        entries = result.data["entries"]
        api_entry = next(e for e in entries if e["topic"] == "API Design")

        assert len(api_entry["links"]) == 2
        assert "https://restfulapi.net" in api_entry["links"]

    def test_to_markdown_format(self, store: KnowledgeStore, setup_wisdom_entries):
        """to_markdown produces readable wisdom digest."""
        report = WisdomDigestReport(store)
        result = report.generate(collection="engineering")

        md = report.to_markdown(result)

        assert "# Wisdom Digest" in md
        assert "## API Design" in md
        assert "Guidelines for designing RESTful APIs." in md
        assert "**Key Points:**" in md
        assert "- Use proper HTTP methods" in md
        assert "**Links:**" in md

    def test_deterministic_sorting(self, store: KnowledgeStore, setup_wisdom_entries):
        """Entries sorted by topic for deterministic output."""
        report = WisdomDigestReport(store)

        result1 = report.generate(collection="engineering")
        result2 = report.generate(collection="engineering")

        # Should be sorted by topic
        topics = [e["topic"] for e in result1.data["entries"]]
        assert topics == sorted(topics)

        # Results should be identical
        assert result1.data == result2.data


class TestReportMetadata:
    """Test report metadata fields."""

    def test_report_type_property(self, store: KnowledgeStore):
        """Each report has correct report_type."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        matrix = CompetitorFeatureMatrixReport(store)
        assert matrix.report_type == "competitor-feature-matrix"

        timeline = FundraisingTimelineReport(store)
        assert timeline.report_type == "fundraising-timeline"

        digest = WisdomDigestReport(store)
        assert digest.report_type == "wisdom-digest"

    def test_generated_at_timestamp(self, store: KnowledgeStore):
        """Reports include generation timestamp."""
        store.register_entity_type("competitor", COMPETITOR_SCHEMA)

        before = datetime.utcnow()
        report = CompetitorFeatureMatrixReport(store)
        result = report.generate()
        after = datetime.utcnow()

        assert before <= result.generated_at <= after
