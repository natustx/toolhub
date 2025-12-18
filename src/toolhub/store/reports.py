"""Report generation framework for toolhub.

Provides deterministic report generation from entities and evidence:
- Base Report class with generate() returning ReportResult
- ReportResult with to_markdown() and to_json() output methods
- v1 reports: feature matrix, fundraising timeline, wisdom digest
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toolhub.store.knowledge import KnowledgeStore


@dataclass
class ReportResult:
    """Result from report generation."""

    title: str
    report_type: str
    generated_at: datetime
    data: dict
    entities_count: int
    has_citations: bool
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "report_type": self.report_type,
            "generated_at": self.generated_at.isoformat(),
            "entities_count": self.entities_count,
            "has_citations": self.has_citations,
            "metadata": self.metadata,
            "data": self.data,
        }

    def to_markdown(self) -> str:
        """Convert to markdown string. Subclasses may override."""
        lines = [f"# {self.title}", ""]
        lines.append(f"*Generated: {self.generated_at.isoformat()}*")
        lines.append(f"*Entities: {self.entities_count}*")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(self.data, indent=2, default=str))
        lines.append("```")
        return "\n".join(lines)


class Report(ABC):
    """Base class for all reports."""

    def __init__(self, store: KnowledgeStore):
        self.store = store

    @property
    @abstractmethod
    def report_type(self) -> str:
        """Unique identifier for this report type."""
        pass

    @abstractmethod
    def generate(self, **kwargs) -> ReportResult:
        """Generate the report with given parameters."""
        pass


class CompetitorFeatureMatrixReport(Report):
    """Generate a feature comparison matrix across competitors.

    Creates a table showing which features each competitor supports,
    grouped by taxonomy (if provided), with verification status based
    on source counts.

    Legend:
    - ✓✓ = Verified (2+ sources)
    - ✓ = Unverified (1 source)
    - — = Not found
    """

    @property
    def report_type(self) -> str:
        return "competitor-feature-matrix"

    def _get_feature_sources(self, features: dict | list, feature_key: str) -> int:
        """Get source count for a feature, handling both old and new schemas.

        Args:
            features: Either dict {key: {sources: N}} or list [key1, key2]
            feature_key: The feature key to look up

        Returns:
            Number of sources (0 if not present)
        """
        if isinstance(features, dict):
            # New schema: {feature-key: {sources: N}}
            feature_data = features.get(feature_key, {})
            if isinstance(feature_data, dict):
                return feature_data.get("sources", 0)
            return 0
        elif isinstance(features, list):
            # Old schema: [feature1, feature2, ...]
            return 1 if feature_key in features else 0
        return 0

    def _extract_feature_keys(self, features: dict | list) -> set[str]:
        """Extract feature keys from either schema format.

        Args:
            features: Either dict {key: {sources: N}} or list [key1, key2]

        Returns:
            Set of feature keys
        """
        if isinstance(features, dict):
            return set(features.keys())
        elif isinstance(features, list):
            return set(features)
        return set()

    def generate(
        self,
        entity_ids: list[str] | None = None,
        tags: list[str] | None = None,
        collection: str | None = None,
        taxonomy: str | None = None,
    ) -> ReportResult:
        """Generate feature matrix report.

        Args:
            entity_ids: Specific entity IDs to include (optional)
            tags: Filter entities by tags (optional)
            collection: Filter entities by collection (optional)
            taxonomy: Taxonomy name for feature grouping (optional)

        Returns:
            ReportResult with feature matrix data
        """
        import uuid

        # Get competitor entities
        if entity_ids:
            entities = [self.store.get_entity(uuid.UUID(eid)) for eid in entity_ids]
            entities = [e for e in entities if e is not None]
        else:
            entities = self.store.list_entities(
                type_key="competitor",
                tags=tags,
                collection=collection,
            )

        # Sort entities by name for deterministic output
        entities = sorted(entities, key=lambda e: e.name)

        # Load taxonomy if specified
        taxonomy_entity = None
        taxonomy_groups: list[dict] = []
        if taxonomy:
            # Try to find taxonomy entity by name
            taxonomy_entity = self.store.get_entity_by_name(
                type_key="feature_taxonomy",
                name=taxonomy,
            )
            if taxonomy_entity:
                taxonomy_groups = taxonomy_entity.profile.get("groups", [])

        # Collect all features from competitors
        all_feature_keys: set[str] = set()
        entity_features: dict[str, dict] = {}

        for entity in entities:
            features = entity.profile.get("features", {})
            feature_keys = self._extract_feature_keys(features)
            all_feature_keys.update(feature_keys)
            entity_features[entity.name] = {
                "id": str(entity.id),
                "features": features,
            }

        # Build grouped structure if taxonomy provided, else flat
        if taxonomy_groups:
            groups_data = self._build_grouped_matrix(
                taxonomy_groups, entities, entity_features, all_feature_keys
            )
        else:
            groups_data = self._build_flat_matrix(
                entities, entity_features, sorted(all_feature_keys)
            )

        # Get citations grouped by competitor
        citations_by_entity: dict[str, list[dict]] = {}
        has_any_citations = False

        for entity in entities:
            citations = self.store.get_evidence_with_citations(entity.id)
            entity_citations: list[dict] = []
            for field_path, cites in citations.items():
                if field_path.startswith("features."):
                    has_any_citations = True
                    for cite in cites:
                        cite_dict = cite.to_dict()
                        cite_dict["feature_key"] = field_path.replace("features.", "")
                        entity_citations.append(cite_dict)
            if entity_citations:
                citations_by_entity[entity.name] = entity_citations

        # Count features and verified features
        verified_count = 0
        total_features = len(all_feature_keys)
        for group in groups_data:
            for feat in group.get("features", []):
                for comp_data in feat.get("competitors", {}).values():
                    if comp_data.get("verified"):
                        verified_count += 1
                        break  # Count feature as verified if any competitor has it verified

        return ReportResult(
            title="Competitor Feature Matrix",
            report_type=self.report_type,
            generated_at=datetime.utcnow(),
            entities_count=len(entities),
            has_citations=has_any_citations,
            metadata={
                "taxonomy": taxonomy,
                "total_features": total_features,
                "verified_features": verified_count,
            },
            data={
                "competitors": list(entity_features.keys()),
                "groups": groups_data,
                "citations": citations_by_entity,
            },
        )

    def _build_grouped_matrix(
        self,
        taxonomy_groups: list[dict],
        entities: list,
        entity_features: dict[str, dict],
        all_feature_keys: set[str],
    ) -> list[dict]:
        """Build matrix grouped by taxonomy structure."""
        groups_data: list[dict] = []
        assigned_keys: set[str] = set()

        for group in taxonomy_groups:
            group_key = group.get("key", "")
            group_label = group.get("label", group_key)
            group_features = group.get("features", [])

            features_data: list[dict] = []
            for feat in group_features:
                feat_key = feat.get("key", "")
                feat_label = feat.get("label", feat_key)
                assigned_keys.add(feat_key)

                competitors_data = {}
                for entity in entities:
                    features = entity_features[entity.name]["features"]
                    sources = self._get_feature_sources(features, feat_key)
                    competitors_data[entity.name] = {
                        "sources": sources,
                        "verified": sources >= 2,
                        "present": sources > 0,
                    }

                features_data.append({
                    "key": feat_key,
                    "label": feat_label,
                    "competitors": competitors_data,
                })

            groups_data.append({
                "key": group_key,
                "label": group_label,
                "features": features_data,
            })

        # Add ungrouped features (features in competitors but not in taxonomy)
        ungrouped_keys = all_feature_keys - assigned_keys
        if ungrouped_keys:
            ungrouped_features: list[dict] = []
            for feat_key in sorted(ungrouped_keys):
                competitors_data = {}
                for entity in entities:
                    features = entity_features[entity.name]["features"]
                    sources = self._get_feature_sources(features, feat_key)
                    competitors_data[entity.name] = {
                        "sources": sources,
                        "verified": sources >= 2,
                        "present": sources > 0,
                    }
                ungrouped_features.append({
                    "key": feat_key,
                    "label": feat_key,  # Use key as label for ungrouped
                    "competitors": competitors_data,
                })

            groups_data.append({
                "key": "_ungrouped",
                "label": "Other Features",
                "features": ungrouped_features,
            })

        return groups_data

    def _build_flat_matrix(
        self,
        entities: list,
        entity_features: dict[str, dict],
        sorted_features: list[str],
    ) -> list[dict]:
        """Build flat matrix without taxonomy grouping."""
        features_data: list[dict] = []

        for feat_key in sorted_features:
            competitors_data = {}
            for entity in entities:
                features = entity_features[entity.name]["features"]
                sources = self._get_feature_sources(features, feat_key)
                competitors_data[entity.name] = {
                    "sources": sources,
                    "verified": sources >= 2,
                    "present": sources > 0,
                }
            features_data.append({
                "key": feat_key,
                "label": feat_key,
                "competitors": competitors_data,
            })

        return [{
            "key": "_all",
            "label": "All Features",
            "features": features_data,
        }]

    def to_markdown(self, result: ReportResult) -> str:
        """Render feature matrix as markdown with grouped tables."""
        lines = [f"# {result.title}", ""]
        lines.append(f"*Generated: {result.generated_at.isoformat()}*")

        metadata = result.metadata
        lines.append(f"*Competitors: {result.entities_count}*")
        taxonomy_name = metadata.get("taxonomy")
        if taxonomy_name:
            lines.append(f"*Taxonomy: {taxonomy_name}*")
        total_feat = metadata.get("total_features", 0)
        verified_feat = metadata.get("verified_features", 0)
        lines.append(f"*Features: {total_feat} ({verified_feat} verified)*")
        lines.append("")

        # Legend
        lines.append("**Legend:** ✓✓ = Verified (2+ sources) | ✓ = Unverified (1 source)")
        lines.append("| — = Not found")
        lines.append("")

        groups = result.data.get("groups", [])
        competitors = result.data.get("competitors", [])

        if not competitors:
            lines.append("*No competitors found.*")
            return "\n".join(lines)

        # Build table for each group
        for group in groups:
            group_label = group.get("label", "Features")
            features = group.get("features", [])

            if not features:
                continue

            lines.append(f"## {group_label}")
            lines.append("")

            # Table header
            header = "| Feature | " + " | ".join(competitors) + " |"
            separator = "|" + "|".join(["---"] * (len(competitors) + 1)) + "|"
            lines.append(header)
            lines.append(separator)

            # Table rows
            for feat in features:
                feat_label = feat.get("label", feat.get("key", ""))
                cells = [feat_label]

                for comp_name in competitors:
                    comp_data = feat.get("competitors", {}).get(comp_name, {})
                    sources = comp_data.get("sources", 0)
                    if sources >= 2:
                        cells.append("✓✓")
                    elif sources == 1:
                        cells.append("✓")
                    else:
                        cells.append("—")

                lines.append("| " + " | ".join(cells) + " |")

            lines.append("")

        # Add citations section
        citations = result.data.get("citations", {})
        if citations:
            lines.append("## Sources")
            lines.append("")
            for entity_name, cites in sorted(citations.items()):
                lines.append(f"### {entity_name}")
                for cite in cites:
                    feature_key = cite.get("feature_key", "")
                    if feature_key:
                        lines.append(f"**{feature_key}**")
                    if cite.get("quote"):
                        lines.append(f'> "{cite["quote"]}"')
                    source = cite.get("source_file") or cite.get("source_url", "")
                    if cite.get("heading_path"):
                        source = f"{source} > {cite['heading_path']}"
                    if source:
                        lines.append(f"— *{source}*")
                    lines.append("")

        return "\n".join(lines)


class FundraisingTimelineReport(Report):
    """Generate a fundraising timeline for competitors.

    Shows funding rounds and amounts over time, with evidence citations.
    """

    @property
    def report_type(self) -> str:
        return "fundraising-timeline"

    def generate(
        self,
        entity_ids: list[str] | None = None,
        tags: list[str] | None = None,
        collection: str | None = None,
    ) -> ReportResult:
        """Generate fundraising timeline report."""
        import uuid

        # Get entities
        if entity_ids:
            entities = [self.store.get_entity(uuid.UUID(eid)) for eid in entity_ids]
            entities = [e for e in entities if e is not None]
        else:
            entities = self.store.list_entities(
                type_key="competitor",
                tags=tags,
                collection=collection,
            )

        # Sort entities by name for deterministic output
        entities = sorted(entities, key=lambda e: e.name)

        # Collect funding data
        funding_data: list[dict] = []
        has_any_citations = False

        for entity in entities:
            funding = entity.profile.get("funding", {})
            total = funding.get("total")
            rounds = funding.get("rounds", [])

            # Get citations for funding fields
            citations = self.store.get_evidence_with_citations(entity.id)
            funding_citations = []
            for field_path in ["funding", "funding.total", "funding.rounds"]:
                if field_path in citations:
                    has_any_citations = True
                    funding_citations.extend([c.to_dict() for c in citations[field_path]])

            funding_data.append(
                {
                    "entity_id": str(entity.id),
                    "name": entity.name,
                    "total": total,
                    "rounds": sorted(rounds) if rounds else [],
                    "citations": funding_citations,
                }
            )

        # Sort by total funding (descending, None values last)
        funding_data.sort(key=lambda x: (x["total"] is None, -(x["total"] or 0)))

        return ReportResult(
            title="Fundraising Timeline",
            report_type=self.report_type,
            generated_at=datetime.utcnow(),
            entities_count=len(entities),
            has_citations=has_any_citations,
            data={"funding": funding_data},
        )

    def to_markdown(self, result: ReportResult) -> str:
        """Render fundraising timeline as markdown."""
        lines = [f"# {result.title}", ""]
        lines.append(f"*Generated: {result.generated_at.isoformat()}*")
        lines.append(f"*Competitors: {result.entities_count}*")
        lines.append("")

        funding_data = result.data.get("funding", [])

        if not funding_data:
            lines.append("*No funding data found.*")
            return "\n".join(lines)

        for entry in funding_data:
            name = entry["name"]
            total = entry["total"]
            rounds = entry["rounds"]

            lines.append(f"## {name}")
            if total is not None:
                lines.append(f"**Total Funding:** ${total:,.0f}")
            else:
                lines.append("**Total Funding:** Not disclosed")

            if rounds:
                lines.append(f"**Rounds:** {', '.join(rounds)}")

            # Add citations
            if entry.get("citations"):
                lines.append("")
                lines.append("*Sources:*")
                for cite in entry["citations"]:
                    if cite.get("quote"):
                        lines.append(f'> "{cite["quote"]}"')
                    source = cite.get("source_file") or cite.get("source_url", "")
                    lines.append(f"— *{source}*")

            lines.append("")

        return "\n".join(lines)


class WisdomDigestReport(Report):
    """Generate a digest of wisdom/insights by topic.

    Groups wisdom entries by topic and includes key points with citations.
    """

    @property
    def report_type(self) -> str:
        return "wisdom-digest"

    def generate(
        self,
        entity_ids: list[str] | None = None,
        tags: list[str] | None = None,
        collection: str | None = None,
    ) -> ReportResult:
        """Generate wisdom digest report."""
        import uuid

        # Get wisdom entities
        if entity_ids:
            entities = [self.store.get_entity(uuid.UUID(eid)) for eid in entity_ids]
            entities = [e for e in entities if e is not None]
        else:
            entities = self.store.list_entities(
                type_key="wisdom",
                tags=tags,
                collection=collection,
            )

        # Sort entities by topic for deterministic output
        entities = sorted(entities, key=lambda e: e.profile.get("topic", e.name))

        # Build digest data
        digest: list[dict] = []
        has_any_citations = False

        for entity in entities:
            topic = entity.profile.get("topic", entity.name)
            summary = entity.profile.get("summary", "")
            key_points = entity.profile.get("key_points", [])
            links = entity.profile.get("links", [])

            # Get citations
            citations = self.store.get_evidence_with_citations(entity.id)
            all_citations: list[dict] = []
            for field_citations in citations.values():
                has_any_citations = True
                all_citations.extend([c.to_dict() for c in field_citations])

            digest.append(
                {
                    "entity_id": str(entity.id),
                    "topic": topic,
                    "summary": summary,
                    "key_points": sorted(key_points) if key_points else [],
                    "links": sorted(links) if links else [],
                    "citations": all_citations,
                }
            )

        return ReportResult(
            title="Wisdom Digest",
            report_type=self.report_type,
            generated_at=datetime.utcnow(),
            entities_count=len(entities),
            has_citations=has_any_citations,
            data={"entries": digest},
        )

    def to_markdown(self, result: ReportResult) -> str:
        """Render wisdom digest as markdown."""
        lines = [f"# {result.title}", ""]
        lines.append(f"*Generated: {result.generated_at.isoformat()}*")
        lines.append(f"*Entries: {result.entities_count}*")
        lines.append("")

        entries = result.data.get("entries", [])

        if not entries:
            lines.append("*No wisdom entries found.*")
            return "\n".join(lines)

        for entry in entries:
            topic = entry["topic"]
            summary = entry["summary"]
            key_points = entry["key_points"]
            links = entry["links"]

            lines.append(f"## {topic}")
            lines.append("")
            if summary:
                lines.append(summary)
                lines.append("")

            if key_points:
                lines.append("**Key Points:**")
                for point in key_points:
                    lines.append(f"- {point}")
                lines.append("")

            if links:
                lines.append("**Links:**")
                for link in links:
                    lines.append(f"- {link}")
                lines.append("")

            # Add citations
            if entry.get("citations"):
                lines.append("*Sources:*")
                for cite in entry["citations"]:
                    if cite.get("quote"):
                        lines.append(f'> "{cite["quote"]}"')
                    source = cite.get("source_file") or cite.get("source_url", "")
                    lines.append(f"— *{source}*")
                lines.append("")

        return "\n".join(lines)


# Report registry
REPORT_TYPES: dict[str, type[Report]] = {
    "competitor-feature-matrix": CompetitorFeatureMatrixReport,
    "fundraising-timeline": FundraisingTimelineReport,
    "wisdom-digest": WisdomDigestReport,
}


def get_report(report_type: str, store: KnowledgeStore) -> Report:
    """Get a report instance by type.

    Args:
        report_type: Report type identifier
        store: KnowledgeStore instance

    Returns:
        Report instance

    Raises:
        KeyError: If report_type is unknown
    """
    if report_type not in REPORT_TYPES:
        raise KeyError(
            f"Unknown report type '{report_type}'. Available: {', '.join(REPORT_TYPES.keys())}"
        )
    return REPORT_TYPES[report_type](store)
