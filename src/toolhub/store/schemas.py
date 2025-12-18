"""JSON Schemas for entity types.

This module defines the canonical JSON schemas for built-in entity types
like feature_taxonomy and competitor. These schemas are registered via
register_entity_type() when the system initializes.
"""

from __future__ import annotations

# Feature Taxonomy Schema
# Holds the canonical feature structure with groups and features
FEATURE_TAXONOMY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["domain", "groups"],
    "properties": {
        "domain": {
            "type": "string",
            "description": "Domain name for this taxonomy (e.g., 'K-12 Fundraising')",
        },
        "groups": {
            "type": "array",
            "description": "Feature groups containing related features",
            "items": {
                "type": "object",
                "required": ["key", "label", "features"],
                "properties": {
                    "key": {
                        "type": "string",
                        "pattern": "^[a-z0-9-]+$",
                        "description": "URL-safe unique identifier for the group",
                    },
                    "label": {
                        "type": "string",
                        "description": "Human-readable group name",
                    },
                    "features": {
                        "type": "array",
                        "description": "Features in this group",
                        "items": {
                            "type": "object",
                            "required": ["key", "label"],
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "pattern": "^[a-z0-9-]+$",
                                    "description": "URL-safe unique identifier for the feature",
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Human-readable feature name",
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

# Updated Competitor Schema with features as object mapping feature-key to source counts
COMPETITOR_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "Brief description of the competitor",
        },
        "website": {
            "type": "string",
            "format": "uri",
            "description": "Competitor's website URL",
        },
        "features": {
            "type": "object",
            "description": "Map of feature-key to source count info",
            "additionalProperties": {
                "type": "object",
                "required": ["sources"],
                "properties": {
                    "sources": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of sources confirming this feature",
                    },
                },
            },
        },
        "funding": {
            "type": "object",
            "description": "Funding information",
            "properties": {
                "total": {
                    "type": ["number", "null"],
                    "description": "Total funding amount in USD",
                },
                "rounds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of funding rounds (e.g., 'Series A', 'Series B')",
                },
            },
        },
        "team_size": {
            "type": "integer",
            "minimum": 0,
            "description": "Approximate team size",
        },
    },
}

# Registry of built-in schemas
BUILTIN_SCHEMAS = {
    "feature_taxonomy": {
        "schema": FEATURE_TAXONOMY_SCHEMA,
        "description": "Grouped feature taxonomy for organizing features into categories",
    },
    "competitor": {
        "schema": COMPETITOR_SCHEMA,
        "description": "Competitor profile with features, funding, and team info",
    },
}


def register_builtin_schemas(store) -> list[str]:
    """Register all built-in entity type schemas.

    Args:
        store: KnowledgeStore instance

    Returns:
        List of type_keys that were registered
    """
    registered = []
    for type_key, info in BUILTIN_SCHEMAS.items():
        store.register_entity_type(
            type_key=type_key,
            json_schema=info["schema"],
            description=info["description"],
        )
        registered.append(type_key)
    return registered
