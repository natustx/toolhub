"""OpenAPI/Swagger spec parser for extracting API operations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from toolhub.store.operations import Operation

logger = logging.getLogger(__name__)

# File patterns that might be OpenAPI specs
OPENAPI_PATTERNS = [
    "openapi.json",
    "openapi.yaml",
    "openapi.yml",
    "swagger.json",
    "swagger.yaml",
    "swagger.yml",
    "api.json",
    "api.yaml",
    "api.yml",
]


def is_openapi_file(path: Path) -> bool:
    """Check if a file might be an OpenAPI spec based on name."""
    name_lower = path.name.lower()
    return name_lower in OPENAPI_PATTERNS or any(
        pattern in name_lower for pattern in ["openapi", "swagger"]
    )


def load_spec(path: Path) -> dict | None:
    """Load an OpenAPI spec from file.

    Args:
        path: Path to the spec file (JSON or YAML)

    Returns:
        Parsed spec dictionary, or None if invalid
    """
    try:
        content = path.read_text(encoding="utf-8")

        # Try JSON first
        if path.suffix.lower() == ".json":
            return json.loads(content)

        # Try YAML
        try:
            import yaml

            return yaml.safe_load(content)
        except ImportError:
            # YAML not available, try JSON anyway
            return json.loads(content)

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"Failed to parse {path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error loading spec from {path}: {e}")
        return None


def is_valid_openapi(spec: dict) -> bool:
    """Check if a dictionary looks like a valid OpenAPI spec."""
    # Check for OpenAPI 3.x
    if "openapi" in spec and "paths" in spec:
        return True

    # Check for Swagger 2.x
    if "swagger" in spec and "paths" in spec:
        return True

    return False


def extract_operations(spec: dict, tool_id: str) -> list[Operation]:
    """Extract operations from an OpenAPI spec.

    Args:
        spec: Parsed OpenAPI spec dictionary
        tool_id: Tool identifier to associate operations with

    Returns:
        List of Operation objects
    """
    operations: list[Operation] = []
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        # Handle $ref at path level (not commonly used but possible)
        if "$ref" in path_item:
            continue

        for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
            if method not in path_item:
                continue

            op_data = path_item[method]
            if not isinstance(op_data, dict):
                continue

            # Generate operation_id if not present
            op_id = op_data.get("operationId") or f"{method}_{path.replace('/', '_').strip('_')}"

            # Extract parameters
            parameters = _extract_parameters(op_data.get("parameters", []))

            # Also include path-level parameters
            path_params = _extract_parameters(path_item.get("parameters", []))
            parameters = path_params + parameters

            # Extract request body (OpenAPI 3.x)
            request_body = _extract_request_body(op_data.get("requestBody"))

            # Extract responses
            responses = _extract_responses(op_data.get("responses", {}))

            operations.append(
                Operation(
                    tool_id=tool_id,
                    operation_id=op_id,
                    method=method.upper(),
                    path=path,
                    summary=op_data.get("summary", ""),
                    description=op_data.get("description", ""),
                    tags=op_data.get("tags", []),
                    parameters=parameters,
                    request_body=request_body,
                    responses=responses,
                )
            )

    logger.info(f"Extracted {len(operations)} operations from spec")
    return operations


def _extract_parameters(params: list[dict]) -> list[dict]:
    """Extract simplified parameter info."""
    result = []
    for param in params:
        if not isinstance(param, dict):
            continue

        # Skip $ref parameters for simplicity
        if "$ref" in param:
            continue

        result.append(
            {
                "name": param.get("name", ""),
                "in": param.get("in", ""),  # path, query, header, cookie
                "required": param.get("required", False),
                "description": param.get("description", ""),
                "type": _get_param_type(param),
            }
        )

    return result


def _get_param_type(param: dict) -> str:
    """Get parameter type from schema."""
    # OpenAPI 3.x
    if "schema" in param:
        schema = param["schema"]
        return schema.get("type", "string")

    # Swagger 2.x
    return param.get("type", "string")


def _extract_request_body(request_body: dict | None) -> dict | None:
    """Extract simplified request body info."""
    if not request_body or not isinstance(request_body, dict):
        return None

    content = request_body.get("content", {})
    if not content:
        return None

    # Get first content type
    content_types = list(content.keys())
    if not content_types:
        return None

    return {
        "content_type": content_types[0],
        "required": request_body.get("required", False),
        "description": request_body.get("description", ""),
    }


def _extract_responses(responses: dict) -> dict[str, str]:
    """Extract simplified response info."""
    result = {}
    for status_code, response in responses.items():
        if not isinstance(response, dict):
            continue

        # Skip $ref responses for simplicity
        if "$ref" in response:
            result[str(status_code)] = "(referenced response)"
            continue

        result[str(status_code)] = response.get("description", "")

    return result


def parse_openapi_files(
    directory: Path,
    tool_id: str,
) -> list[Operation]:
    """Find and parse OpenAPI specs in a directory.

    Args:
        directory: Directory to search for spec files
        tool_id: Tool identifier for operations

    Returns:
        List of operations from all valid specs found
    """
    all_operations: list[Operation] = []

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        if not is_openapi_file(file_path):
            continue

        spec = load_spec(file_path)
        if spec is None:
            continue

        if not is_valid_openapi(spec):
            continue

        logger.info(f"Found OpenAPI spec: {file_path}")
        operations = extract_operations(spec, tool_id)
        all_operations.extend(operations)

    return all_operations


def format_operation_as_text(op: Operation) -> str:
    """Format an operation as human-readable text for embedding.

    Args:
        op: Operation to format

    Returns:
        Formatted text suitable for semantic search
    """
    lines = [f"{op.method} {op.path}"]

    if op.summary:
        lines.append(op.summary)

    if op.description:
        lines.append(op.description)

    if op.tags:
        lines.append(f"Tags: {', '.join(op.tags)}")

    if op.parameters:
        param_strs = []
        for p in op.parameters:
            param_str = f"{p['name']} ({p['in']})"
            if p.get("required"):
                param_str += " [required]"
            param_strs.append(param_str)
        lines.append(f"Parameters: {', '.join(param_strs)}")

    if op.request_body:
        lines.append(f"Request body: {op.request_body.get('content_type', 'unknown')}")

    return "\n".join(lines)
