"""Search module for querying indexed documentation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from toolhub.indexer.embedder import embed_text
from toolhub.store.lance import VectorStore, list_tool_stores
from toolhub.store.operations import Operation, OperationsStore

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Output format for search results."""

    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class SearchResult:
    """A single search result."""

    tool_id: str
    content: str
    source_file: str
    heading: str
    heading_path: str
    is_code: bool
    distance: float  # Cosine distance (lower = more similar)

    @property
    def similarity(self) -> float:
        """Convert distance to similarity score (0-1, higher = more similar)."""
        return 1.0 - self.distance

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "tool_id": self.tool_id,
            "content": self.content,
            "source_file": self.source_file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "is_code": self.is_code,
            "similarity": round(self.similarity, 4),
        }

    def to_markdown(self) -> str:
        """Format as markdown."""
        lines = []

        # Header with source info
        heading_info = f" > {self.heading_path}" if self.heading_path else ""
        lines.append(f"### {self.tool_id}: {self.source_file}{heading_info}")
        lines.append(f"*Similarity: {self.similarity:.2%}*")
        lines.append("")

        # Content (already markdown, or wrap in code block if code)
        if self.is_code and not self.content.startswith("```"):
            lines.append("```")
            lines.append(self.content)
            lines.append("```")
        else:
            lines.append(self.content)

        lines.append("")
        return "\n".join(lines)


@dataclass
class OperationResult:
    """A single API operation result."""

    operation: Operation
    rank: int  # FTS rank (lower = better match)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = self.operation.to_dict()
        d["result_type"] = "operation"
        return d

    def to_markdown(self) -> str:
        """Format as markdown."""
        op = self.operation
        lines = []
        lines.append(f"### {op.tool_id}: {op.method} {op.path}")

        if op.summary:
            lines.append(f"*{op.summary}*")
        lines.append("")

        if op.description:
            lines.append(op.description)
            lines.append("")

        if op.parameters:
            lines.append("**Parameters:**")
            for p in op.parameters:
                req = " (required)" if p.get("required") else ""
                lines.append(f"- `{p['name']}` ({p['in']}){req}: {p.get('description', '')}")
            lines.append("")

        if op.request_body:
            lines.append(f"**Request body:** {op.request_body.get('content_type', 'unknown')}")
            lines.append("")

        if op.responses:
            lines.append("**Responses:**")
            for code, desc in op.responses.items():
                lines.append(f"- {code}: {desc}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class SearchResponse:
    """Response containing search results."""

    query: str
    results: list[SearchResult]
    tools_searched: list[str]
    operations: list[OperationResult] | None = None

    def to_json(self, indent: int = 2) -> str:
        """Format as JSON string."""
        data = {
            "query": self.query,
            "tools_searched": self.tools_searched,
            "result_count": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }
        if self.operations:
            data["operations_count"] = len(self.operations)
            data["operations"] = [r.to_dict() for r in self.operations]
        return json.dumps(data, indent=indent)

    def to_markdown(self) -> str:
        """Format as markdown."""
        lines = []
        lines.append(f"## Search: {self.query}")
        lines.append(f"*Searched: {', '.join(self.tools_searched) or 'all tools'}*")

        total = len(self.results) + (len(self.operations) if self.operations else 0)
        lines.append(f"*Found: {total} results*")
        lines.append("")

        # Show operations first if present
        if self.operations:
            lines.append("### API Operations")
            lines.append("")
            for op_result in self.operations:
                lines.append(op_result.to_markdown())
                lines.append("---")
                lines.append("")

        if self.results:
            if self.operations:
                lines.append("### Documentation")
                lines.append("")
            for result in self.results:
                lines.append(result.to_markdown())
                lines.append("---")
                lines.append("")

        if not self.results and not self.operations:
            lines.append("No results found.")

        return "\n".join(lines)

    def format(self, output_format: OutputFormat | str) -> str:
        """Format results in the specified format."""
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format)

        if output_format == OutputFormat.JSON:
            return self.to_json()
        return self.to_markdown()


def search(
    query: str,
    tool_ids: list[str] | None = None,
    limit: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    include_operations: bool = True,
    timings: dict | None = None,
) -> SearchResponse:
    """Search indexed documentation and API operations.

    Performs hybrid search combining:
    - Semantic vector search on documentation chunks
    - Full-text search on API operations (if available)

    Args:
        query: Natural language search query
        tool_ids: Optional list of tool IDs to search (None = all)
        limit: Maximum results per tool
        model_name: Embedding model to use
        include_operations: Whether to search API operations
        timings: Optional dict to store timing breakdown

    Returns:
        SearchResponse with results
    """
    import time

    # Get query embedding
    t0 = time.perf_counter()
    embed_timings: dict[str, float] = {}
    query_vector = embed_text(query, model_name=model_name, timings=embed_timings)
    if timings is not None:
        timings["embed_query"] = time.perf_counter() - t0
        # Add sub-timings with prefix
        for k, v in embed_timings.items():
            timings[f"embed_{k}"] = v

    # Determine which tools to search
    if tool_ids:
        tools_to_search = tool_ids
    else:
        tools_to_search = list_tool_stores()

    # Search each tool's vector store
    t0 = time.perf_counter()
    all_results: list[SearchResult] = []

    for tool_id in tools_to_search:
        store = VectorStore(tool_id)
        if not store.exists():
            continue

        hits = store.search(query_vector.tolist(), limit=limit)
        for hit in hits:
            all_results.append(
                SearchResult(
                    tool_id=tool_id,
                    content=hit["content"],
                    source_file=hit["source_file"],
                    heading=hit["heading"],
                    heading_path=hit["heading_path"],
                    is_code=hit["is_code"],
                    distance=hit["distance"],
                )
            )

    # Sort by similarity (lowest distance first)
    all_results.sort(key=lambda r: r.distance)

    # Limit total results
    all_results = all_results[:limit]
    if timings is not None:
        timings["vector_search"] = time.perf_counter() - t0

    # Search operations if enabled
    t0 = time.perf_counter()
    operation_results: list[OperationResult] | None = None
    if include_operations:
        try:
            ops_store = OperationsStore()
            # Search for tool_id filter
            tool_filter = tool_ids[0] if tool_ids and len(tool_ids) == 1 else None
            ops = ops_store.search(query, tool_id=tool_filter, limit=limit)
            if ops:
                operation_results = [OperationResult(op, i) for i, op in enumerate(ops)]
        except Exception as e:
            # Operations search is optional, log and continue
            logger.debug(f"Operations search failed: {e}")
    if timings is not None:
        timings["ops_search"] = time.perf_counter() - t0

    return SearchResponse(
        query=query,
        results=all_results,
        tools_searched=tools_to_search,
        operations=operation_results,
    )


def search_tool(
    tool_id: str,
    query: str,
    limit: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> SearchResponse:
    """Search a single tool's documentation.

    Convenience wrapper around search().

    Args:
        tool_id: Tool to search
        query: Natural language search query
        limit: Maximum results
        model_name: Embedding model to use

    Returns:
        SearchResponse with results
    """
    return search(query, tool_ids=[tool_id], limit=limit, model_name=model_name)
