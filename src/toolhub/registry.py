"""Registry management for toolhub sources.

The registry (sources.json) tracks all indexed tools and their sources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from toolhub.paths import ensure_directories, get_sources_path


@dataclass
class Source:
    """A single documentation source for a tool."""

    url: str
    source_type: str  # "github", "llmstxt", "website", "openapi"
    indexed_at: datetime | None = None
    chunk_count: int = 0
    file_count: int = 0  # Actual number of files/pages crawled

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "type": self.source_type,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "chunk_count": self.chunk_count,
            "file_count": self.file_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Source:
        indexed_at = None
        if data.get("indexed_at"):
            indexed_at = datetime.fromisoformat(data["indexed_at"])

        return cls(
            url=data["url"],
            source_type=data["type"],
            indexed_at=indexed_at,
            chunk_count=data.get("chunk_count", 0),
            file_count=data.get("file_count", 0),
        )


@dataclass
class Tool:
    """A registered tool with its documentation sources."""

    tool_id: str
    display_name: str
    sources: list[Source] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(s.chunk_count for s in self.sources)

    @property
    def total_files(self) -> int:
        return sum(s.file_count for s in self.sources)

    def to_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "sources": [s.to_dict() for s in self.sources],
        }

    @classmethod
    def from_dict(cls, tool_id: str, data: dict) -> Tool:
        return cls(
            tool_id=tool_id,
            display_name=data.get("display_name", tool_id),
            sources=[Source.from_dict(s) for s in data.get("sources", [])],
        )


@dataclass
class Registry:
    """The sources registry containing all indexed tools."""

    tools: dict[str, Tool] = field(default_factory=dict)

    def get_tool(self, tool_id: str) -> Tool | None:
        return self.tools.get(tool_id)

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.tool_id] = tool

    def remove_tool(self, tool_id: str) -> bool:
        if tool_id in self.tools:
            del self.tools[tool_id]
            return True
        return False

    def add_source(self, tool_id: str, source: Source, display_name: str | None = None) -> Tool:
        """Add a source to a tool, creating the tool if needed."""
        tool = self.tools.get(tool_id)
        if tool is None:
            tool = Tool(
                tool_id=tool_id,
                display_name=display_name or tool_id,
            )
            self.tools[tool_id] = tool
        elif display_name:
            tool.display_name = display_name

        # Check if source already exists (by URL)
        for i, existing in enumerate(tool.sources):
            if existing.url == source.url:
                tool.sources[i] = source
                return tool

        tool.sources.append(source)
        return tool

    def remove_source(self, tool_id: str, url: str) -> bool:
        """Remove a source from a tool by URL."""
        tool = self.tools.get(tool_id)
        if tool is None:
            return False

        original_count = len(tool.sources)
        tool.sources = [s for s in tool.sources if s.url != url]
        return len(tool.sources) < original_count

    def replace_sources(
        self, tool_id: str, source: Source, display_name: str | None = None
    ) -> Tool:
        """Replace all sources for a tool with a single new source."""
        tool = Tool(
            tool_id=tool_id,
            display_name=display_name or tool_id,
            sources=[source],
        )
        self.tools[tool_id] = tool
        return tool

    def to_dict(self) -> dict:
        return {tool_id: tool.to_dict() for tool_id, tool in self.tools.items()}

    @classmethod
    def from_dict(cls, data: dict) -> Registry:
        tools = {tool_id: Tool.from_dict(tool_id, tool_data) for tool_id, tool_data in data.items()}
        return cls(tools=tools)


def load_registry(path: Path | None = None) -> Registry:
    """Load registry from file.

    If file doesn't exist, returns empty registry.
    """
    registry_path = path or get_sources_path()

    if not registry_path.exists():
        return Registry()

    with open(registry_path) as f:
        data = json.load(f)

    return Registry.from_dict(data)


def save_registry(registry: Registry, path: Path | None = None) -> None:
    """Save registry to file.

    Creates directories if needed.
    """
    ensure_directories()
    registry_path = path or get_sources_path()

    with open(registry_path, "w") as f:
        json.dump(registry.to_dict(), f, indent=2)
