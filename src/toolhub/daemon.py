"""FastAPI daemon for toolhub.

Provides HTTP API for tool management and search.
Keeps embeddings warm for fast subsequent queries.
"""

from __future__ import annotations

import logging
import os
import signal
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from toolhub import __version__
from toolhub.config import load_config
from toolhub.crawler import (
    GitHubCrawler,
    LlmsTxtCrawler,
    WebsiteCrawler,
    detect_source_type,
)
from toolhub.indexer import chunk_directory, embed_chunks
from toolhub.paths import (
    ensure_directories,
    get_pid_path,
    get_source_cache_dir,
    get_tool_cache_dir,
)
from toolhub.registry import Source, load_registry, save_registry
from toolhub.store import VectorStore, search
from toolhub.store.lance import delete_tool_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class AddToolRequest(BaseModel):
    url: str
    name: str | None = None
    replace: bool = False


class AddToolResponse(BaseModel):
    tool_id: str
    chunks: int
    message: str


class SearchRequest(BaseModel):
    query: str
    tool_ids: list[str] | None = None
    limit: int = 5


class SearchResultItem(BaseModel):
    tool_id: str
    content: str
    source_file: str
    heading: str
    heading_path: str
    is_code: bool
    similarity: float


class SearchResponse(BaseModel):
    query: str
    tools_searched: list[str]
    result_count: int
    results: list[SearchResultItem]
    timings: dict[str, float] | None = None


class ToolInfo(BaseModel):
    tool_id: str
    display_name: str
    sources: list[dict]
    total_chunks: int


class ToolListResponse(BaseModel):
    tools: list[ToolInfo]


class HealthResponse(BaseModel):
    status: str
    version: str
    tools: int
    chunks: int


class MessageResponse(BaseModel):
    message: str


def write_pid_file() -> None:
    """Write the daemon PID to file."""
    pid_path = get_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def remove_pid_file() -> None:
    """Remove the daemon PID file."""
    pid_path = get_pid_path()
    if pid_path.exists():
        pid_path.unlink()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    ensure_directories()
    write_pid_file()
    logger.info(f"toolhubd {__version__} started (PID: {os.getpid()})")
    yield
    remove_pid_file()
    logger.info("toolhubd stopped")


app = FastAPI(
    title="toolhubd",
    description="Local documentation index daemon for AI coding agents",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    registry = load_registry()
    total_chunks = sum(t.total_chunks for t in registry.tools.values())
    return HealthResponse(
        status="ok",
        version=__version__,
        tools=len(registry.tools),
        chunks=total_chunks,
    )


@app.post("/tools/add", response_model=AddToolResponse)
async def add_tool(request: AddToolRequest):
    """Add and index a documentation source."""
    url = request.url
    name = request.name
    replace = request.replace

    # Detect source type
    source_type = detect_source_type(url)
    if not source_type:
        raise HTTPException(400, f"Could not detect source type for URL: {url}")

    if source_type not in ("github", "llmstxt", "website"):
        raise HTTPException(400, f"Source type '{source_type}' not yet supported")

    # Infer tool name
    if not name:
        if source_type == "github":
            from toolhub.crawler.github import parse_github_url

            try:
                _, repo, _ = parse_github_url(url)
                name = repo
            except ValueError:
                raise HTTPException(400, "Could not parse GitHub URL")
        elif source_type in ("llmstxt", "website"):
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain_parts = parsed.netloc.split(".")
            if domain_parts[0] in ("www", "docs", "api"):
                name = domain_parts[1] if len(domain_parts) > 1 else domain_parts[0]
            else:
                name = domain_parts[0]
        else:
            raise HTTPException(400, "Please provide name for this source type")

    tool_id = name.lower().replace(" ", "-")

    # Get crawler
    config = load_config()
    if source_type == "github":
        crawler = GitHubCrawler(github_token=config.crawling.github_token or None)
    elif source_type == "llmstxt":
        crawler = LlmsTxtCrawler()
    elif source_type == "website":
        crawler = WebsiteCrawler()
    else:
        raise HTTPException(400, f"Unsupported source type: {source_type}")

    # Crawl to per-source cache directory
    cache_dir = get_source_cache_dir(tool_id, url)
    result = crawler.crawl(url, cache_dir)

    if not result.success:
        raise HTTPException(500, f"Crawl failed: {result.error}")

    file_count = len(result.files_crawled)

    # Chunk
    chunks = chunk_directory(cache_dir, max_tokens=config.embedding.chunk_size_tokens)
    if not chunks:
        raise HTTPException(400, "No content to index")

    # Embed
    embedded = embed_chunks(chunks, model_name=config.embedding.model)

    # Store
    store = VectorStore(tool_id)
    if replace:
        store.clear()
    store.add_chunks(embedded)

    # Update registry
    registry = load_registry()
    source = Source(
        url=url,
        source_type=source_type,
        indexed_at=datetime.now(),
        chunk_count=len(embedded),
        file_count=file_count,
    )

    if replace:
        registry.replace_sources(tool_id, source, display_name=name)
    else:
        registry.add_source(tool_id, source, display_name=name)

    save_registry(registry)

    return AddToolResponse(
        tool_id=tool_id,
        chunks=len(embedded),
        message=f"Indexed {tool_id}: {file_count} files ({len(embedded)} chunks)",
    )


@app.post("/tools/query", response_model=SearchResponse)
async def query_tools(request: SearchRequest):
    """Search indexed documentation."""
    timings: dict[str, float] = {}
    config = load_config()
    response = search(
        request.query,
        tool_ids=request.tool_ids,
        limit=request.limit,
        model_name=config.embedding.model,
        timings=timings,
    )

    return SearchResponse(
        query=response.query,
        tools_searched=response.tools_searched,
        result_count=len(response.results),
        results=[
            SearchResultItem(
                tool_id=r.tool_id,
                content=r.content,
                source_file=r.source_file,
                heading=r.heading,
                heading_path=r.heading_path,
                is_code=r.is_code,
                similarity=r.similarity,
            )
            for r in response.results
        ],
        timings=timings,
    )


@app.get("/tools", response_model=ToolListResponse)
async def list_tools():
    """List all indexed tools."""
    registry = load_registry()

    tools = []
    for tool_id, tool in registry.tools.items():
        tools.append(
            ToolInfo(
                tool_id=tool.tool_id,
                display_name=tool.display_name,
                sources=[s.to_dict() for s in tool.sources],
                total_chunks=tool.total_chunks,
            )
        )

    return ToolListResponse(tools=tools)


@app.get("/tools/{tool_id}", response_model=ToolInfo)
async def get_tool(tool_id: str):
    """Get details for a specific tool."""
    registry = load_registry()
    tool = registry.get_tool(tool_id)

    if not tool:
        raise HTTPException(404, f"Tool not found: {tool_id}")

    return ToolInfo(
        tool_id=tool.tool_id,
        display_name=tool.display_name,
        sources=[s.to_dict() for s in tool.sources],
        total_chunks=tool.total_chunks,
    )


@app.delete("/tools/{tool_id}", response_model=MessageResponse)
async def remove_tool(tool_id: str):
    """Remove an indexed tool."""
    registry = load_registry()
    tool = registry.get_tool(tool_id)

    if not tool:
        raise HTTPException(404, f"Tool not found: {tool_id}")

    # Remove from registry
    registry.remove_tool(tool_id)
    save_registry(registry)

    # Remove vector store
    delete_tool_store(tool_id)

    # Remove cache
    import shutil

    cache_dir = get_tool_cache_dir(tool_id)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    return MessageResponse(message=f"Removed {tool_id}")


@app.post("/stop", response_model=MessageResponse)
async def stop_daemon():
    """Stop the daemon gracefully."""
    logger.info("Stop requested, shutting down...")

    def shutdown():
        os.kill(os.getpid(), signal.SIGTERM)

    # Schedule shutdown after response is sent
    import asyncio

    asyncio.get_event_loop().call_later(0.5, shutdown)

    return MessageResponse(message="Shutting down")


def main():
    """Entry point for toolhubd."""
    import uvicorn

    config = load_config()
    uvicorn.run(
        app,
        host=config.daemon.host,
        port=config.daemon.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
