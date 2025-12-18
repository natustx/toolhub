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

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

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
from toolhub.store.knowledge import KnowledgeStore, SourceStatus

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


# === Knowledge Store Models ===


class KnowledgeQueryRequest(BaseModel):
    """Request for knowledge store search."""

    query: str
    collection: str | None = None
    tags: list[str] | None = None
    type_key: str | None = None
    entity_id: str | None = None
    limit: int = 10
    min_similarity: float = 0.0


class KnowledgeSearchResultItem(BaseModel):
    """A single search result from knowledge store."""

    chunk_id: str
    source_id: str
    content: str
    heading: str | None
    heading_path: str | None
    source_file: str | None
    is_code: bool
    similarity: float
    canonical_url: str
    collection: str


class KnowledgeSearchResponse(BaseModel):
    """Response from knowledge store search."""

    query: str
    collection: str | None
    result_count: int
    total_chunks_searched: int
    results: list[KnowledgeSearchResultItem]
    timings: dict[str, float] = Field(default_factory=dict)


class EntityTypeRequest(BaseModel):
    """Request to register an entity type."""

    type_key: str
    json_schema: dict
    description: str | None = None


class EntityTypeResponse(BaseModel):
    """Response for entity type operations."""

    id: str
    type_key: str
    json_schema: dict
    schema_version: int
    description: str | None
    created_at: str
    updated_at: str


class EntityRequest(BaseModel):
    """Request to create or update an entity."""

    type_key: str
    name: str
    profile: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    collection: str = "default"


class EntityUpdateRequest(BaseModel):
    """Request to update an entity."""

    profile: dict | None = None
    tags: list[str] | None = None


class EntityResponse(BaseModel):
    """Response for entity operations."""

    id: str
    type_key: str
    name: str
    profile: dict
    tags: list[str]
    collection: str
    created_at: str
    updated_at: str


class EntityListResponse(BaseModel):
    """Response for entity list operations."""

    entities: list[EntityResponse]
    count: int


class EvidenceRequest(BaseModel):
    """Request to add evidence."""

    entity_id: str
    field_path: str
    chunk_id: str | None = None
    source_id: str | None = None
    quote: str | None = None
    locator: str | None = None
    confidence: float | None = None


class EvidenceResponse(BaseModel):
    """Response for evidence operations."""

    id: str
    entity_id: str
    field_path: str
    chunk_id: str | None
    source_id: str | None
    quote: str | None
    locator: str | None
    confidence: float | None
    created_at: str


class EvidenceListResponse(BaseModel):
    """Response for evidence list operations."""

    evidence: list[EvidenceResponse]
    count: int


class CitationResponse(BaseModel):
    """Response for citation data."""

    field_path: str
    quote: str | None
    source_url: str
    source_file: str | None
    heading_path: str | None
    chunk_content: str | None
    confidence: float | None


class EntityWithCitationsResponse(BaseModel):
    """Response for entity with citations."""

    entity: EntityResponse
    citations: dict[str, list[CitationResponse]]


class ReportRequest(BaseModel):
    """Request to generate a report."""

    entity_ids: list[str] | None = None
    tags: list[str] | None = None
    collection: str | None = None


class ReportResponse(BaseModel):
    """Response for report generation."""

    title: str
    report_type: str
    generated_at: str
    entities_count: int
    has_citations: bool
    metadata: dict = Field(default_factory=dict)
    data: dict
    markdown: str | None = None


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

# API router with version prefix - all endpoints except /health
api_router = APIRouter(prefix="/api/v1")


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


@api_router.post("/tools/add", response_model=AddToolResponse)
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

    # Store in KnowledgeStore
    ks = _get_knowledge_store()
    try:
        # Create or update source record (upsert on canonical_url + collection)
        ks_source = ks.add_source(
            canonical_url=url,
            source_type=source_type,
            collection=tool_id,
            tags=[tool_id],
            status=SourceStatus.INDEXED,
        )

        # Always delete existing chunks before adding new ones
        # (add_source does upsert, so we may be re-indexing an existing source)
        ks.delete_chunks(ks_source.id)

        # Convert embedded chunks to batch format
        chunk_data = [
            {
                "content": ec.content,
                "heading": ec.heading,
                "heading_path": ec.heading_path,
                "source_file": ec.source_file,
                "is_code": ec.is_code,
                "embedding": list(ec.embedding),
            }
            for ec in embedded
        ]
        ks.add_chunks_batch(ks_source, chunk_data, model_id=config.embedding.model)
    finally:
        ks.close()

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


@api_router.post("/tools/query", response_model=SearchResponse)
async def query_tools(request: SearchRequest):
    """Search indexed documentation using KnowledgeStore."""
    timings: dict[str, float] = {}
    config = load_config()

    # Map tool_ids to collection(s) with OR semantics
    if request.tool_ids and len(request.tool_ids) == 1:
        collection = request.tool_ids[0]
        collections = None
    elif request.tool_ids:
        collection = None
        collections = request.tool_ids  # OR semantics: search in any of these
    else:
        collection = None
        collections = None

    ks = _get_knowledge_store()
    try:
        response = ks.search_text(
            query=request.query,
            collection=collection,
            collections=collections,
            limit=request.limit,
            model_name=config.embedding.model,
            timings=timings,
        )
    finally:
        ks.close()

    # Convert to SearchResponse format
    tools_searched = request.tool_ids if request.tool_ids else ["all"]
    return SearchResponse(
        query=response.query,
        tools_searched=tools_searched,
        result_count=len(response.results),
        results=[
            SearchResultItem(
                tool_id=r.collection,  # collection acts as tool_id
                content=r.content,
                source_file=r.source_file or "",
                heading=r.heading or "",
                heading_path=r.heading_path or "",
                is_code=r.is_code,
                similarity=r.similarity,
            )
            for r in response.results
        ],
        timings=timings,
    )


@api_router.get("/tools", response_model=ToolListResponse)
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


@api_router.get("/tools/{tool_id}", response_model=ToolInfo)
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


@api_router.delete("/tools/{tool_id}", response_model=MessageResponse)
async def remove_tool(tool_id: str):
    """Remove an indexed tool."""
    registry = load_registry()
    tool = registry.get_tool(tool_id)

    if not tool:
        raise HTTPException(404, f"Tool not found: {tool_id}")

    # Remove from registry
    registry.remove_tool(tool_id)
    save_registry(registry)

    # Remove from KnowledgeStore (sources and their chunks cascade)
    ks = _get_knowledge_store()
    try:
        sources = ks.list_sources(collection=tool_id)
        for src in sources:
            ks.delete_source(src.id)
    finally:
        ks.close()

    # Remove cache
    import shutil

    cache_dir = get_tool_cache_dir(tool_id)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    return MessageResponse(message=f"Removed {tool_id}")


# === Knowledge Store API ===


def _get_knowledge_store():
    """Get or create KnowledgeStore instance."""
    config = load_config()
    return KnowledgeStore(config)


def _entity_to_response(entity) -> EntityResponse:
    """Convert Entity to EntityResponse."""
    return EntityResponse(
        id=str(entity.id),
        type_key=entity.type_key,
        name=entity.name,
        profile=entity.profile,
        tags=entity.tags,
        collection=entity.collection,
        created_at=entity.created_at.isoformat(),
        updated_at=entity.updated_at.isoformat(),
    )


def _evidence_to_response(evidence) -> EvidenceResponse:
    """Convert Evidence to EvidenceResponse."""
    return EvidenceResponse(
        id=str(evidence.id),
        entity_id=str(evidence.entity_id),
        field_path=evidence.field_path,
        chunk_id=str(evidence.chunk_id) if evidence.chunk_id else None,
        source_id=str(evidence.source_id) if evidence.source_id else None,
        quote=evidence.quote,
        locator=evidence.locator,
        confidence=evidence.confidence,
        created_at=evidence.created_at.isoformat(),
    )


def _citation_to_response(citation) -> CitationResponse:
    """Convert Citation to CitationResponse."""
    return CitationResponse(
        field_path=citation.field_path,
        quote=citation.quote,
        source_url=citation.source_url,
        source_file=citation.source_file,
        heading_path=citation.heading_path,
        chunk_content=citation.chunk_content,
        confidence=citation.confidence,
    )


@api_router.post("/knowledge/query", response_model=KnowledgeSearchResponse)
async def knowledge_query(request: KnowledgeQueryRequest):
    """Search the knowledge store."""
    import time

    store = _get_knowledge_store()
    timings: dict[str, float] = {}

    try:
        t0 = time.perf_counter()
        config = load_config()

        response = store.search_text(
            query=request.query,
            collection=request.collection,
            tags=request.tags,
            limit=request.limit,
            min_similarity=request.min_similarity,
            model_name=config.embedding.model,
            timings=timings,
        )
        timings["total"] = time.perf_counter() - t0

        return KnowledgeSearchResponse(
            query=response.query,
            collection=response.collection,
            result_count=len(response.results),
            total_chunks_searched=response.total_chunks_searched,
            results=[
                KnowledgeSearchResultItem(
                    chunk_id=str(r.chunk_id),
                    source_id=str(r.source_id),
                    content=r.content,
                    heading=r.heading,
                    heading_path=r.heading_path,
                    source_file=r.source_file,
                    is_code=r.is_code,
                    similarity=r.similarity,
                    canonical_url=r.canonical_url,
                    collection=r.collection,
                )
                for r in response.results
            ],
            timings=timings,
        )
    finally:
        store.close()


# === Entity Type Endpoints ===


@api_router.post("/entity-types", response_model=EntityTypeResponse)
async def register_entity_type(request: EntityTypeRequest):
    """Register or update an entity type."""
    store = _get_knowledge_store()
    try:
        entity_type = store.register_entity_type(
            type_key=request.type_key,
            json_schema=request.json_schema,
            description=request.description,
        )
        return EntityTypeResponse(
            id=str(entity_type.id),
            type_key=entity_type.type_key,
            json_schema=entity_type.json_schema,
            schema_version=entity_type.schema_version,
            description=entity_type.description,
            created_at=entity_type.created_at.isoformat(),
            updated_at=entity_type.updated_at.isoformat(),
        )
    finally:
        store.close()


@api_router.get("/entity-types", response_model=list[EntityTypeResponse])
async def list_entity_types():
    """List all registered entity types."""
    store = _get_knowledge_store()
    try:
        types = store.list_entity_types()
        return [
            EntityTypeResponse(
                id=str(t.id),
                type_key=t.type_key,
                json_schema=t.json_schema,
                schema_version=t.schema_version,
                description=t.description,
                created_at=t.created_at.isoformat(),
                updated_at=t.updated_at.isoformat(),
            )
            for t in types
        ]
    finally:
        store.close()


@api_router.get("/entity-types/{type_key}", response_model=EntityTypeResponse)
async def get_entity_type(type_key: str):
    """Get a specific entity type."""
    store = _get_knowledge_store()
    try:
        entity_type = store.get_entity_type(type_key)
        if not entity_type:
            raise HTTPException(404, f"Entity type not found: {type_key}")

        return EntityTypeResponse(
            id=str(entity_type.id),
            type_key=entity_type.type_key,
            json_schema=entity_type.json_schema,
            schema_version=entity_type.schema_version,
            description=entity_type.description,
            created_at=entity_type.created_at.isoformat(),
            updated_at=entity_type.updated_at.isoformat(),
        )
    finally:
        store.close()


# === Entity Endpoints ===


@api_router.post("/entities", response_model=EntityResponse)
async def create_entity(request: EntityRequest):
    """Create a new entity."""
    from toolhub.store.knowledge import EntityValidationError

    store = _get_knowledge_store()
    try:
        entity = store.create_entity(
            type_key=request.type_key,
            name=request.name,
            profile=request.profile,
            tags=request.tags,
            collection=request.collection,
        )
        return _entity_to_response(entity)
    except KeyError as e:
        raise HTTPException(400, str(e))
    except EntityValidationError as e:
        raise HTTPException(422, str(e))
    finally:
        store.close()


@api_router.get("/entities", response_model=EntityListResponse)
async def list_entities(
    type_key: str | None = None,
    collection: str | None = None,
    tags: str | None = None,
):
    """List entities with optional filters."""
    store = _get_knowledge_store()
    try:
        tag_list = tags.split(",") if tags else None
        entities = store.list_entities(
            type_key=type_key,
            collection=collection,
            tags=tag_list,
        )
        return EntityListResponse(
            entities=[_entity_to_response(e) for e in entities],
            count=len(entities),
        )
    finally:
        store.close()


@api_router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(entity_id: str):
    """Get a specific entity by ID."""
    import uuid

    store = _get_knowledge_store()
    try:
        entity = store.get_entity(uuid.UUID(entity_id))
        if not entity:
            raise HTTPException(404, f"Entity not found: {entity_id}")
        return _entity_to_response(entity)
    except ValueError:
        raise HTTPException(400, f"Invalid entity ID: {entity_id}")
    finally:
        store.close()


@api_router.get("/entities/{entity_id}/citations", response_model=EntityWithCitationsResponse)
async def get_entity_with_citations(entity_id: str):
    """Get an entity with all its citations."""
    import uuid

    store = _get_knowledge_store()
    try:
        entity, citations = store.get_entity_with_citations(uuid.UUID(entity_id))
        if not entity:
            raise HTTPException(404, f"Entity not found: {entity_id}")

        citations_response = {
            field: [_citation_to_response(c) for c in cites] for field, cites in citations.items()
        }

        return EntityWithCitationsResponse(
            entity=_entity_to_response(entity),
            citations=citations_response,
        )
    except ValueError:
        raise HTTPException(400, f"Invalid entity ID: {entity_id}")
    finally:
        store.close()


@api_router.put("/entities/{entity_id}", response_model=EntityResponse)
async def update_entity(entity_id: str, request: EntityUpdateRequest):
    """Update an existing entity."""
    import uuid

    from toolhub.store.knowledge import EntityValidationError

    store = _get_knowledge_store()
    try:
        entity = store.update_entity(
            entity_id=uuid.UUID(entity_id),
            profile=request.profile,
            tags=request.tags,
        )
        return _entity_to_response(entity)
    except KeyError:
        raise HTTPException(404, f"Entity not found: {entity_id}")
    except ValueError:
        raise HTTPException(400, f"Invalid entity ID: {entity_id}")
    except EntityValidationError as e:
        raise HTTPException(422, str(e))
    finally:
        store.close()


@api_router.delete("/entities/{entity_id}", response_model=MessageResponse)
async def delete_entity(entity_id: str):
    """Delete an entity."""
    import uuid

    store = _get_knowledge_store()
    try:
        deleted = store.delete_entity(uuid.UUID(entity_id))
        if not deleted:
            raise HTTPException(404, f"Entity not found: {entity_id}")
        return MessageResponse(message=f"Deleted entity {entity_id}")
    except ValueError:
        raise HTTPException(400, f"Invalid entity ID: {entity_id}")
    finally:
        store.close()


# === Evidence Endpoints ===


@api_router.post("/evidence", response_model=EvidenceResponse)
async def add_evidence(request: EvidenceRequest):
    """Add evidence for an entity field."""
    import uuid

    store = _get_knowledge_store()
    try:
        evidence = store.add_evidence(
            entity_id=uuid.UUID(request.entity_id),
            field_path=request.field_path,
            chunk_id=uuid.UUID(request.chunk_id) if request.chunk_id else None,
            source_id=uuid.UUID(request.source_id) if request.source_id else None,
            quote=request.quote,
            locator=request.locator,
            confidence=request.confidence,
        )
        return _evidence_to_response(evidence)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        store.close()


@api_router.get("/evidence", response_model=EvidenceListResponse)
async def list_evidence(
    entity_id: str | None = None,
    field_path: str | None = None,
):
    """List evidence with optional filters."""
    import uuid

    store = _get_knowledge_store()
    try:
        entity_uuid = uuid.UUID(entity_id) if entity_id else None
        evidence_list = store.list_evidence(
            entity_id=entity_uuid,
            field_path=field_path,
        )
        return EvidenceListResponse(
            evidence=[_evidence_to_response(e) for e in evidence_list],
            count=len(evidence_list),
        )
    except ValueError:
        raise HTTPException(400, f"Invalid entity ID: {entity_id}")
    finally:
        store.close()


@api_router.get("/evidence/{evidence_id}", response_model=EvidenceResponse)
async def get_evidence(evidence_id: str):
    """Get a specific evidence by ID."""
    import uuid

    store = _get_knowledge_store()
    try:
        evidence = store.get_evidence(uuid.UUID(evidence_id))
        if not evidence:
            raise HTTPException(404, f"Evidence not found: {evidence_id}")
        return _evidence_to_response(evidence)
    except ValueError:
        raise HTTPException(400, f"Invalid evidence ID: {evidence_id}")
    finally:
        store.close()


@api_router.delete("/evidence/{evidence_id}", response_model=MessageResponse)
async def delete_evidence(evidence_id: str):
    """Delete evidence."""
    import uuid

    store = _get_knowledge_store()
    try:
        deleted = store.delete_evidence(uuid.UUID(evidence_id))
        if not deleted:
            raise HTTPException(404, f"Evidence not found: {evidence_id}")
        return MessageResponse(message=f"Deleted evidence {evidence_id}")
    except ValueError:
        raise HTTPException(400, f"Invalid evidence ID: {evidence_id}")
    finally:
        store.close()


# === Report Endpoints ===


@api_router.get("/reports", response_model=list[str])
async def list_report_types():
    """List available report types."""
    from toolhub.store.reports import REPORT_TYPES

    return list(REPORT_TYPES.keys())


@api_router.post("/reports/{report_type}", response_model=ReportResponse)
async def generate_report(report_type: str, request: ReportRequest):
    """Generate a report by type."""
    from toolhub.store.reports import get_report

    store = _get_knowledge_store()
    try:
        report = get_report(report_type, store)
        result = report.generate(
            entity_ids=request.entity_ids,
            tags=request.tags,
            collection=request.collection,
        )

        # Get markdown if the report has a custom to_markdown method
        markdown = None
        if hasattr(report, "to_markdown"):
            markdown = report.to_markdown(result)
        else:
            markdown = result.to_markdown()

        return ReportResponse(
            title=result.title,
            report_type=result.report_type,
            generated_at=result.generated_at.isoformat(),
            entities_count=result.entities_count,
            has_citations=result.has_citations,
            metadata=result.metadata,
            data=result.data,
            markdown=markdown,
        )
    except KeyError as e:
        raise HTTPException(400, str(e))
    finally:
        store.close()


@api_router.post("/stop", response_model=MessageResponse)
async def stop_daemon():
    """Stop the daemon gracefully."""
    logger.info("Stop requested, shutting down...")

    def shutdown():
        os.kill(os.getpid(), signal.SIGTERM)

    # Schedule shutdown after response is sent
    import asyncio

    asyncio.get_event_loop().call_later(0.5, shutdown)

    return MessageResponse(message="Shutting down")


# Include the versioned API router
app.include_router(api_router)


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
