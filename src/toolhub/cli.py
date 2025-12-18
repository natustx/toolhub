"""Command-line interface for toolhub."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from toolhub import __version__
from toolhub.client import (
    DaemonClient,
    is_daemon_running,
    start_daemon,
    stop_daemon,
)
from toolhub.config import load_config
from toolhub.crawler import (
    GitHubCrawler,
    LlmsTxtCrawler,
    WebsiteCrawler,
    detect_source_type,
)
from toolhub.indexer import chunk_directory, embed_chunks
from toolhub.indexer.openapi import parse_openapi_files
from toolhub.paths import ensure_directories, get_source_cache_dir, get_tool_cache_dir
from toolhub.registry import Source, load_registry, save_registry
from toolhub.store import OutputFormat
from toolhub.store.knowledge import KnowledgeStore, SourceStatus
from toolhub.store.operations import OperationsStore

app = typer.Typer(
    name="toolhub",
    help="Local documentation index for AI coding agents.",
    no_args_is_help=True,
)

daemon_app = typer.Typer(help="Manage the toolhub daemon")
app.add_typer(daemon_app, name="daemon")

console = Console()


def _get_crawler(source_type: str, config=None, max_pages: int | None = None):
    """Get appropriate crawler for source type."""
    if config is None:
        config = load_config()

    if source_type == "github":
        return GitHubCrawler(github_token=config.crawling.github_token or None)

    if source_type == "llmstxt":
        return LlmsTxtCrawler()

    if source_type == "website":
        from toolhub.crawler.website import CrawlConfig

        crawl_config = CrawlConfig(max_pages=max_pages) if max_pages else None
        return WebsiteCrawler(config=crawl_config)

    raise typer.BadParameter(f"Unsupported source type: {source_type}")


# === Daemon commands ===


@daemon_app.command(name="status")
def daemon_status() -> None:
    """Check if the daemon is running."""
    from toolhub.paths import get_pid_path

    if is_daemon_running():
        pid_path = get_pid_path()
        pid = pid_path.read_text().strip() if pid_path.exists() else "unknown"

        # Get health info from daemon
        try:
            client = DaemonClient()
            health = client.health()
            client.close()
            console.print(f"[green]Daemon is running[/green] (PID: {pid})")
            console.print(f"  Tools: {health['tools']}, Chunks: {health['chunks']}")
        except Exception:
            console.print(f"[green]Daemon is running[/green] (PID: {pid})")
    else:
        console.print("[dim]Daemon is not running[/dim]")


@daemon_app.command(name="start")
def daemon_start() -> None:
    """Start the daemon."""
    if is_daemon_running():
        console.print("[yellow]Daemon is already running[/yellow]")
        raise typer.Exit(0)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Starting daemon...", total=None)
        success = start_daemon()

    if success:
        console.print("[green]✓ Daemon started[/green]")
    else:
        console.print("[red]Failed to start daemon[/red]")
        raise typer.Exit(1)


@daemon_app.command(name="stop")
def daemon_stop() -> None:
    """Stop the daemon."""
    if not is_daemon_running():
        console.print("[dim]Daemon is not running[/dim]")
        raise typer.Exit(0)

    success = stop_daemon()
    if success:
        console.print("[green]✓ Daemon stopped[/green]")
    else:
        console.print("[red]Failed to stop daemon[/red]")
        raise typer.Exit(1)


# === Tool commands ===


@app.command()
def add(
    name: Annotated[str, typer.Argument(help="Tool name (e.g., 'bittensor', 'fastapi')")],
    url: Annotated[str, typer.Argument(help="URL to documentation source")],
    replace: Annotated[
        bool,
        typer.Option("--replace", "-r", help="Replace existing sources for this tool"),
    ] = False,
    max_pages: Annotated[
        int | None,
        typer.Option("--max-pages", "-m", help="Maximum pages to crawl (website sources only)"),
    ] = None,
) -> None:
    """Add a documentation source and index it.

    Examples:
        toolhub add bittensor https://github.com/opentensor/bittensor
        toolhub add bittensor https://github.com/opentensor/btcli
        toolhub add fastapi https://github.com/fastapi/fastapi
        toolhub add docs https://docs.example.com --max-pages 500
    """
    ensure_directories()

    # Detect source type
    source_type = detect_source_type(url)
    if not source_type:
        console.print(f"[red]Could not detect source type for URL: {url}[/red]")
        raise typer.Exit(1)

    if source_type not in ("github", "llmstxt", "website"):
        console.print(f"[red]Source type '{source_type}' not yet supported[/red]")
        raise typer.Exit(1)

    tool_id = name.lower().replace(" ", "-")

    console.print(f"[bold]Adding {tool_id}[/bold] from {url}")

    # Get crawler
    try:
        crawler = _get_crawler(source_type, max_pages=max_pages)
    except typer.BadParameter as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Crawl to per-source cache directory
        task = progress.add_task("Crawling documentation...", total=None)
        cache_dir = get_source_cache_dir(tool_id, url)

        # Progress callback for website crawler
        def on_crawl_progress(pages_crawled: int, max_pages: int) -> None:
            progress.update(task, description=f"Crawling... {pages_crawled}/{max_pages} pages")

        # Pass callback if crawler supports it (website crawler)
        if source_type == "website":
            result = crawler.crawl(url, cache_dir, on_progress=on_crawl_progress)
        else:
            result = crawler.crawl(url, cache_dir)

        if not result.success:
            progress.stop()
            console.print(f"[red]Crawl failed: {result.error}[/red]")
            raise typer.Exit(1)

        file_count = len(result.files_crawled)
        progress.update(task, description=f"Found {file_count} files")

        # Parse OpenAPI specs
        progress.update(task, description="Checking for OpenAPI specs...")
        operations = parse_openapi_files(cache_dir, tool_id)
        if operations:
            ops_store = OperationsStore()
            if replace:
                ops_store.delete_tool_operations(tool_id)
            ops_store.add_operations(operations)
            progress.update(task, description=f"Indexed {len(operations)} API operations")

        # Chunk
        progress.update(task, description="Chunking documents...")
        config = load_config()
        chunks = chunk_directory(cache_dir, max_tokens=config.embedding.chunk_size_tokens)

        if not chunks:
            progress.stop()
            console.print("[yellow]No content to index[/yellow]")
            raise typer.Exit(1)

        progress.update(task, description=f"Created {len(chunks)} chunks")

        # Embed (this is the slow part)
        progress.update(task, description="Generating embeddings...")
        embedded = embed_chunks(chunks, model_name=config.embedding.model)

        # Store in KnowledgeStore
        progress.update(task, description="Indexing in knowledge store...")
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

    console.print(
        f"[green]✓ Indexed {tool_id}: {file_count} files ({len(embedded)} chunks)[/green]"
    )


def _search_via_daemon(
    query: str,
    tool_ids: list[str] | None,
    limit: int,
    timings: dict | None = None,
) -> dict | None:
    """Try to search via daemon, returning None on failure."""
    import time

    try:
        # Auto-start daemon if not running
        t0 = time.perf_counter()
        daemon_was_running = is_daemon_running()
        if timings is not None:
            timings["daemon_check"] = time.perf_counter() - t0

        if not daemon_was_running:
            t0 = time.perf_counter()
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Starting daemon...", total=None)
                if not start_daemon():
                    return None
            if timings is not None:
                timings["daemon_start"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        client = DaemonClient()
        result = client.search(query, tool_ids=tool_ids, limit=limit)
        client.close()
        if timings is not None:
            timings["daemon_roundtrip"] = time.perf_counter() - t0
            # Copy server-side timings if present
            if result.get("timings"):
                for k, v in result["timings"].items():
                    timings[f"server_{k}"] = v
            timings["via"] = "daemon"

        return result
    except Exception:
        return None


def _search_direct(
    query: str,
    tool_ids: list[str] | None,
    limit: int,
    timings: dict | None = None,
) -> dict:
    """Search directly without daemon using KnowledgeStore."""
    import time

    t0 = time.perf_counter()
    config = load_config()
    if timings is not None:
        timings["config_load"] = time.perf_counter() - t0

    # Use KnowledgeStore for search
    # Map tool_ids to collection(s) with OR semantics
    if tool_ids and len(tool_ids) == 1:
        collection = tool_ids[0]
        collections = None
    elif tool_ids:
        collection = None
        collections = tool_ids  # OR semantics: search in any of these
    else:
        collection = None
        collections = None

    ks = _get_knowledge_store()
    try:
        response = ks.search_text(
            query=query,
            collection=collection,
            collections=collections,
            limit=limit,
            model_name=config.embedding.model,
            timings=timings,
        )
    finally:
        ks.close()

    if timings is not None:
        timings["via"] = "direct"

    # Convert to old response format for compatibility
    tools_searched = tool_ids if tool_ids else ["all"]
    return {
        "query": response.query,
        "tools_searched": tools_searched,
        "result_count": len(response.results),
        "results": [
            {
                "tool_id": r.collection,  # collection acts as tool_id
                "content": r.content,
                "source_file": r.source_file or "",
                "heading": r.heading or "",
                "heading_path": r.heading_path or "",
                "is_code": r.is_code,
                "similarity": r.similarity,
            }
            for r in response.results
        ],
    }


@app.command(name="search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Search query")],
    tool: Annotated[
        str | None,
        typer.Option("--tool", "-t", help="Limit search to specific tool"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum number of results"),
    ] = 5,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or markdown"),
    ] = "markdown",
    no_daemon: Annotated[
        bool,
        typer.Option("--no-daemon", help="Run directly without using daemon"),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option("--profile", help="Show timing breakdown"),
    ] = False,
) -> None:
    """Search indexed documentation."""
    import time

    total_start = time.perf_counter()
    timings: dict = {} if profile else None  # type: ignore

    tool_ids = [tool] if tool else None

    try:
        output_format = OutputFormat(format.lower())
    except ValueError:
        console.print(f"[red]Invalid format: {format}. Use 'json' or 'markdown'[/red]")
        raise typer.Exit(1)

    # Try daemon first (unless --no-daemon)
    result = None
    if not no_daemon:
        result = _search_via_daemon(query, tool_ids, limit, timings)

    # Fall back to direct search
    if result is None:
        result = _search_direct(query, tool_ids, limit, timings)

    if not result["results"]:
        console.print("[yellow]No results found[/yellow]")
        raise typer.Exit(0)

    # Format output
    t0 = time.perf_counter()
    if output_format == OutputFormat.JSON:
        import json

        console.print(json.dumps(result, indent=2))
    else:
        # Markdown format
        lines = []
        lines.append(f"## Search: {result['query']}")
        lines.append(f"*Searched: {', '.join(result['tools_searched']) or 'all tools'}*")
        lines.append(f"*Found: {result['result_count']} results*")
        lines.append("")

        for r in result["results"]:
            heading_info = f" > {r['heading_path']}" if r.get("heading_path") else ""
            lines.append(f"### {r['tool_id']}: {r['source_file']}{heading_info}")
            lines.append(f"*Similarity: {r['similarity']:.2%}*")
            lines.append("")
            content = r["content"]
            if r.get("is_code") and not content.startswith("```"):
                lines.append("```")
                lines.append(content)
                lines.append("```")
            else:
                lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

        console.print("\n".join(lines))

    if profile:
        timings["format_output"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - total_start

        console.print("\n[dim]─── Profile ───[/dim]")
        console.print(f"[dim]Via: {timings.get('via', 'unknown')}[/dim]")
        for key, val in timings.items():
            if key != "via" and isinstance(val, float):
                console.print(f"[dim]  {key}: {val * 1000:.1f}ms[/dim]")


@app.command(name="list")
def list_cmd() -> None:
    """List all indexed tools."""
    registry = load_registry()

    if not registry.tools:
        console.print("[yellow]No tools indexed yet. Use 'toolhub add' to add one.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Indexed Tools")
    table.add_column("Tool", style="cyan")
    table.add_column("Sources", justify="right")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Last Indexed")

    for tool_id, tool in sorted(registry.tools.items()):
        source_count = len(tool.sources)
        file_count = tool.total_files
        chunk_count = tool.total_chunks

        # Get most recent index time
        last_indexed = None
        for source in tool.sources:
            if source.indexed_at:
                if last_indexed is None or source.indexed_at > last_indexed:
                    last_indexed = source.indexed_at

        last_str = last_indexed.strftime("%Y-%m-%d %H:%M") if last_indexed else "never"

        table.add_row(
            tool.display_name,
            str(source_count),
            str(file_count),
            str(chunk_count),
            last_str,
        )

    console.print(table)


@app.command()
def info(
    tool: Annotated[str, typer.Argument(help="Tool name to get info for")],
) -> None:
    """Show detailed information about an indexed tool."""
    registry = load_registry()
    tool_data = registry.get_tool(tool.lower())

    if not tool_data:
        console.print(f"[red]Tool not found: {tool}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]{tool_data.display_name}[/bold]")
    console.print(f"  ID: {tool_data.tool_id}")
    console.print(f"  Total: {tool_data.total_files} files ({tool_data.total_chunks} chunks)")
    console.print()

    if tool_data.sources:
        console.print("[bold]Sources:[/bold]")
        for source in tool_data.sources:
            console.print(f"  • {source.url}")
            console.print(f"    Type: {source.source_type}")
            console.print(f"    Size: {source.file_count} files ({source.chunk_count} chunks)")
            if source.indexed_at:
                console.print(f"    Indexed: {source.indexed_at.strftime('%Y-%m-%d %H:%M')}")
            console.print()


@app.command()
def remove(
    tool: Annotated[str, typer.Argument(help="Tool name to remove")],
    source: Annotated[
        str | None,
        typer.Option("--source", "-s", help="Remove only this source URL"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Remove an indexed tool or source."""
    registry = load_registry()
    tool_id = tool.lower()
    tool_data = registry.get_tool(tool_id)

    if not tool_data:
        console.print(f"[red]Tool not found: {tool}[/red]")
        raise typer.Exit(1)

    if source:
        # Remove single source
        if not force:
            confirm = typer.confirm(f"Remove source {source} from {tool_id}?")
            if not confirm:
                raise typer.Abort()

        removed = registry.remove_source(tool_id, source)
        if removed:
            save_registry(registry)
            console.print(f"[green]✓ Removed source from {tool_id}[/green]")
            console.print(
                "[yellow]Note: Vector store still contains chunks from this source[/yellow]"
            )
        else:
            console.print(f"[red]Source not found: {source}[/red]")
            raise typer.Exit(1)
    else:
        # Remove entire tool
        if not force:
            confirm = typer.confirm(f"Remove tool {tool_id} and all its data?")
            if not confirm:
                raise typer.Abort()

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

        # Remove operations
        ops_store = OperationsStore()
        ops_store.delete_tool_operations(tool_id)

        # Remove cache
        cache_dir = get_tool_cache_dir(tool_id)
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)

        console.print(f"[green]✓ Removed {tool_id}[/green]")


@app.command()
def status() -> None:
    """Show toolhub status."""
    from toolhub.paths import get_pid_path
    from toolhub.scheduler import get_stale_sources

    registry = load_registry()
    config = load_config()

    # Get KnowledgeStore stats
    ks = _get_knowledge_store()
    try:
        ks_sources = ks.list_sources()
        ks_collections = set(s.collection for s in ks_sources)
        total_ks_chunks = ks.get_chunk_count()
    finally:
        ks.close()

    # Build status table
    table = Table(title="toolhub status", show_header=False, box=None)
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Version", __version__)
    table.add_row("Tools indexed", str(len(registry.tools)))
    table.add_row("KnowledgeStore collections", str(len(ks_collections)))
    table.add_row("KnowledgeStore sources", str(len(ks_sources)))

    total_chunks = sum(t.total_chunks for t in registry.tools.values())
    table.add_row("Registry chunks", str(total_chunks))
    table.add_row("KnowledgeStore chunks", str(total_ks_chunks))

    # Check for stale sources
    stale = get_stale_sources(config.updates.max_age_hours)
    if stale:
        table.add_row("Stale sources", f"[yellow]{len(stale)}[/yellow]")
    else:
        table.add_row("Stale sources", "[green]0[/green]")

    # Auto-update status
    status_str = "[green]enabled[/green]" if config.updates.enabled else "[dim]disabled[/dim]"
    table.add_row("Auto-updates", status_str)

    # Daemon status
    if is_daemon_running():
        pid_path = get_pid_path()
        pid = pid_path.read_text().strip() if pid_path.exists() else "?"
        table.add_row("Daemon", f"[green]running[/green] (PID {pid})")
    else:
        table.add_row("Daemon", "[dim]not running[/dim]")

    console.print(table)


@app.command()
def update(
    tool: Annotated[
        str | None,
        typer.Argument(help="Tool name to update (default: all stale tools)"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Update even if not stale"),
    ] = False,
) -> None:
    """Update documentation for indexed tools."""
    from toolhub.scheduler import get_stale_sources, update_source

    config = load_config()
    registry = load_registry()

    if tool:
        # Update specific tool
        tool_id = tool.lower()
        tool_data = registry.get_tool(tool_id)
        if not tool_data:
            console.print(f"[red]Tool not found: {tool}[/red]")
            raise typer.Exit(1)

        if not tool_data.sources:
            console.print(f"[yellow]No sources to update for {tool}[/yellow]")
            raise typer.Exit(0)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Updating {tool_data.display_name}...", total=None)
            updated = 0
            for source in tool_data.sources:
                if update_source(
                    tool_data,
                    source,
                    on_progress=lambda msg: progress.update(task, description=msg),
                ):
                    updated += 1

        if updated:
            console.print(f"[green]✓ Updated {updated} source(s) for {tool}[/green]")
        else:
            console.print(f"[yellow]No sources updated for {tool}[/yellow]")
    else:
        # Update all stale sources
        if force:
            stale = [(t, s) for t in registry.tools.values() for s in t.sources]
        else:
            stale = get_stale_sources(config.updates.max_age_hours)

        if not stale:
            console.print("[green]All sources are up to date[/green]")
            raise typer.Exit(0)

        console.print(f"Found {len(stale)} source(s) to update")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Updating...", total=len(stale))
            updated = 0
            for tool_data, source in stale:
                progress.update(task, description=f"Updating {tool_data.display_name}...")
                if update_source(tool_data, source):
                    updated += 1
                progress.advance(task)

        console.print(f"[green]✓ Updated {updated}/{len(stale)} source(s)[/green]")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"toolhub {__version__}")


# === Knowledge Store CLI ===

entity_app = typer.Typer(help="Manage entities")
app.add_typer(entity_app, name="entity")

entity_type_app = typer.Typer(help="Manage entity types")
entity_app.add_typer(entity_type_app, name="type")

evidence_app = typer.Typer(help="Manage evidence")
app.add_typer(evidence_app, name="evidence")

report_app = typer.Typer(help="Generate reports")
app.add_typer(report_app, name="report")

source_app = typer.Typer(help="Manage documentation sources")
app.add_typer(source_app, name="source")

db_app = typer.Typer(help="Database management")
app.add_typer(db_app, name="db")


def _get_knowledge_store():
    """Get or create KnowledgeStore instance."""
    config = load_config()
    return KnowledgeStore(config)


# === Entity Commands ===


@entity_app.command(name="create")
def entity_create(
    type_key: Annotated[str, typer.Argument(help="Entity type (e.g., 'competitor')")],
    name: Annotated[str, typer.Argument(help="Entity name")],
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Profile as JSON string"),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option("--tags", "-t", help="Comma-separated tags"),
    ] = None,
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection name"),
    ] = "default",
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Create a new entity."""
    import json

    from toolhub.store.knowledge import EntityValidationError

    store = _get_knowledge_store()
    try:
        profile_dict = json.loads(profile) if profile else {}
        tag_list = [t.strip() for t in tags.split(",")] if tags else []

        entity = store.create_entity(
            type_key=type_key,
            name=name,
            profile=profile_dict,
            tags=tag_list,
            collection=collection,
        )

        if format == "json":
            console.print(json.dumps(entity.to_dict(), indent=2))
        else:
            console.print(f"[green]✓ Created entity: {entity.name}[/green]")
            console.print(f"  ID: {entity.id}")
            console.print(f"  Type: {entity.type_key}")
            console.print(f"  Collection: {entity.collection}")
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except EntityValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in profile: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@entity_app.command(name="list")
def entity_list(
    type_key: Annotated[
        str | None,
        typer.Option("--type", "-T", help="Filter by entity type"),
    ] = None,
    collection: Annotated[
        str | None,
        typer.Option("--collection", "-c", help="Filter by collection"),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option("--tags", "-t", help="Filter by tags (comma-separated)"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """List entities with optional filters."""
    import json

    store = _get_knowledge_store()
    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        entities = store.list_entities(
            type_key=type_key,
            collection=collection,
            tags=tag_list,
        )

        if format == "json":
            console.print(json.dumps([e.to_dict() for e in entities], indent=2))
        else:
            if not entities:
                console.print("[yellow]No entities found[/yellow]")
                return

            table = Table(title=f"Entities ({len(entities)})")
            table.add_column("ID", style="dim")
            table.add_column("Name", style="cyan")
            table.add_column("Type")
            table.add_column("Collection")
            table.add_column("Tags")

            for entity in entities:
                table.add_row(
                    str(entity.id)[:8] + "...",
                    entity.name,
                    entity.type_key,
                    entity.collection,
                    ", ".join(entity.tags) if entity.tags else "-",
                )

            console.print(table)
    finally:
        store.close()


@entity_app.command(name="show")
def entity_show(
    entity_id: Annotated[str, typer.Argument(help="Entity ID")],
    citations: Annotated[
        bool,
        typer.Option("--citations", "-C", help="Include citations"),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Show details of an entity."""
    import json
    import uuid

    store = _get_knowledge_store()
    try:
        if citations:
            entity, citation_data = store.get_entity_with_citations(uuid.UUID(entity_id))
        else:
            entity = store.get_entity(uuid.UUID(entity_id))
            citation_data = {}

        if not entity:
            console.print(f"[red]Entity not found: {entity_id}[/red]")
            raise typer.Exit(1)

        if format == "json":
            output = entity.to_dict()
            if citations:
                output["citations"] = {
                    field: [c.to_dict() for c in cites] for field, cites in citation_data.items()
                }
            console.print(json.dumps(output, indent=2))
        else:
            console.print(f"[bold]{entity.name}[/bold]")
            console.print(f"  ID: {entity.id}")
            console.print(f"  Type: {entity.type_key}")
            console.print(f"  Collection: {entity.collection}")
            console.print(f"  Tags: {', '.join(entity.tags) if entity.tags else '-'}")
            console.print()
            console.print("[bold]Profile:[/bold]")
            console.print(json.dumps(entity.profile, indent=2))

            if citations and citation_data:
                console.print()
                console.print("[bold]Citations:[/bold]")
                for field, cites in citation_data.items():
                    console.print(f"  {field}:")
                    for cite in cites:
                        console.print(f"    • {cite.to_markdown()}")
    except ValueError:
        console.print(f"[red]Invalid entity ID: {entity_id}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@entity_app.command(name="update")
def entity_update(
    entity_id: Annotated[str, typer.Argument(help="Entity ID")],
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="New profile as JSON string"),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option("--tags", "-t", help="New tags (comma-separated)"),
    ] = None,
) -> None:
    """Update an entity's profile and/or tags."""
    import json
    import uuid

    from toolhub.store.knowledge import EntityValidationError

    store = _get_knowledge_store()
    try:
        profile_dict = json.loads(profile) if profile else None
        tag_list = [t.strip() for t in tags.split(",")] if tags else None

        entity = store.update_entity(
            entity_id=uuid.UUID(entity_id),
            profile=profile_dict,
            tags=tag_list,
        )

        console.print(f"[green]✓ Updated entity: {entity.name}[/green]")
    except KeyError:
        console.print(f"[red]Entity not found: {entity_id}[/red]")
        raise typer.Exit(1)
    except ValueError:
        console.print(f"[red]Invalid entity ID: {entity_id}[/red]")
        raise typer.Exit(1)
    except EntityValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in profile: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@entity_app.command(name="delete")
def entity_delete(
    entity_id: Annotated[str, typer.Argument(help="Entity ID")],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation"),
    ] = False,
) -> None:
    """Delete an entity."""
    import uuid

    store = _get_knowledge_store()
    try:
        entity = store.get_entity(uuid.UUID(entity_id))
        if not entity:
            console.print(f"[red]Entity not found: {entity_id}[/red]")
            raise typer.Exit(1)

        if not force:
            confirm = typer.confirm(f"Delete entity '{entity.name}'?")
            if not confirm:
                raise typer.Abort()

        store.delete_entity(uuid.UUID(entity_id))
        console.print(f"[green]✓ Deleted entity: {entity.name}[/green]")
    except ValueError:
        console.print(f"[red]Invalid entity ID: {entity_id}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


# === Entity Type Commands ===


@entity_type_app.command(name="list")
def entity_type_list(
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """List all registered entity types."""
    import json

    store = _get_knowledge_store()
    try:
        entity_types = store.list_entity_types()

        if format == "json":
            console.print(json.dumps([et.to_dict() for et in entity_types], indent=2))
        else:
            if not entity_types:
                console.print("[yellow]No entity types registered[/yellow]")
                console.print("Register with: toolhub entity type register <type_key> '<schema>'")
                return

            table = Table(title=f"Entity Types ({len(entity_types)})")
            table.add_column("Type Key", style="cyan")
            table.add_column("Version")
            table.add_column("Description")

            for et in entity_types:
                table.add_row(
                    et.type_key,
                    str(et.schema_version),
                    et.description or "-",
                )

            console.print(table)
    finally:
        store.close()


@entity_type_app.command(name="show")
def entity_type_show(
    type_key: Annotated[str, typer.Argument(help="Entity type key (e.g., 'competitor')")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Show details of an entity type including its schema."""
    import json

    store = _get_knowledge_store()
    try:
        entity_type = store.get_entity_type(type_key)

        if not entity_type:
            console.print(f"[red]Entity type not found: {type_key}[/red]")
            raise typer.Exit(1)

        if format == "json":
            console.print(json.dumps(entity_type.to_dict(), indent=2))
        else:
            console.print(f"[bold]{entity_type.type_key}[/bold]")
            console.print(f"  Version: {entity_type.schema_version}")
            console.print(f"  Description: {entity_type.description or '-'}")
            console.print(f"  Created: {entity_type.created_at}")
            console.print(f"  Updated: {entity_type.updated_at}")
            console.print()
            console.print("[bold]Schema:[/bold]")
            console.print(json.dumps(entity_type.json_schema, indent=2))
    finally:
        store.close()


@entity_type_app.command(name="register")
def entity_type_register(
    type_key: Annotated[str, typer.Argument(help="Entity type key (e.g., 'competitor')")],
    json_schema: Annotated[str, typer.Argument(help="JSON Schema as string")],
    description: Annotated[
        str | None,
        typer.Option("--description", "-d", help="Description of the entity type"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Register a new entity type with a JSON Schema."""
    import json

    store = _get_knowledge_store()
    try:
        schema = json.loads(json_schema)
        entity_type = store.register_entity_type(
            type_key=type_key,
            json_schema=schema,
            description=description,
        )

        if format == "json":
            console.print(json.dumps(entity_type.to_dict(), indent=2))
        else:
            console.print(f"[green]✓ Registered entity type: {entity_type.type_key}[/green]")
            console.print(f"  Version: {entity_type.schema_version}")
            if entity_type.description:
                console.print(f"  Description: {entity_type.description}")
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON schema: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


# === Taxonomy Commands ===

taxonomy_app = typer.Typer(help="Manage feature taxonomies")
app.add_typer(taxonomy_app, name="taxonomy")


@taxonomy_app.command(name="create")
def taxonomy_create(
    name: Annotated[str, typer.Argument(help="Taxonomy name (e.g., 'k12-fundraising')")],
    domain: Annotated[str, typer.Argument(help="Domain label (e.g., 'K-12 Fundraising')")],
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection name"),
    ] = "default",
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Create a new feature taxonomy.

    Creates an empty taxonomy with the given name and domain.
    Use 'taxonomy add-group' and 'taxonomy add-feature' to populate it.

    Examples:
        toolhub taxonomy create k12-fundraising "K-12 Fundraising"
        toolhub taxonomy create saas-crm "SaaS CRM" --collection market-research
    """
    import json

    from toolhub.store.knowledge import EntityValidationError
    from toolhub.store.schemas import FEATURE_TAXONOMY_SCHEMA

    store = _get_knowledge_store()
    try:
        # Ensure feature_taxonomy type is registered
        if not store.get_entity_type("feature_taxonomy"):
            store.register_entity_type(
                "feature_taxonomy",
                FEATURE_TAXONOMY_SCHEMA,
                description="Grouped feature taxonomy for organizing features into categories",
            )

        # Create taxonomy entity with empty groups
        entity = store.create_entity(
            type_key="feature_taxonomy",
            name=name,
            profile={
                "domain": domain,
                "groups": [],
            },
            collection=collection,
        )

        if format == "json":
            console.print(json.dumps(entity.to_dict(), indent=2))
        else:
            console.print(f"[green]✓ Created taxonomy: {entity.name}[/green]")
            console.print(f"  ID: {entity.id}")
            console.print(f"  Domain: {domain}")
            console.print(f"  Collection: {entity.collection}")
            console.print()
            console.print("Next steps:")
            console.print(f"  toolhub taxonomy add-group {entity.id} <group-key> <group-label>")
    except EntityValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@taxonomy_app.command(name="list")
def taxonomy_list(
    collection: Annotated[
        str | None,
        typer.Option("--collection", "-c", help="Filter by collection"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """List all feature taxonomies.

    Examples:
        toolhub taxonomy list
        toolhub taxonomy list --collection market-research
        toolhub taxonomy list --format json
    """
    import json

    store = _get_knowledge_store()
    try:
        taxonomies = store.list_entities(type_key="feature_taxonomy", collection=collection)

        if format == "json":
            console.print(json.dumps([t.to_dict() for t in taxonomies], indent=2))
        else:
            if not taxonomies:
                console.print("[yellow]No taxonomies found[/yellow]")
                console.print("Create one with: toolhub taxonomy create <name> <domain>")
                return

            table = Table(title=f"Feature Taxonomies ({len(taxonomies)})")
            table.add_column("ID", style="dim")
            table.add_column("Name", style="cyan")
            table.add_column("Domain")
            table.add_column("Groups")
            table.add_column("Features")
            table.add_column("Collection")

            for taxonomy in taxonomies:
                groups = taxonomy.profile.get("groups", [])
                group_count = len(groups)
                feature_count = sum(len(g.get("features", [])) for g in groups)

                table.add_row(
                    str(taxonomy.id)[:8] + "...",
                    taxonomy.name,
                    taxonomy.profile.get("domain", "-"),
                    str(group_count),
                    str(feature_count),
                    taxonomy.collection,
                )

            console.print(table)
    finally:
        store.close()


@taxonomy_app.command(name="show")
def taxonomy_show(
    taxonomy_id: Annotated[str, typer.Argument(help="Taxonomy ID or name")],
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection (when using name lookup)"),
    ] = "default",
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Show details of a feature taxonomy.

    You can specify either the taxonomy ID or its name.

    Examples:
        toolhub taxonomy show k12-fundraising
        toolhub taxonomy show abc12345-...
        toolhub taxonomy show k12-fundraising --format json
    """
    import json
    import uuid

    store = _get_knowledge_store()
    try:
        # Try as UUID first, then as name
        taxonomy = None
        try:
            taxonomy = store.get_entity(uuid.UUID(taxonomy_id))
            if taxonomy and taxonomy.type_key != "feature_taxonomy":
                taxonomy = None
        except ValueError:
            # Not a UUID, try as name
            taxonomy = store.get_entity_by_name("feature_taxonomy", taxonomy_id, collection)

        if not taxonomy:
            console.print(f"[red]Taxonomy not found: {taxonomy_id}[/red]")
            raise typer.Exit(1)

        if format == "json":
            console.print(json.dumps(taxonomy.to_dict(), indent=2))
        else:
            console.print(f"[bold]{taxonomy.name}[/bold]")
            console.print(f"  ID: {taxonomy.id}")
            console.print(f"  Domain: {taxonomy.profile.get('domain', '-')}")
            console.print(f"  Collection: {taxonomy.collection}")
            console.print()

            groups = taxonomy.profile.get("groups", [])
            if not groups:
                console.print("[yellow]No groups defined yet[/yellow]")
                console.print(f"Add groups: toolhub taxonomy add-group {taxonomy.id} <key> <label>")
            else:
                console.print("[bold]Groups:[/bold]")
                for group in groups:
                    features = group.get("features", [])
                    console.print(f"  • {group['label']} ({group['key']})")
                    if features:
                        for feature in features:
                            console.print(f"      - {feature['label']} ({feature['key']})")
                    else:
                        console.print("      [dim](no features)[/dim]")
    finally:
        store.close()


def _get_taxonomy(store, taxonomy_id: str, collection: str = "default"):
    """Helper to get taxonomy by ID or name."""
    import uuid as uuid_mod

    taxonomy = None
    try:
        taxonomy = store.get_entity(uuid_mod.UUID(taxonomy_id))
        if taxonomy and taxonomy.type_key != "feature_taxonomy":
            taxonomy = None
    except ValueError:
        taxonomy = store.get_entity_by_name("feature_taxonomy", taxonomy_id, collection)
    return taxonomy


@taxonomy_app.command(name="add-group")
def taxonomy_add_group(
    taxonomy_id: Annotated[str, typer.Argument(help="Taxonomy ID or name")],
    group_key: Annotated[str, typer.Argument(help="Group key (e.g., 'donor-management')")],
    group_label: Annotated[str, typer.Argument(help="Group label (e.g., 'Donor Management')")],
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection (when using name lookup)"),
    ] = "default",
) -> None:
    """Add a group to a taxonomy.

    Examples:
        toolhub taxonomy add-group k12-fundraising donor-management "Donor Management"
        toolhub taxonomy add-group k12-fundraising online-giving "Online Giving"
    """
    import re

    from toolhub.store.knowledge import EntityValidationError

    if not re.match(r"^[a-z0-9-]+$", group_key):
        console.print(f"[red]Invalid group key: {group_key}[/red]")
        console.print("Keys must be lowercase alphanumeric with hyphens only")
        raise typer.Exit(1)

    store = _get_knowledge_store()
    try:
        taxonomy = _get_taxonomy(store, taxonomy_id, collection)
        if not taxonomy:
            console.print(f"[red]Taxonomy not found: {taxonomy_id}[/red]")
            raise typer.Exit(1)

        groups = taxonomy.profile.get("groups", [])

        # Check for duplicate key
        if any(g["key"] == group_key for g in groups):
            console.print(f"[red]Group key already exists: {group_key}[/red]")
            raise typer.Exit(1)

        # Add new group
        groups.append({
            "key": group_key,
            "label": group_label,
            "features": [],
        })

        store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        console.print(f"[green]✓ Added group: {group_label}[/green]")
        console.print(f"Add features: toolhub taxonomy add-feature {taxonomy.name} {group_key} ...")
    except EntityValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@taxonomy_app.command(name="add-feature")
def taxonomy_add_feature(
    taxonomy_id: Annotated[str, typer.Argument(help="Taxonomy ID or name")],
    group_key: Annotated[str, typer.Argument(help="Group key to add feature to")],
    feature_key: Annotated[str, typer.Argument(help="Feature key (e.g., 'donor-profiles')")],
    feature_label: Annotated[str, typer.Argument(help="Feature label (e.g., 'Donor Profiles')")],
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection (when using name lookup)"),
    ] = "default",
) -> None:
    """Add a feature to a taxonomy group.

    Examples:
        toolhub taxonomy add-feature k12-fundraising donor-management donor-profiles "..."
        toolhub taxonomy add-feature k12-fundraising online-giving donation-forms "..."
    """
    import re

    from toolhub.store.knowledge import EntityValidationError

    if not re.match(r"^[a-z0-9-]+$", feature_key):
        console.print(f"[red]Invalid feature key: {feature_key}[/red]")
        console.print("Keys must be lowercase alphanumeric with hyphens only")
        raise typer.Exit(1)

    store = _get_knowledge_store()
    try:
        taxonomy = _get_taxonomy(store, taxonomy_id, collection)
        if not taxonomy:
            console.print(f"[red]Taxonomy not found: {taxonomy_id}[/red]")
            raise typer.Exit(1)

        groups = taxonomy.profile.get("groups", [])

        # Find the group
        group_idx = None
        for i, g in enumerate(groups):
            if g["key"] == group_key:
                group_idx = i
                break

        if group_idx is None:
            console.print(f"[red]Group not found: {group_key}[/red]")
            raise typer.Exit(1)

        # Check for duplicate feature key across all groups
        for g in groups:
            if any(f["key"] == feature_key for f in g.get("features", [])):
                console.print(f"[red]Feature key already exists: {feature_key}[/red]")
                raise typer.Exit(1)

        # Add feature to group
        groups[group_idx]["features"].append({
            "key": feature_key,
            "label": feature_label,
        })

        store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        group_name = groups[group_idx]["label"]
        console.print(f"[green]✓ Added feature: {feature_label} to {group_name}[/green]")
    except EntityValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@taxonomy_app.command(name="rename-group")
def taxonomy_rename_group(
    taxonomy_id: Annotated[str, typer.Argument(help="Taxonomy ID or name")],
    old_key: Annotated[str, typer.Argument(help="Current group key")],
    new_key: Annotated[str, typer.Argument(help="New group key")],
    new_label: Annotated[
        str | None,
        typer.Option("--label", "-l", help="New label (optional)"),
    ] = None,
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection (when using name lookup)"),
    ] = "default",
) -> None:
    """Rename a group in a taxonomy.

    Group renames don't affect competitor data since features are keyed by feature key, not group.

    Examples:
        toolhub taxonomy rename-group k12-fundraising donor-mgt donor-management
        toolhub taxonomy rename-group k12-fundraising dm donor-management --label "Donor Management"
    """
    import re

    from toolhub.store.knowledge import EntityValidationError

    if not re.match(r"^[a-z0-9-]+$", new_key):
        console.print(f"[red]Invalid group key: {new_key}[/red]")
        console.print("Keys must be lowercase alphanumeric with hyphens only")
        raise typer.Exit(1)

    store = _get_knowledge_store()
    try:
        taxonomy = _get_taxonomy(store, taxonomy_id, collection)
        if not taxonomy:
            console.print(f"[red]Taxonomy not found: {taxonomy_id}[/red]")
            raise typer.Exit(1)

        groups = taxonomy.profile.get("groups", [])

        # Find the group
        group_idx = None
        for i, g in enumerate(groups):
            if g["key"] == old_key:
                group_idx = i
                break

        if group_idx is None:
            console.print(f"[red]Group not found: {old_key}[/red]")
            raise typer.Exit(1)

        # Check new key doesn't conflict
        if old_key != new_key and any(g["key"] == new_key for g in groups):
            console.print(f"[red]Group key already exists: {new_key}[/red]")
            raise typer.Exit(1)

        # Update group
        groups[group_idx]["key"] = new_key
        if new_label:
            groups[group_idx]["label"] = new_label

        store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        console.print(f"[green]✓ Renamed group: {old_key} → {new_key}[/green]")
    except EntityValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@taxonomy_app.command(name="rename-feature")
def taxonomy_rename_feature(
    taxonomy_id: Annotated[str, typer.Argument(help="Taxonomy ID or name")],
    old_key: Annotated[str, typer.Argument(help="Current feature key")],
    new_key: Annotated[str, typer.Argument(help="New feature key")],
    new_label: Annotated[
        str | None,
        typer.Option("--label", "-l", help="New label (optional)"),
    ] = None,
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection (when using name lookup)"),
    ] = "default",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be migrated without making changes"),
    ] = False,
) -> None:
    """Rename a feature in a taxonomy and migrate all references.

    This command updates:
    - The feature key in the taxonomy
    - All competitor.features entries using this key
    - All evidence.field_path entries referencing this feature

    Examples:
        toolhub taxonomy rename-feature k12-fundraising donor-prof donor-profiles
        toolhub taxonomy rename-feature k12-fundraising dp donor-profiles --label "Donor Profiles"
        toolhub taxonomy rename-feature k12-fundraising old-key new-key --dry-run
    """
    import re

    from toolhub.store.knowledge import EntityValidationError

    if not re.match(r"^[a-z0-9-]+$", new_key):
        console.print(f"[red]Invalid feature key: {new_key}[/red]")
        console.print("Keys must be lowercase alphanumeric with hyphens only")
        raise typer.Exit(1)

    store = _get_knowledge_store()
    try:
        taxonomy = _get_taxonomy(store, taxonomy_id, collection)
        if not taxonomy:
            console.print(f"[red]Taxonomy not found: {taxonomy_id}[/red]")
            raise typer.Exit(1)

        groups = taxonomy.profile.get("groups", [])

        # Find the feature
        feature_group_idx = None
        feature_idx = None
        for gi, g in enumerate(groups):
            for fi, f in enumerate(g.get("features", [])):
                if f["key"] == old_key:
                    feature_group_idx = gi
                    feature_idx = fi
                    break
            if feature_idx is not None:
                break

        if feature_idx is None:
            console.print(f"[red]Feature not found: {old_key}[/red]")
            raise typer.Exit(1)

        # Check new key doesn't conflict
        if old_key != new_key:
            for g in groups:
                if any(f["key"] == new_key for f in g.get("features", [])):
                    console.print(f"[red]Feature key already exists: {new_key}[/red]")
                    raise typer.Exit(1)

        # Find competitors with this feature
        competitors = store.list_entities(type_key="competitor")
        affected_competitors = []
        for comp in competitors:
            if old_key in comp.profile.get("features", {}):
                affected_competitors.append(comp)

        # Find evidence with this field_path
        old_field_path = f"features.{old_key}"
        new_field_path = f"features.{new_key}"
        affected_evidence = store.list_evidence(field_path=old_field_path)

        if dry_run:
            console.print("[bold]Dry run - no changes made[/bold]")
            console.print(f"\nWould rename: {old_key} → {new_key}")
            console.print(f"\nAffected competitors: {len(affected_competitors)}")
            for comp in affected_competitors:
                console.print(f"  • {comp.name}")
            console.print(f"\nAffected evidence records: {len(affected_evidence)}")
            return

        # Perform migration in transaction
        # 1. Update taxonomy
        groups[feature_group_idx]["features"][feature_idx]["key"] = new_key
        if new_label:
            groups[feature_group_idx]["features"][feature_idx]["label"] = new_label

        store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        # 2. Update competitor profiles
        for comp in affected_competitors:
            features = comp.profile.get("features", {})
            features[new_key] = features.pop(old_key)
            store.update_entity(comp.id, profile={**comp.profile, "features": features})

        # 3. Update evidence field_paths
        for ev in affected_evidence:
            store.update_evidence_field_path(ev.id, new_field_path)

        console.print(f"[green]✓ Renamed feature: {old_key} → {new_key}[/green]")
        console.print(f"  Updated {len(affected_competitors)} competitor(s)")
        console.print(f"  Updated {len(affected_evidence)} evidence record(s)")
    except EntityValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@taxonomy_app.command(name="move-feature")
def taxonomy_move_feature(
    taxonomy_id: Annotated[str, typer.Argument(help="Taxonomy ID or name")],
    feature_key: Annotated[str, typer.Argument(help="Feature key to move")],
    target_group: Annotated[str, typer.Argument(help="Target group key")],
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection (when using name lookup)"),
    ] = "default",
) -> None:
    """Move a feature to a different group.

    Moving features between groups doesn't affect competitor data since features
    are keyed by feature key, not by group. This is purely organizational.

    Examples:
        toolhub taxonomy move-feature k12-fundraising recurring-gifts online-giving
    """
    from toolhub.store.knowledge import EntityValidationError

    store = _get_knowledge_store()
    try:
        taxonomy = _get_taxonomy(store, taxonomy_id, collection)
        if not taxonomy:
            console.print(f"[red]Taxonomy not found: {taxonomy_id}[/red]")
            raise typer.Exit(1)

        groups = taxonomy.profile.get("groups", [])

        # Find the feature
        source_group_idx = None
        feature_idx = None
        feature_data = None
        for gi, g in enumerate(groups):
            for fi, f in enumerate(g.get("features", [])):
                if f["key"] == feature_key:
                    source_group_idx = gi
                    feature_idx = fi
                    feature_data = f
                    break
            if feature_idx is not None:
                break

        if feature_idx is None:
            console.print(f"[red]Feature not found: {feature_key}[/red]")
            raise typer.Exit(1)

        # Find target group
        target_group_idx = None
        for i, g in enumerate(groups):
            if g["key"] == target_group:
                target_group_idx = i
                break

        if target_group_idx is None:
            console.print(f"[red]Target group not found: {target_group}[/red]")
            raise typer.Exit(1)

        if source_group_idx == target_group_idx:
            console.print("[yellow]Feature is already in that group[/yellow]")
            return

        # Move feature
        source_group_name = groups[source_group_idx]["label"]
        target_group_name = groups[target_group_idx]["label"]

        groups[source_group_idx]["features"].pop(feature_idx)
        groups[target_group_idx]["features"].append(feature_data)

        store.update_entity(
            taxonomy.id,
            profile={**taxonomy.profile, "groups": groups},
        )

        console.print(f"[green]✓ Moved {feature_key}[/green]")
        console.print(f"  From: {source_group_name} → {target_group_name}")
    except EntityValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


# === Evidence Commands ===


@evidence_app.command(name="add")
def evidence_add(
    entity_id: Annotated[str, typer.Argument(help="Entity ID")],
    field_path: Annotated[str, typer.Argument(help="Field path (e.g., 'funding.total')")],
    chunk_id: Annotated[
        str | None,
        typer.Option("--chunk", "-k", help="Chunk ID for evidence"),
    ] = None,
    source_id: Annotated[
        str | None,
        typer.Option("--source", "-s", help="Source ID (if no chunk)"),
    ] = None,
    quote: Annotated[
        str | None,
        typer.Option("--quote", "-q", help="Quote from source"),
    ] = None,
    confidence: Annotated[
        float | None,
        typer.Option("--confidence", "-c", help="Confidence score (0-1)"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Add evidence for an entity field."""
    import json
    import uuid

    store = _get_knowledge_store()
    try:
        evidence = store.add_evidence(
            entity_id=uuid.UUID(entity_id),
            field_path=field_path,
            chunk_id=uuid.UUID(chunk_id) if chunk_id else None,
            source_id=uuid.UUID(source_id) if source_id else None,
            quote=quote,
            confidence=confidence,
        )

        if format == "json":
            console.print(json.dumps(evidence.to_dict(), indent=2))
        else:
            console.print(f"[green]✓ Added evidence for {field_path}[/green]")
            console.print(f"  ID: {evidence.id}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@evidence_app.command(name="list")
def evidence_list(
    entity_id: Annotated[
        str | None,
        typer.Option("--entity", "-e", help="Filter by entity ID"),
    ] = None,
    field_path: Annotated[
        str | None,
        typer.Option("--field", "-F", help="Filter by field path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """List evidence with optional filters."""
    import json
    import uuid

    store = _get_knowledge_store()
    try:
        entity_uuid = uuid.UUID(entity_id) if entity_id else None
        evidence_list_result = store.list_evidence(
            entity_id=entity_uuid,
            field_path=field_path,
        )

        if format == "json":
            console.print(json.dumps([e.to_dict() for e in evidence_list_result], indent=2))
        else:
            if not evidence_list_result:
                console.print("[yellow]No evidence found[/yellow]")
                return

            table = Table(title=f"Evidence ({len(evidence_list_result)})")
            table.add_column("ID", style="dim")
            table.add_column("Field Path")
            table.add_column("Quote")
            table.add_column("Confidence")

            for evidence in evidence_list_result:
                quote_preview = evidence.quote or "-"
                if evidence.quote and len(evidence.quote) > 30:
                    quote_preview = evidence.quote[:30] + "..."
                conf_str = f"{evidence.confidence:.0%}" if evidence.confidence else "-"
                table.add_row(
                    str(evidence.id)[:8] + "...",
                    evidence.field_path,
                    quote_preview,
                    conf_str,
                )

            console.print(table)
    except ValueError:
        console.print(f"[red]Invalid entity ID: {entity_id}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


@evidence_app.command(name="show")
def evidence_show(
    evidence_id: Annotated[str, typer.Argument(help="Evidence ID")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Show details of evidence."""
    import json
    import uuid

    store = _get_knowledge_store()
    try:
        evidence = store.get_evidence(uuid.UUID(evidence_id))
        if not evidence:
            console.print(f"[red]Evidence not found: {evidence_id}[/red]")
            raise typer.Exit(1)

        if format == "json":
            console.print(json.dumps(evidence.to_dict(), indent=2))
        else:
            console.print(f"[bold]Evidence: {evidence.id}[/bold]")
            console.print(f"  Entity ID: {evidence.entity_id}")
            console.print(f"  Field Path: {evidence.field_path}")
            if evidence.quote:
                console.print(f'  Quote: "{evidence.quote}"')
            if evidence.confidence:
                console.print(f"  Confidence: {evidence.confidence:.0%}")
            if evidence.chunk_id:
                console.print(f"  Chunk ID: {evidence.chunk_id}")
            if evidence.source_id:
                console.print(f"  Source ID: {evidence.source_id}")
    except ValueError:
        console.print(f"[red]Invalid evidence ID: {evidence_id}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


# === Report Commands ===


@report_app.command(name="list")
def report_list() -> None:
    """List available report types."""
    from toolhub.store.reports import REPORT_TYPES

    console.print("[bold]Available Reports:[/bold]")
    for report_type in sorted(REPORT_TYPES.keys()):
        console.print(f"  • {report_type}")


@report_app.command(name="generate")
def report_generate(
    report_type: Annotated[
        str,
        typer.Argument(help="Report type (e.g., 'competitor-feature-matrix')"),
    ],
    collection: Annotated[
        str | None,
        typer.Option("--collection", "-c", help="Filter by collection"),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option("--tags", "-t", help="Filter by tags (comma-separated)"),
    ] = None,
    entity_ids: Annotated[
        str | None,
        typer.Option("--entities", "-e", help="Entity IDs (comma-separated)"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or markdown"),
    ] = "markdown",
) -> None:
    """Generate a report."""
    from toolhub.store.reports import get_report

    store = _get_knowledge_store()
    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        entity_id_list = [e.strip() for e in entity_ids.split(",")] if entity_ids else None

        report = get_report(report_type, store)
        result = report.generate(
            entity_ids=entity_id_list,
            tags=tag_list,
            collection=collection,
        )

        if format == "json":
            console.print(result.to_json())
        else:
            # Use custom to_markdown if available
            if hasattr(report, "to_markdown"):
                console.print(report.to_markdown(result))
            else:
                console.print(result.to_markdown())
    except KeyError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


# === Source Commands ===


@source_app.command(name="add")
def source_add(
    url: Annotated[str, typer.Argument(help="Source URL")],
    source_type: Annotated[
        str,
        typer.Option("--type", "-t", help="Source type (help_docs, marketing, review)"),
    ] = "website",
    collection: Annotated[
        str,
        typer.Option("--collection", "-c", help="Collection name"),
    ] = "default",
    tags: Annotated[
        str | None,
        typer.Option("--tags", help="Tags (comma-separated)"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Add a documentation source."""
    import json

    store = _get_knowledge_store()
    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        source = store.add_source(
            canonical_url=url,
            source_type=source_type,
            collection=collection,
            tags=tag_list,
        )

        if format == "json":
            console.print(json.dumps(source.to_dict(), indent=2))
        else:
            console.print("[green]✓ Added source[/green]")
            console.print(f"  ID: {source.id}")
            console.print(f"  URL: {source.canonical_url}")
            console.print(f"  Type: {source.source_type}")
            console.print(f"  Collection: {source.collection}")
    finally:
        store.close()


@source_app.command(name="list")
def source_list(
    collection: Annotated[
        str | None,
        typer.Option("--collection", "-c", help="Filter by collection"),
    ] = None,
    source_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Filter by source type"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """List documentation sources."""
    import json

    store = _get_knowledge_store()
    try:
        sources = store.list_sources(collection=collection, source_type=source_type)

        if format == "json":
            console.print(json.dumps([s.to_dict() for s in sources], indent=2))
        else:
            if not sources:
                console.print("[dim]No sources found[/dim]")
                return

            table = Table(title="Sources")
            table.add_column("ID", style="dim")
            table.add_column("URL")
            table.add_column("Type")
            table.add_column("Collection")
            table.add_column("Status")

            for source in sources:
                table.add_row(
                    str(source.id)[:8],
                    source.canonical_url[:50] + ("..." if len(source.canonical_url) > 50 else ""),
                    source.source_type,
                    source.collection,
                    source.status.value,
                )

            console.print(table)
    finally:
        store.close()


@source_app.command(name="show")
def source_show(
    source_id: Annotated[str, typer.Argument(help="Source ID")],
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Show details of a source."""
    import json
    import uuid

    store = _get_knowledge_store()
    try:
        source = store.get_source(uuid.UUID(source_id))
        if not source:
            console.print(f"[red]Source not found: {source_id}[/red]")
            raise typer.Exit(1)

        if format == "json":
            console.print(json.dumps(source.to_dict(), indent=2))
        else:
            console.print(f"[bold]{source.canonical_url}[/bold]")
            console.print(f"  ID: {source.id}")
            console.print(f"  Type: {source.source_type}")
            console.print(f"  Collection: {source.collection}")
            console.print(f"  Status: {source.status.value}")
            console.print(f"  Tags: {', '.join(source.tags) if source.tags else 'none'}")
            console.print(f"  Created: {source.created_at}")
    except ValueError:
        console.print(f"[red]Invalid UUID: {source_id}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()


# === Knowledge Query Command ===


@app.command(name="query")
def knowledge_query(
    query: Annotated[str, typer.Argument(help="Search query")],
    collection: Annotated[
        str | None,
        typer.Option("--collection", "-c", help="Filter by collection"),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option("--tags", "-t", help="Filter by tags (comma-separated)"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum results"),
    ] = 10,
    min_similarity: Annotated[
        float,
        typer.Option("--min-sim", help="Minimum similarity score"),
    ] = 0.0,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or markdown"),
    ] = "markdown",
) -> None:
    """Search the knowledge store (Postgres/pgvector backend)."""
    import json

    store = _get_knowledge_store()
    config = load_config()
    try:
        tag_list = [t.strip() for t in tags.split(",")] if tags else None
        timings: dict[str, float] = {}

        response = store.search_text(
            query=query,
            collection=collection,
            tags=tag_list,
            limit=limit,
            min_similarity=min_similarity,
            model_name=config.embedding.model,
            timings=timings,
        )

        if format == "json":
            console.print(json.dumps(response.to_dict(), indent=2))
        else:
            console.print(response.to_markdown())
    finally:
        store.close()


# === Database Commands ===


@db_app.command(name="seed")
def db_seed(
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Register built-in entity type schemas (feature_taxonomy, competitor, etc.).

    This command registers the canonical JSON schemas for built-in entity types.
    Safe to run multiple times - existing types are updated with latest schema.

    Examples:
        toolhub db seed
        toolhub db seed --format json
    """
    import json

    from toolhub.store.schemas import BUILTIN_SCHEMAS, register_builtin_schemas

    store = _get_knowledge_store()
    try:
        registered = register_builtin_schemas(store)

        if format == "json":
            result = {
                "registered": registered,
                "schemas": {k: v["description"] for k, v in BUILTIN_SCHEMAS.items()},
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[green]✓ Registered {len(registered)} entity types:[/green]")
            for type_key in registered:
                desc = BUILTIN_SCHEMAS[type_key]["description"]
                console.print(f"  • {type_key}: {desc}")
    finally:
        store.close()


@db_app.command(name="migrate")
def db_migrate(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show migrations without applying"),
    ] = False,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Apply pending database migrations.

    Migrations are numbered SQL files in the migrations/ directory.
    Each migration is applied once and tracked in schema_migrations table.

    Examples:
        toolhub db migrate              # Apply pending migrations
        toolhub db migrate --dry-run    # Preview what would be applied
    """
    import json
    from pathlib import Path

    config = load_config()
    migrations_dir = Path(__file__).parent.parent.parent / "migrations"

    if not migrations_dir.exists():
        console.print(f"[yellow]No migrations directory found at {migrations_dir}[/yellow]")
        console.print("Create migrations/ with numbered SQL files (e.g., 001_initial.sql)")
        return

    # Get migration files sorted by number
    migration_files = sorted(migrations_dir.glob("*.sql"))
    if not migration_files:
        console.print("[yellow]No migration files found[/yellow]")
        return

    import psycopg

    conn = psycopg.connect(config.postgres.url)
    try:
        cur = conn.cursor()

        # Ensure schema_migrations table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.commit()

        # Get applied migrations
        cur.execute("SELECT version FROM schema_migrations ORDER BY version")
        applied = {row[0] for row in cur.fetchall()}

        pending = [f for f in migration_files if f.name not in applied]

        if not pending:
            if format == "json":
                console.print(json.dumps({"status": "up-to-date", "applied": len(applied)}))
            else:
                msg = f"[green]✓ Database is up to date ({len(applied)} migrations applied)[/green]"
                console.print(msg)
            return

        if dry_run:
            if format == "json":
                console.print(json.dumps({"pending": [f.name for f in pending]}))
            else:
                console.print(f"[yellow]Pending migrations ({len(pending)}):[/yellow]")
                for f in pending:
                    console.print(f"  • {f.name}")
            return

        # Apply pending migrations
        applied_now = []
        for migration_file in pending:
            sql = migration_file.read_text()
            try:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (migration_file.name,),
                )
                conn.commit()
                applied_now.append(migration_file.name)
                console.print(f"[green]✓ Applied: {migration_file.name}[/green]")
            except Exception as e:
                conn.rollback()
                console.print(f"[red]✗ Failed: {migration_file.name}[/red]")
                console.print(f"[red]  Error: {e}[/red]")
                raise typer.Exit(1)

        if format == "json":
            console.print(json.dumps({"applied": applied_now}))
        else:
            console.print(f"\n[green]✓ Applied {len(applied_now)} migrations[/green]")

    finally:
        conn.close()


@db_app.command(name="status")
def db_status(
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: json or table"),
    ] = "table",
) -> None:
    """Show database migration status.

    Examples:
        toolhub db status
        toolhub db status --format json
    """
    import json
    from pathlib import Path

    config = load_config()
    migrations_dir = Path(__file__).parent.parent.parent / "migrations"

    import psycopg

    try:
        conn = psycopg.connect(config.postgres.url)
    except Exception as e:
        console.print(f"[red]Cannot connect to database: {e}[/red]")
        raise typer.Exit(1)

    try:
        cur = conn.cursor()

        # Check if schema_migrations exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'schema_migrations'
            )
        """)
        has_migrations_table = cur.fetchone()[0]

        applied = set()
        if has_migrations_table:
            cur.execute("SELECT version FROM schema_migrations ORDER BY version")
            applied = {row[0] for row in cur.fetchall()}

        # Get migration files
        migration_files = []
        if migrations_dir.exists():
            migration_files = sorted(migrations_dir.glob("*.sql"))

        pending = [f.name for f in migration_files if f.name not in applied]

        if format == "json":
            result = {
                "applied": sorted(applied),
                "pending": pending,
                "total_files": len(migration_files),
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print("[bold]Migration Status[/bold]")
            console.print(f"  Applied: {len(applied)}")
            console.print(f"  Pending: {len(pending)}")
            if pending:
                console.print("\n[yellow]Pending migrations:[/yellow]")
                for name in pending:
                    console.print(f"  • {name}")
            else:
                console.print("\n[green]✓ Database is up to date[/green]")

    finally:
        conn.close()


if __name__ == "__main__":
    app()
