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
from toolhub.store import OutputFormat, VectorStore, search
from toolhub.store.lance import delete_tool_store, list_tool_stores
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

        # Store
        progress.update(task, description="Indexing in vector store...")
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
    """Search directly without daemon."""
    import time

    t0 = time.perf_counter()
    config = load_config()
    if timings is not None:
        timings["config_load"] = time.perf_counter() - t0

    response = search(
        query,
        tool_ids=tool_ids,
        limit=limit,
        model_name=config.embedding.model,
        timings=timings,
    )
    if timings is not None:
        timings["via"] = "direct"

    return {
        "query": response.query,
        "tools_searched": response.tools_searched,
        "result_count": len(response.results),
        "results": [r.to_dict() for r in response.results],
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
                console.print(f"[dim]  {key}: {val*1000:.1f}ms[/dim]")


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

        # Remove vector store
        delete_tool_store(tool_id)

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
    stores = list_tool_stores()
    config = load_config()

    # Build status table
    table = Table(title="toolhub status", show_header=False, box=None)
    table.add_column("Key", style="dim")
    table.add_column("Value")

    table.add_row("Version", __version__)
    table.add_row("Tools indexed", str(len(registry.tools)))
    table.add_row("Vector stores", str(len(stores)))

    total_chunks = sum(t.total_chunks for t in registry.tools.values())
    table.add_row("Total chunks", str(total_chunks))

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


if __name__ == "__main__":
    app()
