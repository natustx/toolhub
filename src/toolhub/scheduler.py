"""Background update scheduler for automatic re-crawling of documentation sources."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime, timedelta

from toolhub.config import load_config
from toolhub.registry import Source, Tool, load_registry, save_registry

logger = logging.getLogger(__name__)


def get_stale_sources(max_age_hours: int) -> list[tuple[Tool, Source]]:
    """Find sources that haven't been updated within max_age_hours.

    Args:
        max_age_hours: Maximum age in hours before a source is considered stale

    Returns:
        List of (tool, source) tuples that need updating
    """
    registry = load_registry()
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    stale = []

    for tool in registry.tools.values():
        for source in tool.sources:
            if source.indexed_at is None or source.indexed_at < cutoff:
                stale.append((tool, source))

    return stale


def update_source(
    tool: Tool,
    source: Source,
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """Re-index a single source.

    Args:
        tool: The tool containing the source
        source: The source to update
        on_progress: Optional callback for progress updates

    Returns:
        True if update succeeded, False otherwise
    """
    from toolhub.crawler import GitHubCrawler, LlmsTxtCrawler, WebsiteCrawler
    from toolhub.indexer import chunk_directory, embed_chunks
    from toolhub.paths import get_source_cache_dir
    from toolhub.store import VectorStore

    config = load_config()

    if on_progress:
        on_progress(f"Updating {tool.display_name} from {source.url}")

    # Get appropriate crawler
    if source.source_type == "github":
        crawler = GitHubCrawler(github_token=config.crawling.github_token or None)
    elif source.source_type == "llmstxt":
        crawler = LlmsTxtCrawler()
    elif source.source_type == "website":
        crawler = WebsiteCrawler()
    else:
        logger.warning(f"Unknown source type: {source.source_type}")
        return False

    try:
        # Crawl to per-source cache directory
        cache_dir = get_source_cache_dir(tool.tool_id, source.url)
        result = crawler.crawl(source.url, cache_dir)

        if not result.success:
            logger.error(f"Crawl failed for {source.url}: {result.error}")
            return False

        file_count = len(result.files_crawled)
        if on_progress:
            on_progress(f"  Found {file_count} files")

        # Chunk and embed
        chunks = chunk_directory(cache_dir, max_tokens=config.embedding.chunk_size_tokens)
        if not chunks:
            logger.warning(f"No chunks generated for {source.url}")
            return False

        if on_progress:
            on_progress(f"  Created {len(chunks)} chunks")

        embedded = embed_chunks(chunks, model_name=config.embedding.model)

        # Store
        store = VectorStore(tool.tool_id)
        store.clear()
        store.add_chunks(embedded)

        # Update registry
        registry = load_registry()
        source.indexed_at = datetime.now()
        source.chunk_count = len(embedded)
        source.file_count = file_count
        save_registry(registry)

        if on_progress:
            on_progress(f"  ✓ Updated {tool.display_name}")

        return True

    except Exception as e:
        logger.exception(f"Failed to update {source.url}: {e}")
        return False


def check_and_update_stale(on_progress: Callable[[str], None] | None = None) -> int:
    """Check for stale sources and update them.

    Args:
        on_progress: Optional callback for progress updates

    Returns:
        Number of sources updated
    """
    config = load_config()
    stale = get_stale_sources(config.updates.max_age_hours)

    if not stale:
        if on_progress:
            on_progress("All sources are up to date")
        return 0

    if on_progress:
        on_progress(f"Found {len(stale)} stale source(s)")

    updated = 0
    for tool, source in stale:
        if update_source(tool, source, on_progress):
            updated += 1

    return updated


class UpdateScheduler:
    """Background scheduler for automatic documentation updates."""

    def __init__(self, interval_hours: int = 24):
        """Initialize scheduler.

        Args:
            interval_hours: How often to check for updates (in hours)
        """
        self.interval_hours = interval_hours
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True
        logger.info(f"Update scheduler started (interval: {self.interval_hours}h)")

    def stop(self) -> None:
        """Stop the background scheduler."""
        if not self._running:
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        logger.info("Update scheduler stopped")

    def _run(self) -> None:
        """Background thread main loop."""
        interval_seconds = self.interval_hours * 3600

        while not self._stop_event.is_set():
            try:
                updated = check_and_update_stale(on_progress=lambda msg: logger.info(msg))
                if updated:
                    logger.info(f"Updated {updated} stale source(s)")
            except Exception as e:
                logger.exception(f"Error in update scheduler: {e}")

            # Wait for interval or stop signal
            self._stop_event.wait(interval_seconds)

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Global scheduler instance (created on demand)
_scheduler: UpdateScheduler | None = None


def get_scheduler() -> UpdateScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        config = load_config()
        _scheduler = UpdateScheduler(interval_hours=config.updates.interval_hours)
    return _scheduler


def start_background_updates() -> None:
    """Start background update scheduler if enabled in config."""
    config = load_config()
    if config.updates.enabled:
        scheduler = get_scheduler()
        scheduler.start()


def stop_background_updates() -> None:
    """Stop background update scheduler."""
    if _scheduler is not None:
        _scheduler.stop()
