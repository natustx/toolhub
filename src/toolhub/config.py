"""Configuration management for toolhub.

Handles loading and saving config.toml with sensible defaults.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from toolhub.paths import ensure_directories, get_config_path


@dataclass
class DaemonConfig:
    """Daemon server configuration."""

    host: str = "127.0.0.1"
    port: int = 9742


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = "all-MiniLM-L6-v2"
    chunk_size_tokens: int = 500


@dataclass
class SearchConfig:
    """Search defaults."""

    default_limit: int = 5


@dataclass
class CrawlingConfig:
    """Crawling configuration."""

    github_token: str = ""


@dataclass
class UpdateConfig:
    """Background update configuration."""

    enabled: bool = False
    interval_hours: int = 24  # How often to check for updates
    max_age_hours: int = 168  # Re-index if older than this (1 week default)


@dataclass
class Config:
    """Root configuration object."""

    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    crawling: CrawlingConfig = field(default_factory=CrawlingConfig)
    updates: UpdateConfig = field(default_factory=UpdateConfig)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "daemon": {
                "host": self.daemon.host,
                "port": self.daemon.port,
            },
            "embedding": {
                "model": self.embedding.model,
                "chunk_size_tokens": self.embedding.chunk_size_tokens,
            },
            "search": {
                "default_limit": self.search.default_limit,
            },
            "crawling": {
                "github_token": self.crawling.github_token,
            },
            "updates": {
                "enabled": self.updates.enabled,
                "interval_hours": self.updates.interval_hours,
                "max_age_hours": self.updates.max_age_hours,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """Create config from dictionary."""
        daemon_data = data.get("daemon", {})
        embedding_data = data.get("embedding", {})
        search_data = data.get("search", {})
        crawling_data = data.get("crawling", {})
        updates_data = data.get("updates", {})

        return cls(
            daemon=DaemonConfig(
                host=daemon_data.get("host", "127.0.0.1"),
                port=daemon_data.get("port", 9742),
            ),
            embedding=EmbeddingConfig(
                model=embedding_data.get("model", "all-MiniLM-L6-v2"),
                chunk_size_tokens=embedding_data.get("chunk_size_tokens", 500),
            ),
            search=SearchConfig(
                default_limit=search_data.get("default_limit", 5),
            ),
            crawling=CrawlingConfig(
                github_token=crawling_data.get("github_token", ""),
            ),
            updates=UpdateConfig(
                enabled=updates_data.get("enabled", False),
                interval_hours=updates_data.get("interval_hours", 24),
                max_age_hours=updates_data.get("max_age_hours", 168),
            ),
        )


def load_config(path: Path | None = None) -> Config:
    """Load configuration from file.

    If file doesn't exist, returns default config.
    """
    config_path = path or get_config_path()

    if not config_path.exists():
        return Config()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    return Config.from_dict(data)


def save_config(config: Config, path: Path | None = None) -> None:
    """Save configuration to file.

    Creates directories if needed.
    """
    ensure_directories()
    config_path = path or get_config_path()

    with open(config_path, "wb") as f:
        tomli_w.dump(config.to_dict(), f)


def ensure_config() -> Config:
    """Load config, creating default file if it doesn't exist."""
    config_path = get_config_path()

    if not config_path.exists():
        config = Config()
        save_config(config)
        return config

    return load_config()
