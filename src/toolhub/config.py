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
class PostgresConfig:
    """Postgres database configuration.

    Note: pool_size and pool_timeout are defined for future use but
    connection pooling is not yet implemented. KnowledgeStore currently
    creates a new connection per instance.
    """

    url: str = "postgresql://localhost:5432/toolhub"
    pool_size: int = 5  # Not yet implemented
    pool_timeout: int = 30  # Not yet implemented


@dataclass
class S3Config:
    """S3/MinIO artifact storage configuration.

    SECURITY WARNING: The default access_key and secret_key are MinIO's
    well-known development credentials. These MUST be changed in production.
    Use environment variables or a secrets manager for production deployments.
    """

    endpoint_url: str = "http://localhost:9000"
    bucket: str = "toolhub"
    access_key: str = "minioadmin"  # WARNING: Change in production!
    secret_key: str = "minioadmin"  # WARNING: Change in production!
    region: str = "us-east-1"
    use_ssl: bool = False


@dataclass
class Config:
    """Root configuration object."""

    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    crawling: CrawlingConfig = field(default_factory=CrawlingConfig)
    updates: UpdateConfig = field(default_factory=UpdateConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    s3: S3Config = field(default_factory=S3Config)

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
            "postgres": {
                "url": self.postgres.url,
                "pool_size": self.postgres.pool_size,
                "pool_timeout": self.postgres.pool_timeout,
            },
            "s3": {
                "endpoint_url": self.s3.endpoint_url,
                "bucket": self.s3.bucket,
                "access_key": self.s3.access_key,
                "secret_key": self.s3.secret_key,
                "region": self.s3.region,
                "use_ssl": self.s3.use_ssl,
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
        postgres_data = data.get("postgres", {})
        s3_data = data.get("s3", {})

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
            postgres=PostgresConfig(
                url=postgres_data.get("url", "postgresql://localhost:5432/toolhub"),
                pool_size=postgres_data.get("pool_size", 5),
                pool_timeout=postgres_data.get("pool_timeout", 30),
            ),
            s3=S3Config(
                endpoint_url=s3_data.get("endpoint_url", "http://localhost:9000"),
                bucket=s3_data.get("bucket", "toolhub"),
                access_key=s3_data.get("access_key", "minioadmin"),
                secret_key=s3_data.get("secret_key", "minioadmin"),
                region=s3_data.get("region", "us-east-1"),
                use_ssl=s3_data.get("use_ssl", False),
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
